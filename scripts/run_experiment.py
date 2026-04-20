"""
run_experiment.py — Main training and evaluation script.

Runs REINFORCE training on a specific task + reward function combination,
with periodic evaluation and reward hacking detection.

Usage:
    # Seed the database first (if not done):
    python scripts/seed_db.py

    # Train REINFORCE on Task 1 with composite reward:
    python scripts/run_experiment.py --task task_01_simple --reward composite

    # Train on all tasks with all reward functions:
    python scripts/run_experiment.py --all

    # Dry-run (evaluate random + rule agents only, no training):
    python scripts/run_experiment.py --dry_run

Author: Kartik Munjal
"""

import argparse
import json
import re
import sys
import time
from pathlib import Path
from typing import Optional

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import yaml

from src.env.sql_env import SQLQueryEnv
from src.tasks.base import load_task_registry
from src.agents import RandomAgent, RuleAgent, REINFORCEAgent
from src.rewards import get_reward_fn
from src.analysis.reward_hacking_detector import RewardHackingDetector


def _extract_detector_stats(sql: str, reward_components: dict) -> dict:
    """Extract the detector inputs from the final SQL and reward components."""
    op_m = re.search(r"\bWHERE\s+\S+\s*(=|!=|>=|<=|>|<|LIKE|IN|NOT\s+IN)\b", sql, re.IGNORECASE)
    where_op = op_m.group(1).upper().replace("  ", " ") if op_m else "NONE"

    sel_m = re.search(r"\bSELECT\s+(.*?)\s+FROM\b", sql, re.IGNORECASE | re.DOTALL)
    if sel_m:
        select_cols = [c.strip() for c in sel_m.group(1).split(",") if c.strip()]
        select_col_count = len(select_cols)
    else:
        select_col_count = 0

    predicted_rows = int(
        reward_components.get(
            "exec_pred_rows_no_limit",
            reward_components.get("exec_pred_rows", 0),
        )
    )

    return {
        "predicted_rows": predicted_rows,
        "where_operator": where_op,
        "select_col_count": select_col_count,
    }


def _build_trace_entry(
    episode: int,
    task_id: str,
    reward_name: str,
    final_info: dict,
    ep_reward: float,
) -> dict:
    """Persist enough per-episode context for offline detector calibration."""
    rc = final_info.get("reward_components", {})
    sql = final_info.get("sql_so_far", "")
    detector_stats = _extract_detector_stats(sql, rc)
    return {
        "episode": episode,
        "task_id": task_id,
        "reward_name": reward_name,
        "nl_query": final_info.get("nl_query", ""),
        "sql": sql,
        "reward": ep_reward,
        "r_exec": rc.get("r_exec", rc.get("execution_match", 0.0)),
        "r_partial": rc.get("r_partial", rc.get("partial_credit", 0.0)),
        "r_exact": rc.get("r_exact", rc.get("exact_match", 0.0)),
        "exec_reason": rc.get("exec_reason", rc.get("reason", "")),
        "exec_pred_rows": rc.get("exec_pred_rows", rc.get("pred_rows", 0)),
        "exec_pred_rows_no_limit": rc.get("exec_pred_rows_no_limit", rc.get("pred_rows_no_limit", 0)),
        "exec_gold_rows": rc.get("exec_gold_rows", rc.get("gold_rows", 0)),
        "efficiency_penalty": rc.get("efficiency_penalty", 0.0),
        **detector_stats,
    }


def run_episodes(
    env: SQLQueryEnv,
    agent,
    n_episodes: int,
    training: bool = False,
    detector: RewardHackingDetector = None,
    detector_trace: Optional[list[dict]] = None,
    task_id: str = "",
    reward_name: str = "",
    start_episode: int = 0,
) -> dict:
    """Run n_episodes and return aggregate metrics."""
    rewards = []
    exec_matches = []
    partial_credits = []
    sql_samples = []

    for ep in range(n_episodes):
        obs, info = env.reset()

        if hasattr(agent, "set_nl_query"):
            agent.set_nl_query(info["nl_query"])
        agent.reset_episode()

        terminated = truncated = False
        ep_reward = 0.0
        final_info = info

        while not terminated and not truncated:
            action = agent.act(obs, info["action_mask"])
            obs, reward, terminated, truncated, info = env.step(action)
            ep_reward = reward   # terminal reward only
            final_info = info

        if hasattr(agent, "on_episode_end"):
            agent.on_episode_end(ep_reward, final_info)

        rewards.append(ep_reward)
        rc = final_info.get("reward_components", {})
        exec_matches.append(rc.get("r_exec", rc.get("execution_match", 0.0)))
        partial_credits.append(rc.get("r_partial", rc.get("partial_credit", 0.0)))

        global_episode = start_episode + ep + 1

        if ep < 5 or ep % 100 == 0:
            sql_samples.append({
                "episode": global_episode,
                "nl": final_info.get("nl_query", ""),
                "sql": final_info.get("sql_so_far", ""),
                "reward": ep_reward,
            })

        trace_entry = _build_trace_entry(
            episode=global_episode,
            task_id=task_id,
            reward_name=reward_name,
            final_info=final_info,
            ep_reward=ep_reward,
        )

        if detector:
            detector.update({
                "predicted_rows": trace_entry["predicted_rows"],
                "where_operator": trace_entry["where_operator"],
                "select_col_count": trace_entry["select_col_count"],
                "assembled_sql": trace_entry["sql"],
            })

        if detector_trace is not None:
            detector_trace.append(trace_entry)

    return {
        "mean_reward": float(np.mean(rewards)),
        "std_reward": float(np.std(rewards)),
        "mean_exec_match": float(np.mean(exec_matches)),
        "mean_partial": float(np.mean(partial_credits)),
        "n_episodes": n_episodes,
        "sql_samples": sql_samples,
    }


def train_reinforce(
    task_id: str,
    reward_name: str,
    project_root: Path,
    max_episodes: int = 2000,
    eval_every: int = 100,
    eval_episodes: int = 100,
    save_detector_trace: bool = False,
    verbose: bool = True,
) -> dict:
    cfg_path = str(project_root / "configs/env_config.yaml")
    env_cfg = yaml.safe_load((project_root / "configs/env_config.yaml").read_text())
    reward_cfg = yaml.safe_load((project_root / "configs/reward_config.yaml").read_text())

    db_path = str(project_root / env_cfg["db_path"])
    task_registry = load_task_registry(
        str(project_root / "configs/task_config.yaml"), str(project_root)
    )
    reward_fn = get_reward_fn(reward_name, reward_cfg, db_path)

    env = SQLQueryEnv(
        db_path=db_path,
        task_registry=task_registry,
        reward_fn=reward_fn,
        nl_vocab_path=str(project_root / "configs/nl_vocab.json"),
        seed=env_cfg.get("seed", 42),
        task_id=task_id,
    )

    agent = REINFORCEAgent.from_config(str(project_root / "configs/agent_config.yaml"))
    detector = RewardHackingDetector(n_schema_cols=20)

    eval_env = SQLQueryEnv(
        db_path=db_path,
        task_registry=task_registry,
        reward_fn=reward_fn,
        nl_vocab_path=str(project_root / "configs/nl_vocab.json"),
        seed=99,
        task_id=task_id,
    )

    results_dir = project_root / "results" / task_id / reward_name
    results_dir.mkdir(parents=True, exist_ok=True)

    training_log = []
    detector_trace: list[dict] = []
    best_eval_reward = -float("inf")

    print(f"\n{'='*60}")
    print(f"Training REINFORCE | Task: {task_id} | Reward: {reward_name}")
    print(f"{'='*60}")

    start_time = time.time()

    for ep_batch in range(0, max_episodes, eval_every):
        # Training
        agent.train_mode()
        train_metrics = run_episodes(
            env,
            agent,
            n_episodes=eval_every,
            training=True,
            detector=detector,
            detector_trace=detector_trace if save_detector_trace else None,
            task_id=task_id,
            reward_name=reward_name,
            start_episode=ep_batch,
        )

        # Hacking detection
        hacking_alerts = detector.detect()
        if hacking_alerts:
            for alert in hacking_alerts:
                if alert.signal == "COMPOSITE":
                    print(f"  ⚠ {alert.description}")

        # Evaluation
        agent.eval_mode()
        eval_metrics = run_episodes(
            eval_env,
            agent,
            n_episodes=eval_episodes,
            training=False,
            task_id=task_id,
            reward_name=reward_name,
        )

        log_entry = {
            "episode": ep_batch + eval_every,
            "train_mean_reward": train_metrics["mean_reward"],
            "eval_mean_reward": eval_metrics["mean_reward"],
            "eval_exec_match": eval_metrics["mean_exec_match"],
            "eval_partial": eval_metrics["mean_partial"],
            "elapsed_s": time.time() - start_time,
        }
        training_log.append(log_entry)

        if verbose:
            print(
                f"  Ep {ep_batch + eval_every:4d} | "
                f"Train R={train_metrics['mean_reward']:+.3f} | "
                f"Eval R={eval_metrics['mean_reward']:+.3f} | "
                f"ExecMatch={eval_metrics['mean_exec_match']:.3f} | "
                f"Partial={eval_metrics['mean_partial']:.3f}"
            )

        if eval_metrics["mean_reward"] > best_eval_reward:
            best_eval_reward = eval_metrics["mean_reward"]
            agent.save(str(results_dir / "best_model.pt"))

    # Save results
    (results_dir / "training_log.json").write_text(json.dumps(training_log, indent=2))
    detector.save_alerts(str(results_dir / "hacking_alerts.json"))
    if save_detector_trace:
        trace_payload = {
            "task_id": task_id,
            "reward_name": reward_name,
            "max_episodes": max_episodes,
            "eval_every": eval_every,
            "eval_episodes": eval_episodes,
            "n_trace_entries": len(detector_trace),
            "entries": detector_trace,
        }
        (results_dir / "detector_trace.json").write_text(json.dumps(trace_payload, indent=2))

    print(f"\nBest eval reward: {best_eval_reward:.4f}")
    print(f"Results saved to: {results_dir}")

    return {
        "task_id": task_id,
        "reward_name": reward_name,
        "best_eval_reward": best_eval_reward,
        "training_log": training_log,
    }


def main():
    parser = argparse.ArgumentParser(description="Run sql-rl-env experiment")
    parser.add_argument("--task", default="task_01_simple",
                        choices=["task_01_simple", "task_02_aggregation", "task_03_join",
                                 "task_04_subquery", "task_05_window"])
    parser.add_argument("--reward", default="composite",
                        choices=["exact", "execution", "partial", "composite"])
    parser.add_argument("--episodes", type=int, default=2000)
    parser.add_argument("--eval_every", type=int, default=100)
    parser.add_argument("--eval_episodes", type=int, default=100)
    parser.add_argument("--save_detector_trace", action="store_true",
                        help="Persist per-episode detector inputs for offline threshold calibration")
    parser.add_argument("--all", action="store_true",
                        help="Run all task × reward combinations")
    parser.add_argument("--dry_run", action="store_true",
                        help="Evaluate random + rule agents only (no training)")
    args = parser.parse_args()

    root = PROJECT_ROOT

    if args.dry_run:
        print("[dry_run] Evaluating random and rule-based agents...")
        env_cfg = yaml.safe_load((root / "configs/env_config.yaml").read_text())
        reward_cfg = yaml.safe_load((root / "configs/reward_config.yaml").read_text())
        db_path = str(root / env_cfg["db_path"])
        task_registry = load_task_registry(str(root / "configs/task_config.yaml"), str(root))
        reward_fn = get_reward_fn("composite", reward_cfg, db_path)
        env = SQLQueryEnv(
            db_path=db_path, task_registry=task_registry, reward_fn=reward_fn,
            nl_vocab_path=str(root / "configs/nl_vocab.json"), seed=42
        )
        random_metrics = run_episodes(env, RandomAgent(seed=42), n_episodes=200)
        print(f"[random] mean_reward={random_metrics['mean_reward']:.4f}, "
              f"exec_match={random_metrics['mean_exec_match']:.4f}")
        return

    if args.all:
        all_tasks = ["task_01_simple", "task_02_aggregation", "task_03_join",
                     "task_04_subquery", "task_05_window"]
        all_rewards = ["composite", "execution", "partial"]
        for task in all_tasks:
            for reward in all_rewards:
                train_reinforce(task, reward, root,
                                max_episodes=args.episodes,
                                eval_every=args.eval_every,
                                eval_episodes=args.eval_episodes,
                                save_detector_trace=args.save_detector_trace)
    else:
        train_reinforce(args.task, args.reward, root,
                        max_episodes=args.episodes,
                        eval_every=args.eval_every,
                        eval_episodes=args.eval_episodes,
                        save_detector_trace=args.save_detector_trace)


if __name__ == "__main__":
    main()
