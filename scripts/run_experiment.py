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
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import yaml

from src.env.sql_env import SQLQueryEnv
from src.tasks.base import load_task_registry
from src.agents import RandomAgent, RuleAgent, REINFORCEAgent
from src.rewards import get_reward_fn
from src.analysis.reward_hacking_detector import RewardHackingDetector


def run_episodes(
    env: SQLQueryEnv,
    agent,
    n_episodes: int,
    training: bool = False,
    detector: RewardHackingDetector = None,
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

        if ep < 5 or ep % 100 == 0:
            sql_samples.append({
                "episode": ep,
                "nl": final_info.get("nl_query", ""),
                "sql": final_info.get("sql_so_far", ""),
                "reward": ep_reward,
            })

        # Update hacking detector
        if detector:
            detector.update({
                "predicted_rows": rc.get("exec_pred_rows", 0),
                "where_operator": "=",   # TODO: extract from SQL
                "select_col_count": 1,   # TODO: count from SQL
            })

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
    best_eval_reward = -float("inf")

    print(f"\n{'='*60}")
    print(f"Training REINFORCE | Task: {task_id} | Reward: {reward_name}")
    print(f"{'='*60}")

    start_time = time.time()

    for ep_batch in range(0, max_episodes, eval_every):
        # Training
        agent.train_mode()
        train_metrics = run_episodes(
            env, agent, n_episodes=eval_every, training=True, detector=detector
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
            eval_env, agent, n_episodes=eval_episodes, training=False
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
                                eval_every=args.eval_every)
    else:
        train_reinforce(args.task, args.reward, root,
                        max_episodes=args.episodes,
                        eval_every=args.eval_every)


if __name__ == "__main__":
    main()
