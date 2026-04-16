"""
run_extra_experiments.py — Additional ablations for the rl-env research project.

Experiments:
  K. R4 weight sensitivity: three weight configs × Tasks 2 and 5 (600 episodes each)
  N. Extended curriculum: T1→T2 and T3→T4 transfer paths vs scratch (800 episodes)
  O. Reward signal × hacking rate: 4 rewards × 5 tasks, first-alert episode table

Saves:
  results/weight_ablation.json       — K
  results/extended_curriculum.json   — N
  results/reward_hacking_table.json  — O

Usage:
    python scripts/run_extra_experiments.py [--skip_K] [--skip_N] [--skip_O]

Author: Kartik Munjal
"""

import argparse
import json
import re as _re
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import yaml

from src.env.sql_env import SQLQueryEnv
from src.tasks.base import load_task_registry
from src.agents.reinforce_agent import REINFORCEAgent
from src.agents.random_agent import RandomAgent
from src.rewards import get_reward_fn, CompositeReward
from src.rewards.exact_match import ExactMatchReward
from src.rewards.execution_match import ExecutionMatchReward
from src.rewards.partial_credit import PartialCreditReward
from src.env.executor import SQLExecutor
from src.analysis.reward_hacking_detector import RewardHackingDetector

TASKS = [
    "task_01_simple",
    "task_02_aggregation",
    "task_03_join",
    "task_04_subquery",
    "task_05_window",
]
REWARDS = ["exact", "execution", "partial", "composite"]
OBS_DIM = 1121

WEIGHT_PROFILES = {
    "balanced":    {"w_exec": 0.50, "w_partial": 0.30, "w_exact": 0.15, "w_eff": 0.05},
    "high_exec":   {"w_exec": 0.70, "w_partial": 0.20, "w_exact": 0.05, "w_eff": 0.05},
    "high_partial":{"w_exec": 0.30, "w_partial": 0.50, "w_exact": 0.15, "w_eff": 0.05},
}


# ============================================================
# Shared helpers
# ============================================================

def _load_configs():
    env_cfg = yaml.safe_load((PROJECT_ROOT / "configs/env_config.yaml").read_text())
    reward_cfg = yaml.safe_load((PROJECT_ROOT / "configs/reward_config.yaml").read_text())
    db_path = str(PROJECT_ROOT / "data/ecommerce.db")
    task_registry = load_task_registry(
        str(PROJECT_ROOT / "configs/task_config.yaml"), str(PROJECT_ROOT)
    )
    return env_cfg, reward_cfg, db_path, task_registry


def make_env(task_id, reward_fn, seed=42):
    _, _, db_path, task_registry = _load_configs()
    return SQLQueryEnv(
        db_path=db_path,
        task_registry=task_registry,
        reward_fn=reward_fn,
        nl_vocab_path=str(PROJECT_ROOT / "configs/nl_vocab.json"),
        seed=seed,
        task_id=task_id,
        max_episode_steps=15,
    )


def make_composite_reward(weights: dict, db_path: str, reward_cfg: dict) -> CompositeReward:
    return CompositeReward(
        exact_reward=ExactMatchReward.from_config(reward_cfg),
        exec_reward=ExecutionMatchReward.from_config(reward_cfg, db_path),
        partial_reward=PartialCreditReward.from_config(reward_cfg),
        executor=SQLExecutor(db_path),
        w_exec=weights["w_exec"],
        w_partial=weights["w_partial"],
        w_exact=weights["w_exact"],
        w_eff=weights["w_eff"],
    )


def make_reinforce(seed=42):
    return REINFORCEAgent(
        obs_dim=OBS_DIM,
        hidden_dim=256,
        hidden_dim2=128,
        dropout=0.10,
        learning_rate=3e-4,
        baseline_lr=1e-3,
        gamma=1.0,
        entropy_coeff=0.01,
        grad_clip=0.5,
        device="cpu",
        seed=seed,
    )


def run_episode(env, agent, training=False):
    obs, info = env.reset()
    if hasattr(agent, "set_nl_query"):
        agent.set_nl_query(info["nl_query"])
    agent.reset_episode()
    terminated = truncated = False
    while not terminated and not truncated:
        action = agent.act(obs, info["action_mask"])
        obs, reward, terminated, truncated, info = env.step(action)
    if training and hasattr(agent, "on_episode_end"):
        agent.on_episode_end(reward, info)
    return reward, info


def _extract_sql_stats(sql: str, reward_components: dict) -> dict:
    op_m = _re.search(r'\bWHERE\s+\S+\s*(=|!=|>=|<=|>|<|LIKE)\b', sql, _re.IGNORECASE)
    where_op = op_m.group(1).upper() if op_m else "NONE"
    sel_m = _re.search(r'SELECT\s+(.*?)\s+FROM\b', sql, _re.IGNORECASE | _re.DOTALL)
    col_count = len(sel_m.group(1).split(",")) if sel_m else 1
    pred_rows = int(reward_components.get(
        "exec_pred_rows", reward_components.get("exec_pred_rows_no_limit", 0)
    ))
    return {"predicted_rows": pred_rows, "where_operator": where_op, "select_col_count": col_count}


def eval_quick(agent, env, n=50):
    rewards, exec_vals, partial_vals = [], [], []
    for _ in range(n):
        r, info = run_episode(env, agent, training=False)
        rewards.append(r)
        rc = info.get("reward_components", {})
        exec_vals.append(rc.get("r_exec", rc.get("exec_execution_match", rc.get("execution_match", 0.0))))
        partial_vals.append(rc.get("r_partial", rc.get("partial_partial_credit", rc.get("partial_credit", 0.0))))
    return {
        "mean_reward": float(np.mean(rewards)),
        "std_reward": float(np.std(rewards)),
        "exec_match": float(np.mean(exec_vals)),
        "partial_credit": float(np.mean(partial_vals)),
    }


# ============================================================
# K: R4 weight sensitivity ablation
# ============================================================

def run_weight_ablation(n_episodes=600, eval_every=100):
    print("\n" + "=" * 60)
    print("K: WEIGHT SENSITIVITY ABLATION — Tasks 2 & 5 × 3 configs")
    print("=" * 60)

    _, reward_cfg, db_path, _ = _load_configs()
    results = {}
    ablation_tasks = ["task_02_aggregation", "task_05_window"]

    for task_id in ablation_tasks:
        results[task_id] = {}
        for profile_name, weights in WEIGHT_PROFILES.items():
            print(f"\n  [{task_id}] [{profile_name}] "
                  f"w_exec={weights['w_exec']}, w_partial={weights['w_partial']}, "
                  f"w_exact={weights['w_exact']}")
            reward_fn = make_composite_reward(weights, db_path, reward_cfg)
            env = make_env(task_id, reward_fn, seed=42)
            eval_env = make_env(task_id, reward_fn, seed=99)
            agent = make_reinforce(seed=42)
            detector = RewardHackingDetector(n_schema_cols=20)
            curve = []
            hacking_count = 0
            first_alert_ep = None
            t0 = time.time()

            for ep in range(1, n_episodes + 1):
                agent.train_mode()
                r, info = run_episode(env, agent, training=True)
                rc = info.get("reward_components", {})
                sql = info.get("sql_so_far", "")
                detector.update(_extract_sql_stats(sql, rc))

                if ep % eval_every == 0:
                    agent.eval_mode()
                    metrics = eval_quick(agent, eval_env, n=50)
                    alerts = detector.detect()
                    composite_alerts = [a for a in alerts if a.signal == "COMPOSITE"]
                    hacking_count += len(composite_alerts)
                    if composite_alerts and first_alert_ep is None:
                        first_alert_ep = ep
                    curve.append({
                        "episode": ep,
                        **metrics,
                        "elapsed_s": round(time.time() - t0, 1),
                        "composite_alerts": len(composite_alerts),
                    })
                    print(f"    ep {ep:4d} | R={metrics['mean_reward']:.3f} "
                          f"exec={metrics['exec_match']:.3f} "
                          f"partial={metrics['partial_credit']:.3f} "
                          f"alerts={hacking_count}")

            final = curve[-1] if curve else {}
            results[task_id][profile_name] = {
                "weights": weights,
                "curve": curve,
                "final_mean_reward": final.get("mean_reward", 0.0),
                "final_exec_match": final.get("exec_match", 0.0),
                "final_partial_credit": final.get("partial_credit", 0.0),
                "total_hacking_alerts": hacking_count,
                "first_alert_episode": first_alert_ep,
            }

    out = PROJECT_ROOT / "results/weight_ablation.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"\n✓ Weight ablation → {out}")
    return results


# ============================================================
# N: Extended curriculum ablation (T1→T2 and T3→T4)
# ============================================================

def run_extended_curriculum(n_episodes=800, eval_every=100):
    print("\n" + "=" * 60)
    print("N: EXTENDED CURRICULUM — T1→T2 and T3→T4")
    print("=" * 60)

    _, reward_cfg, db_path, _ = _load_configs()
    reward_fn_factory = lambda: CompositeReward.from_config(
        {"composite": {"w_exec": 0.50, "w_partial": 0.30, "w_exact": 0.15, "w_eff": 0.05},
         **yaml.safe_load((PROJECT_ROOT / "configs/reward_config.yaml").read_text())},
        db_path,
    )

    paths = [
        {
            "name": "T1_to_T2",
            "source": "task_01_simple",
            "target": "task_02_aggregation",
            "ckpt": PROJECT_ROOT / "results/task_01_simple/reinforce_composite.pt",
        },
        {
            "name": "T3_to_T4",
            "source": "task_03_join",
            "target": "task_04_subquery",
            "ckpt": PROJECT_ROOT / "results/task_03_join/reinforce_composite.pt",
        },
    ]

    results = {}

    def train_task(agent, task_id, label, reward_cfg_obj):
        env = make_env(task_id, reward_cfg_obj, seed=42)
        eval_env = make_env(task_id, reward_cfg_obj, seed=99)
        curve = []
        t0 = time.time()
        for ep in range(1, n_episodes + 1):
            agent.train_mode()
            run_episode(env, agent, training=True)
            if ep % eval_every == 0:
                agent.eval_mode()
                metrics = eval_quick(agent, eval_env, n=50)
                curve.append({"episode": ep, **metrics, "elapsed_s": round(time.time() - t0, 1)})
                print(f"    [{label}] ep {ep:4d} | R={metrics['mean_reward']:.3f} "
                      f"exec={metrics['exec_match']:.3f}")
        return curve

    for path_cfg in paths:
        name = path_cfg["name"]
        target = path_cfg["target"]
        ckpt = path_cfg["ckpt"]
        results[name] = {"source": path_cfg["source"], "target": target}

        reward_fn = CompositeReward.from_config(
            yaml.safe_load((PROJECT_ROOT / "configs/reward_config.yaml").read_text()),
            db_path,
        )

        if not ckpt.exists():
            print(f"  ⚠ {name}: checkpoint {ckpt} not found — skipping curriculum")
            results[name]["curriculum"] = []
        else:
            print(f"\n-- {name}: from {path_cfg['source']} checkpoint → {target} --")
            curriculum_agent = make_reinforce(seed=7)
            curriculum_agent.load(str(ckpt))
            results[name]["curriculum"] = train_task(
                curriculum_agent, target, f"{name}/curriculum", reward_fn
            )

        print(f"\n-- {name}: from scratch → {target} --")
        reward_fn2 = CompositeReward.from_config(
            yaml.safe_load((PROJECT_ROOT / "configs/reward_config.yaml").read_text()),
            db_path,
        )
        scratch_agent = make_reinforce(seed=7)
        results[name]["scratch"] = train_task(
            scratch_agent, target, f"{name}/scratch", reward_fn2
        )

        # Convergence speed: episode at which curriculum first exceeds scratch ep-300 level
        if results[name]["curriculum"] and results[name]["scratch"]:
            scratch_300 = next(
                (p["mean_reward"] for p in results[name]["scratch"] if p["episode"] >= 300),
                results[name]["scratch"][-1]["mean_reward"],
            )
            curric_cross = next(
                (p["episode"] for p in results[name]["curriculum"]
                 if p["mean_reward"] >= scratch_300),
                None,
            )
            results[name]["scratch_ep300_reward"] = scratch_300
            results[name]["curriculum_crossover_ep"] = curric_cross
            print(f"  Scratch ep-300 reward: {scratch_300:.3f}")
            print(f"  Curriculum crosses that level at episode: {curric_cross}")

    out = PROJECT_ROOT / "results/extended_curriculum.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"\n✓ Extended curriculum → {out}")
    return results


# ============================================================
# O: Reward signal × hacking rate interaction
# ============================================================

def run_hacking_table(n_episodes=400):
    """
    For each (task, reward_signal) pair, load the REINFORCE checkpoint (if any)
    and run n_episodes of evaluation — tracking hacking detector updates.
    Reports first alert episode and total alert count per cell.
    """
    print("\n" + "=" * 60)
    print("O: REWARD × HACKING RATE INTERACTION TABLE")
    print("=" * 60)

    _, reward_cfg, db_path, _ = _load_configs()
    results_dir = PROJECT_ROOT / "results"
    table = {}

    for task_id in TASKS:
        table[task_id] = {}
        ckpt = results_dir / task_id / "reinforce_composite.pt"

        for reward_name in REWARDS:
            print(f"\n  [{task_id}][{reward_name}]", end=" ", flush=True)
            reward_fn = get_reward_fn(reward_name, reward_cfg, db_path)
            env = make_env(task_id, reward_fn, seed=77)

            # Use REINFORCE checkpoint if available, else Random
            if ckpt.exists():
                agent = make_reinforce(seed=0)
                agent.load(str(ckpt))
                agent.eval_mode()
                agent_label = "reinforce"
            else:
                agent = RandomAgent(seed=0)
                agent_label = "random"

            detector = RewardHackingDetector(n_schema_cols=20, window_size=100, check_every=50)
            alerts_log = []

            for ep in range(1, n_episodes + 1):
                r, info = run_episode(env, agent, training=False)
                rc = info.get("reward_components", {})
                sql = info.get("sql_so_far", "")
                detector.update(_extract_sql_stats(sql, rc))

                if ep % detector.check_every == 0:
                    alerts = detector.detect()
                    for a in alerts:
                        if a.signal == "COMPOSITE":
                            alerts_log.append({"episode": ep, "severity": a.severity})

            first_alert = alerts_log[0]["episode"] if alerts_log else None
            total_alerts = len(alerts_log)
            print(f"alerts={total_alerts}, first={first_alert}, agent={agent_label}")

            table[task_id][reward_name] = {
                "agent": agent_label,
                "total_composite_alerts": total_alerts,
                "first_alert_episode": first_alert,
                "alert_rate": round(total_alerts / (n_episodes / detector.check_every), 3),
            }

    out = PROJECT_ROOT / "results/reward_hacking_table.json"
    out.write_text(json.dumps(table, indent=2))
    print(f"\n✓ Reward hacking table → {out}")
    return table


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip_K", action="store_true")
    parser.add_argument("--skip_N", action="store_true")
    parser.add_argument("--skip_O", action="store_true")
    args = parser.parse_args()

    t_start = time.time()

    if not args.skip_K:
        weight_results = run_weight_ablation(n_episodes=600, eval_every=100)

    if not args.skip_N:
        curriculum_results = run_extended_curriculum(n_episodes=800, eval_every=100)

    if not args.skip_O:
        hacking_table = run_hacking_table(n_episodes=400)

    print(f"\n✓ All extra experiments complete in {(time.time()-t_start)/60:.1f} minutes")
