"""
run_all_experiments.py — Master training + evaluation script.

Runs:
  1. REINFORCE training on all 5 tasks (composite reward, 1500 episodes each)
  2. Baseline evaluation: random + rule-based on all tasks × all reward functions
  3. REINFORCE evaluation vs baselines
  4. Curriculum ablation: Task 3 from scratch vs Task 1-pretrained checkpoint
  5. Reward hacking observations logged during training

Saves:
  results/training_curves.json    — per-task learning curves (episode → reward)
  results/eval_table.json         — full agent × task × reward table
  results/curriculum_ablation.json — Task 3 convergence comparison
  results/hacking_observations.json — observed hacking signals

Usage:
    python scripts/run_all_experiments.py

Author: Kartik Munjal
"""

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
from src.agents.random_agent import RandomAgent
from src.agents.rule_agent import RuleAgent
from src.agents.reinforce_agent import REINFORCEAgent
from src.rewards import get_reward_fn
from src.analysis.reward_hacking_detector import RewardHackingDetector

TASKS = [
    "task_01_simple",
    "task_02_aggregation",
    "task_03_join",
    "task_04_subquery",
    "task_05_window",
]
REWARDS = ["exact", "execution", "partial", "composite"]
OBS_DIM = 1121   # schema(160) + nl(128) + partial(832) + phase(1)


def make_env(task_id, reward_name, seed=42):
    env_cfg = yaml.safe_load((PROJECT_ROOT / "configs/env_config.yaml").read_text())
    reward_cfg = yaml.safe_load((PROJECT_ROOT / "configs/reward_config.yaml").read_text())
    db_path = str(PROJECT_ROOT / "data/ecommerce.db")
    task_registry = load_task_registry(
        str(PROJECT_ROOT / "configs/task_config.yaml"), str(PROJECT_ROOT)
    )
    reward_fn = get_reward_fn(reward_name, reward_cfg, db_path)
    return SQLQueryEnv(
        db_path=db_path,
        task_registry=task_registry,
        reward_fn=reward_fn,
        nl_vocab_path=str(PROJECT_ROOT / "configs/nl_vocab.json"),
        seed=seed,
        task_id=task_id,
        max_episode_steps=15,
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


def eval_agent(agent, task_id, reward_name, n=200, seed=99):
    env = make_env(task_id, reward_name, seed=seed)
    rewards, exec_m, partial_m = [], [], []
    for _ in range(n):
        r, info = run_episode(env, agent, training=False)
        rewards.append(r)
        rc = info.get("reward_components", {})
        exec_m.append(rc.get("r_exec", rc.get("exec_execution_match", rc.get("execution_match", 0.0))))
        partial_m.append(rc.get("r_partial", rc.get("partial_partial_credit", rc.get("partial_credit", 0.0))))
    return {
        "mean_reward": float(np.mean(rewards)),
        "std_reward": float(np.std(rewards)),
        "exec_match": float(np.mean(exec_m)),
        "partial_credit": float(np.mean(partial_m)),
    }


# ============================================================
# 1. REINFORCE training on all 5 tasks (composite)
# ============================================================
def train_all_tasks(n_episodes=1500, eval_every=100):
    print("\n" + "="*60)
    print("REINFORCE TRAINING — all 5 tasks, composite reward")
    print("="*60)

    training_curves = {}
    results_dir = PROJECT_ROOT / "results"
    results_dir.mkdir(exist_ok=True)

    for task_id in TASKS:
        print(f"\n--- {task_id} ---")
        env = make_env(task_id, "composite", seed=42)
        eval_env = make_env(task_id, "composite", seed=99)
        agent = make_reinforce(seed=42)
        detector = RewardHackingDetector(n_schema_cols=20)

        curve = []
        hacking_obs = []
        t0 = time.time()

        for ep in range(1, n_episodes + 1):
            agent.train_mode()
            r, info = run_episode(env, agent, training=True)

            # Log to hacking detector — extract real values from assembled SQL
            rc = info.get("reward_components", {})
            sql = info.get("sql_so_far", "")
            import re as _re
            op_m = _re.search(r'\bWHERE\s+\S+\s*(=|!=|>=|<=|>|<|LIKE)\b', sql, _re.IGNORECASE)
            where_op = op_m.group(1).upper() if op_m else "NONE"
            sel_m = _re.search(r'SELECT\s+(.*?)\s+FROM\b', sql, _re.IGNORECASE | _re.DOTALL)
            col_count = len(sel_m.group(1).split(",")) if sel_m else 1
            detector.update({
                "predicted_rows": int(rc.get("exec_pred_rows", rc.get("exec_pred_rows_no_limit", 0))),
                "where_operator": where_op,
                "select_col_count": col_count,
            })

            if ep % eval_every == 0:
                agent.eval_mode()
                eval_rewards = []
                eval_exec = []
                for _ in range(50):
                    er, einfo = run_episode(eval_env, agent, training=False)
                    eval_rewards.append(er)
                    erc = einfo.get("reward_components", {})
                    exec_val = erc.get("r_exec", erc.get("exec_execution_match", 0.0))
                    eval_exec.append(exec_val)

                mean_r = float(np.mean(eval_rewards))
                mean_exec = float(np.mean(eval_exec))
                curve.append({
                    "episode": ep,
                    "eval_mean_reward": mean_r,
                    "eval_exec_match": mean_exec,
                    "elapsed_s": round(time.time() - t0, 1),
                })
                print(f"  ep {ep:4d} | eval_R={mean_r:.3f} | exec={mean_exec:.3f} | {time.time()-t0:.0f}s")

                # Check for hacking
                alerts = detector.detect()
                for alert in alerts:
                    if alert.signal == "COMPOSITE":
                        obs_entry = {
                            "task": task_id, "episode": ep,
                            "signal": alert.signal, "severity": alert.severity,
                            "description": alert.description,
                        }
                        hacking_obs.append(obs_entry)
                        print(f"  ⚠ HACKING: {alert.description[:80]}")

        # Save checkpoint
        ckpt_dir = results_dir / task_id
        ckpt_dir.mkdir(exist_ok=True)
        agent.save(str(ckpt_dir / "reinforce_composite.pt"))
        training_curves[task_id] = {
            "curve": curve,
            "hacking_observations": hacking_obs,
        }
        print(f"  Saved checkpoint → results/{task_id}/reinforce_composite.pt")

    Path("results/training_curves.json").write_text(json.dumps(training_curves, indent=2))
    print("\n✓ Training curves → results/training_curves.json")
    return training_curves


# ============================================================
# 2. Baseline + REINFORCE evaluation table
# ============================================================
def eval_all_agents():
    print("\n" + "="*60)
    print("EVALUATION TABLE — all agents × tasks × rewards")
    print("="*60)

    results = {}
    for task_id in TASKS:
        results[task_id] = {}
        for reward_name in REWARDS:
            results[task_id][reward_name] = {}

            # Random
            agent = RandomAgent(seed=0)
            m = eval_agent(agent, task_id, reward_name, n=300)
            results[task_id][reward_name]["random"] = m
            print(f"[{task_id}][{reward_name}] random:  R={m['mean_reward']:.3f}  exec={m['exec_match']:.3f}")

            # Rule-based
            agent = RuleAgent.from_config(str(PROJECT_ROOT / "configs/agent_config.yaml"))
            m = eval_agent(agent, task_id, reward_name, n=300)
            results[task_id][reward_name]["rule"] = m
            print(f"[{task_id}][{reward_name}] rule:    R={m['mean_reward']:.3f}  exec={m['exec_match']:.3f}")

            # REINFORCE (if checkpoint exists)
            ckpt = PROJECT_ROOT / "results" / task_id / "reinforce_composite.pt"
            if ckpt.exists():
                agent = make_reinforce(seed=0)
                agent.load(str(ckpt))
                agent.eval_mode()
                m = eval_agent(agent, task_id, reward_name, n=300)
                results[task_id][reward_name]["reinforce"] = m
                print(f"[{task_id}][{reward_name}] reinforce: R={m['mean_reward']:.3f}  exec={m['exec_match']:.3f}")

    Path("results/eval_table.json").write_text(json.dumps(results, indent=2))
    print("\n✓ Evaluation table → results/eval_table.json")
    return results


# ============================================================
# 3. Curriculum ablation
# ============================================================
def curriculum_ablation(n_episodes=1200, eval_every=100):
    print("\n" + "="*60)
    print("CURRICULUM ABLATION — Task 3 from scratch vs Task 1 pretrained")
    print("="*60)

    task3_id = "task_03_join"
    task1_id = "task_01_simple"
    results = {"scratch": [], "curriculum": []}

    # Check Task 1 checkpoint
    task1_ckpt = PROJECT_ROOT / "results" / task1_id / "reinforce_composite.pt"
    if not task1_ckpt.exists():
        print("  Task 1 checkpoint not found — run train_all_tasks first")
        return results

    def train_task3(agent, label):
        env = make_env(task3_id, "composite", seed=42)
        eval_env = make_env(task3_id, "composite", seed=99)
        curve = []
        t0 = time.time()
        for ep in range(1, n_episodes + 1):
            agent.train_mode()
            run_episode(env, agent, training=True)
            if ep % eval_every == 0:
                agent.eval_mode()
                eval_rewards, eval_exec = [], []
                for _ in range(50):
                    er, einfo = run_episode(eval_env, agent, training=False)
                    eval_rewards.append(er)
                    erc = einfo.get("reward_components", {})
                    eval_exec.append(erc.get("r_exec", erc.get("exec_execution_match", 0.0)))
                mean_r = float(np.mean(eval_rewards))
                mean_exec = float(np.mean(eval_exec))
                curve.append({
                    "episode": ep,
                    "eval_mean_reward": mean_r,
                    "eval_exec_match": mean_exec,
                    "elapsed_s": round(time.time() - t0, 1),
                })
                print(f"  [{label}] ep {ep:4d} | R={mean_r:.3f} | exec={mean_exec:.3f}")
        return curve

    # Scratch
    print("\n-- Task 3 from scratch --")
    scratch_agent = make_reinforce(seed=7)
    results["scratch"] = train_task3(scratch_agent, "scratch")

    # Curriculum (Task 1 weights → Task 3 fine-tune)
    print("\n-- Task 3 from Task-1 checkpoint --")
    curric_agent = make_reinforce(seed=7)
    curric_agent.load(str(task1_ckpt))
    results["curriculum"] = train_task3(curric_agent, "curriculum")

    Path("results/curriculum_ablation.json").write_text(json.dumps(results, indent=2))
    print("\n✓ Curriculum ablation → results/curriculum_ablation.json")
    return results


# ============================================================
# Main
# ============================================================
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_eval", action="store_true")
    parser.add_argument("--skip_curriculum", action="store_true")
    parser.add_argument("--episodes", type=int, default=1500)
    args = parser.parse_args()

    t_start = time.time()

    if not args.skip_train:
        curves = train_all_tasks(n_episodes=args.episodes)

    if not args.skip_eval:
        table = eval_all_agents()

    if not args.skip_curriculum:
        ablation = curriculum_ablation(n_episodes=1200)

    print(f"\n✓ All experiments complete in {(time.time()-t_start)/60:.1f} minutes")
