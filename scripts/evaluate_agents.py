"""
evaluate_agents.py — Compare all three agents across all tasks and reward functions.

Produces a results table: agent × task × reward_fn → (mean_reward, exec_match, partial)

Usage:
    python scripts/evaluate_agents.py [--episodes 500] [--output results/eval_table.json]

Author: Kartik Munjal
"""

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import yaml

from src.env.sql_env import SQLQueryEnv
from src.tasks.base import load_task_registry
from src.agents import RandomAgent, RuleAgent, REINFORCEAgent
from src.rewards import get_reward_fn
from scripts.run_experiment import run_episodes


def evaluate_all(
    n_episodes: int = 500,
    output_path: str = "results/eval_table.json",
) -> dict:
    root = PROJECT_ROOT
    env_cfg = yaml.safe_load((root / "configs/env_config.yaml").read_text())
    reward_cfg = yaml.safe_load((root / "configs/reward_config.yaml").read_text())
    db_path = str(root / env_cfg["db_path"])
    task_registry = load_task_registry(
        str(root / "configs/task_config.yaml"), str(root)
    )

    tasks = task_registry.task_ids()
    reward_names = ["exact", "execution", "partial", "composite"]
    agent_names = ["random", "rule"]

    # Check for trained REINFORCE models
    for task in tasks:
        model_path = root / "results" / task / "composite" / "best_model.pt"
        if model_path.exists():
            agent_names.append(f"reinforce_{task}")

    results = {}

    for task_id in tasks:
        results[task_id] = {}
        for reward_name in reward_names:
            results[task_id][reward_name] = {}
            reward_fn = get_reward_fn(reward_name, reward_cfg, db_path)

            env = SQLQueryEnv(
                db_path=db_path,
                task_registry=task_registry,
                reward_fn=reward_fn,
                nl_vocab_path=str(root / "configs/nl_vocab.json"),
                seed=123,
                task_id=task_id,
            )

            # Random agent
            random_agent = RandomAgent(seed=0)
            m = run_episodes(env, random_agent, n_episodes, training=False)
            results[task_id][reward_name]["random"] = {
                "mean_reward": m["mean_reward"],
                "exec_match": m["mean_exec_match"],
                "partial": m["mean_partial"],
            }
            print(f"[{task_id}][{reward_name}] random: R={m['mean_reward']:.4f}")

            # Rule-based agent
            rule_agent = RuleAgent.from_config(str(root / "configs/agent_config.yaml"))
            m = run_episodes(env, rule_agent, n_episodes, training=False)
            results[task_id][reward_name]["rule"] = {
                "mean_reward": m["mean_reward"],
                "exec_match": m["mean_exec_match"],
                "partial": m["mean_partial"],
            }
            print(f"[{task_id}][{reward_name}] rule:   R={m['mean_reward']:.4f}")

            # REINFORCE (if trained model exists)
            model_path = root / "results" / task_id / reward_name / "best_model.pt"
            if model_path.exists():
                rl_agent = REINFORCEAgent.from_config(
                    str(root / "configs/agent_config.yaml")
                )
                rl_agent.load(str(model_path))
                rl_agent.eval_mode()
                m = run_episodes(env, rl_agent, n_episodes, training=False)
                results[task_id][reward_name]["reinforce"] = {
                    "mean_reward": m["mean_reward"],
                    "exec_match": m["mean_exec_match"],
                    "partial": m["mean_partial"],
                }
                print(f"[{task_id}][{reward_name}] reinforce: R={m['mean_reward']:.4f}")

    # Save
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_path).write_text(json.dumps(results, indent=2))
    print(f"\nResults saved to {output_path}")

    # Print summary table
    _print_table(results, tasks, reward_names)
    return results


def _print_table(results: dict, tasks: list, rewards: list) -> None:
    print("\n" + "=" * 80)
    print(f"{'EVALUATION SUMMARY':^80}")
    print("=" * 80)
    header = f"{'Task':<25} {'Reward':<12} {'Agent':<12} {'MeanR':>8} {'ExecMatch':>10} {'Partial':>9}"
    print(header)
    print("-" * 80)
    for task in tasks:
        for reward in rewards:
            for agent_name, m in results.get(task, {}).get(reward, {}).items():
                print(
                    f"{task:<25} {reward:<12} {agent_name:<12} "
                    f"{m['mean_reward']:>8.4f} {m['exec_match']:>10.4f} {m['partial']:>9.4f}"
                )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=500)
    parser.add_argument("--output", default="results/eval_table.json")
    args = parser.parse_args()
    evaluate_all(args.episodes, args.output)
