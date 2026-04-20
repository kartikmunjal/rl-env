"""
run_threshold_sensitivity.py — detector threshold sensitivity on saved checkpoints.

This script replays the saved REINFORCE checkpoints under a fixed evaluation
protocol, records the detector inputs for each episode, and then sweeps
kl_threshold over a small range while holding the entropy and coverage
thresholds fixed.

The result is not a substitute for training-trace calibration: it measures
stability on checkpoint evaluation traces. It is still useful because it
replaces purely speculative discussion of threshold sensitivity with a concrete
artifact generated from the released models.

Saves:
  results/threshold_sensitivity_eval.json
"""

from __future__ import annotations

import json
import re as _re
import sys
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.agents.reinforce_agent import REINFORCEAgent
from src.analysis.reward_hacking_detector import RewardHackingDetector
from src.env.sql_env import SQLQueryEnv
from src.rewards import get_reward_fn
from src.tasks.base import load_task_registry

TASKS = [
    "task_01_simple",
    "task_02_aggregation",
    "task_03_join",
    "task_04_subquery",
    "task_05_window",
]
OBS_DIM = 1121
KL_THRESHOLDS = [0.35, 0.50, 0.65]


def _load_configs():
    env_cfg = yaml.safe_load((PROJECT_ROOT / "configs/env_config.yaml").read_text())
    reward_cfg = yaml.safe_load((PROJECT_ROOT / "configs/reward_config.yaml").read_text())
    db_path = str(PROJECT_ROOT / "data/ecommerce.db")
    task_registry = load_task_registry(
        str(PROJECT_ROOT / "configs/task_config.yaml"), str(PROJECT_ROOT)
    )
    return env_cfg, reward_cfg, db_path, task_registry


def make_env(task_id: str, reward_fn, seed: int = 77) -> SQLQueryEnv:
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


def make_reinforce(seed: int = 0) -> REINFORCEAgent:
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


def run_episode(env, agent):
    obs, info = env.reset()
    if hasattr(agent, "set_nl_query"):
        agent.set_nl_query(info["nl_query"])
    agent.reset_episode()
    terminated = truncated = False
    while not terminated and not truncated:
        action = agent.act(obs, info["action_mask"])
        obs, reward, terminated, truncated, info = env.step(action)
    return reward, info


def _extract_sql_stats(sql: str, reward_components: dict) -> dict:
    op_m = _re.search(r"\bWHERE\s+\S+\s*(=|!=|>=|<=|>|<|LIKE)\b", sql, _re.IGNORECASE)
    where_op = op_m.group(1).upper() if op_m else "NONE"
    sel_m = _re.search(r"SELECT\s+(.*?)\s+FROM\b", sql, _re.IGNORECASE | _re.DOTALL)
    col_count = len(sel_m.group(1).split(",")) if sel_m else 1
    pred_rows = int(
        reward_components.get("exec_pred_rows", reward_components.get("exec_pred_rows_no_limit", 0))
    )
    return {
        "predicted_rows": pred_rows,
        "where_operator": where_op,
        "select_col_count": col_count,
    }


def collect_episode_stats(task_id: str, n_episodes: int) -> list[dict]:
    _, reward_cfg, db_path, _ = _load_configs()
    reward_fn = get_reward_fn("composite", reward_cfg, db_path)
    env = make_env(task_id, reward_fn, seed=77)
    ckpt = PROJECT_ROOT / "results" / task_id / "reinforce_composite.pt"
    if not ckpt.exists():
        raise FileNotFoundError(f"Missing checkpoint: {ckpt}")

    agent = make_reinforce(seed=0)
    agent.load(str(ckpt))
    agent.eval_mode()

    traces = []
    for _ in range(n_episodes):
        _, info = run_episode(env, agent)
        rc = info.get("reward_components", {})
        traces.append(_extract_sql_stats(info.get("sql_so_far", ""), rc))
    return traces


def replay_thresholds(traces: list[dict]) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for tau in KL_THRESHOLDS:
        detector = RewardHackingDetector(
            n_schema_cols=20,
            window_size=100,
            check_every=50,
            kl_threshold=tau,
        )
        alerts_log = []
        for ep, stats in enumerate(traces, start=1):
            detector.update(stats)
            if ep % detector.check_every == 0:
                alerts = detector.detect()
                composite_alerts = [a for a in alerts if a.signal == "COMPOSITE"]
                for alert in composite_alerts:
                    alerts_log.append({"episode": ep, "severity": alert.severity})
        out[f"{tau:.2f}"] = {
            "kl_threshold": tau,
            "first_alert_episode": alerts_log[0]["episode"] if alerts_log else None,
            "total_composite_alerts": len(alerts_log),
            "alert_rate": round(len(alerts_log) / 8.0, 3),
        }
    return out


def main(n_episodes: int = 400) -> None:
    results = {
        "experiment": "checkpoint_eval_kl_threshold_sensitivity",
        "n_episodes": n_episodes,
        "thresholds": {"kl": KL_THRESHOLDS, "entropy": 0.80, "coverage_trend": 0.40},
        "notes": [
            "Uses released REINFORCE checkpoints under fixed evaluation.",
            "Measures sensitivity on checkpoint evaluation traces, not original training traces.",
            "Past training runs cannot be replayed exactly from artifacts because per-episode detector traces were not persisted.",
        ],
        "tasks": {},
    }

    for task_id in TASKS:
        traces = collect_episode_stats(task_id, n_episodes=n_episodes)
        results["tasks"][task_id] = replay_thresholds(traces)

    out = PROJECT_ROOT / "results/threshold_sensitivity_eval.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"Saved threshold sensitivity results to {out}")


if __name__ == "__main__":
    main()
