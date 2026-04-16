"""
composite.py — R4: Composite reward (primary training signal for REINFORCE).

Design rationale (see research_notes/design_decisions.md §R4):

The composite reward is a weighted linear combination of R1, R2, and R3,
plus an efficiency penalty that fires when the predicted query returns
far more rows than expected (counter-measure for H1.1, H2.1, H4.2).

Formula:
    r = w_exec * R2 + w_partial * R3 + w_exact * R1 - w_eff * efficiency_penalty

Default weights (reward_config.yaml):
    w_exec=0.50, w_partial=0.30, w_exact=0.15, w_eff=0.05

The efficiency penalty:
    penalty = log(actual_rows_no_limit + 1) / log(expected_rows + 2)
    if actual_rows_no_limit > efficiency_multiplier * expected_rows else 0.0

    actual_rows_no_limit: COUNT(*) of predicted SQL WITHOUT the LIMIT clause
    This is the counter-measure for H4.2 (LIMIT-based penalty suppression):
    an agent that adds "LIMIT 100" cannot hide the true result-set size.

Weight sensitivity (documented in reward_hacking_report.md §H4.1):
    High w_exec → sparse reward, correct policy, slow learning
    High w_partial → dense reward, fast learning, partial-credit plateau risk
    The experiments/02_reward_signal_study.py ablates all three weight profiles.

Returns: (scalar_reward, component_dict)
    The component_dict contains all sub-rewards for logging and analysis.

Author: Kartik Munjal
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from src.env.executor import SQLExecutor, ExecutionStatus
from .exact_match import ExactMatchReward
from .execution_match import ExecutionMatchReward
from .partial_credit import PartialCreditReward


@dataclass
class CompositeReward:
    exact_reward: ExactMatchReward
    exec_reward: ExecutionMatchReward
    partial_reward: PartialCreditReward
    executor: SQLExecutor

    w_exec: float = 0.50
    w_partial: float = 0.30
    w_exact: float = 0.15
    w_eff: float = 0.05
    efficiency_multiplier: float = 5.0
    efficiency_penalty_max: float = 0.20

    def __call__(
        self,
        predicted_sql: str,
        gold_sql: str,
        db_path=None,
        schema=None,
    ) -> tuple[float, dict]:
        # R1: Exact match
        r_exact, exact_info = self.exact_reward(predicted_sql, gold_sql)

        # R2: Execution match
        r_exec, exec_info = self.exec_reward(predicted_sql, gold_sql)

        # R3: Partial credit
        r_partial, partial_info = self.partial_reward(predicted_sql, gold_sql)

        # Efficiency penalty
        eff_penalty = self._efficiency_penalty(
            predicted_sql=predicted_sql,
            gold_sql=gold_sql,
            exec_info=exec_info,
        )

        reward = (
            self.w_exec * r_exec
            + self.w_partial * r_partial
            + self.w_exact * r_exact
            - self.w_eff * eff_penalty
        )

        components = {
            "composite": reward,
            "r_exact": r_exact,
            "r_exec": r_exec,
            "r_partial": r_partial,
            "efficiency_penalty": eff_penalty,
            **{f"exec_{k}": v for k, v in exec_info.items()},
            **{f"partial_{k}": v for k, v in partial_info.items()},
        }
        return reward, components

    def _efficiency_penalty(
        self,
        predicted_sql: str,
        gold_sql: str,
        exec_info: dict,
    ) -> float:
        """
        Fire an efficiency penalty when predicted query returns far more rows
        than the gold query (symptom of cross-join or SELECT * exploitation).

        Counter-measure for H1.1, H2.1, H4.2:
        - Uses row_count_no_limit (COUNT(*) without LIMIT clause) from exec_info
        - This prevents H4.2: agent cannot suppress penalty by adding LIMIT
        """
        pred_rows = exec_info.get("pred_rows_no_limit", 0)
        gold_rows = exec_info.get("gold_rows", 0)

        if gold_rows == 0:
            return 0.0

        ratio = pred_rows / max(gold_rows, 1)
        if ratio <= self.efficiency_multiplier:
            return 0.0

        # Log-scaled penalty: grows slowly, capped at efficiency_penalty_max
        penalty = math.log(ratio + 1) / math.log(self.efficiency_multiplier + 2)
        return min(penalty, self.efficiency_penalty_max)

    @classmethod
    def from_config(cls, cfg: dict, db_path: str) -> "CompositeReward":
        comp_cfg = cfg.get("composite", {})
        executor = SQLExecutor(db_path)
        return cls(
            exact_reward=ExactMatchReward.from_config(cfg),
            exec_reward=ExecutionMatchReward.from_config(cfg, db_path),
            partial_reward=PartialCreditReward.from_config(cfg),
            executor=executor,
            w_exec=comp_cfg.get("w_exec", 0.50),
            w_partial=comp_cfg.get("w_partial", 0.30),
            w_exact=comp_cfg.get("w_exact", 0.15),
            w_eff=comp_cfg.get("w_eff", 0.05),
        )
