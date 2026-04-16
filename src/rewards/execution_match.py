"""
execution_match.py — R2: Execution match reward.

Design rationale (see research_notes/design_decisions.md §R2):

Execution match is semantically richer than exact match: two SQL queries
with different syntax but identical result sets score equally well.
This is the correct semantic criterion for query equivalence.

Reward structure:
    1.0  — result sets are exactly equal (frozenset comparison, order-invariant)
    0.5  — result sets overlap by ≥ overlap_threshold (Jaccard ≥ 0.5)
    0.0  — disjoint result sets, both empty, or one empty and one non-empty
   -0.1  — syntax error, timeout, or execution failure

The -0.1 error penalty is a key design decision (H2.2 mitigation):
    Without it, a policy that always generates broken SQL receives 0.0
    for every episode and has zero gradient signal. The policy stagnates
    rather than learns. The -0.1 penalty creates a gradient away from
    broken SQL, nudging the agent toward syntactically valid queries first.

Vulnerability to reward hacking:
    H2.1 — Large result set overlap: returning SELECT * FROM table always
            overlaps with gold queries on the same table. Mitigated by
            using Jaccard (|∩|/|∪|) not Recall (|∩|/|gold|).
    H2.2 — Timeout stagnation: mitigated by -0.1 error penalty.
            Documented in reward_hacking_report.md §H2.2.

Use case: primary evaluation metric; second input to composite reward.

Author: Kartik Munjal
"""

from __future__ import annotations

from dataclasses import dataclass

from src.env.executor import SQLExecutor, ExecutionStatus


@dataclass
class ExecutionMatchReward:
    """
    Callable reward function for execution match.

    Uses a shared SQLExecutor so the DB connection overhead is amortised
    across many reward computations in the same experiment.
    """

    executor: SQLExecutor
    overlap_threshold: float = 0.50
    overlap_reward: float = 0.50
    error_penalty: float = -0.10

    def __call__(
        self,
        predicted_sql: str,
        gold_sql: str,
        db_path=None,
        schema=None,
    ) -> tuple[float, dict]:
        if predicted_sql.startswith("<INVALID"):
            return self.error_penalty, {
                "execution_match": self.error_penalty,
                "reason": "invalid_sql_assembly",
            }

        pred_result = self.executor.run(predicted_sql)
        gold_result = self.executor.run(gold_sql)

        # Gold SQL execution failure is a bug in the task definition
        if not gold_result.success:
            return 0.0, {
                "execution_match": 0.0,
                "reason": f"gold_sql_failed: {gold_result.error_msg}",
            }

        if not pred_result.success:
            return self.error_penalty, {
                "execution_match": self.error_penalty,
                "pred_status": pred_result.status.name,
                "error_msg": pred_result.error_msg,
            }

        exact_match, jaccard = self.executor.compare_results(pred_result, gold_result)

        if exact_match:
            reward = 1.0
            reason = "exact_result_set_match"
        elif jaccard >= self.overlap_threshold:
            reward = self.overlap_reward
            reason = f"overlap_match (jaccard={jaccard:.3f})"
        else:
            reward = 0.0
            reason = f"no_match (jaccard={jaccard:.3f})"

        return reward, {
            "execution_match": reward,
            "jaccard": jaccard,
            "pred_rows": pred_result.n_rows,
            "gold_rows": gold_result.n_rows,
            "pred_rows_no_limit": pred_result.row_count_no_limit,
            "reason": reason,
        }

    @classmethod
    def from_config(cls, cfg: dict, db_path: str) -> "ExecutionMatchReward":
        em_cfg = cfg.get("execution_match", {})
        executor = SQLExecutor(db_path)
        return cls(
            executor=executor,
            overlap_threshold=em_cfg.get("overlap_threshold", 0.50),
            overlap_reward=em_cfg.get("overlap_reward", 0.50),
            error_penalty=em_cfg.get("error_penalty", -0.10),
        )
