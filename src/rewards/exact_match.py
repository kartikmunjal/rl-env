"""
exact_match.py — R1: Exact match reward with SQL normalisation.

Design rationale (see research_notes/design_decisions.md §R1):

Exact match is the STRICTEST reward signal. It requires the predicted SQL
to be structurally identical to the gold SQL after normalisation.

Normalisation pipeline (applied to both predicted and gold before comparison):
    1. Lowercase
    2. Collapse whitespace
    3. Strip table aliases (c., o., p., oi.)
    4. Strip column aliases (AS alias_name)
    5. Sort comma-separated SELECT columns (column ordering invariant)

Reward: 1.0 if normalised strings are equal, 0.0 otherwise.

Vulnerability to reward hacking:
    H1.1 — Alias stripping can make a cross-join appear identical to an INNER JOIN
            in some normalisation configurations. Documented with a concrete example
            in reward_hacking_report.md §H1.1.
    H1.2 — If SELECT * is a valid action and normalisation expands * to all columns,
            the agent can always SELECT * and get 1.0 on single-table queries.
            Mitigated by restricting * in the action mask.

Use case: evaluation metric (not training signal — too sparse).

Author: Kartik Munjal
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional


@dataclass
class ExactMatchReward:
    """
    Callable reward function for exact match.

    Call signature:
        reward, components = reward_fn(predicted_sql, gold_sql, db_path, schema)
    """

    lowercase: bool = True
    strip_aliases: bool = True
    sort_select_cols: bool = True
    collapse_whitespace: bool = True

    def __call__(
        self,
        predicted_sql: str,
        gold_sql: str,
        db_path: Optional[str] = None,
        schema=None,
    ) -> tuple[float, dict]:
        norm_pred = self._normalise(predicted_sql)
        norm_gold = self._normalise(gold_sql)
        match = norm_pred == norm_gold
        reward = 1.0 if match else 0.0
        return reward, {
            "exact_match": reward,
            "normalised_pred": norm_pred,
            "normalised_gold": norm_gold,
        }

    # ------------------------------------------------------------------
    # Normalisation
    # ------------------------------------------------------------------
    def _normalise(self, sql: str) -> str:
        if not sql or sql.startswith("<INVALID"):
            return "<INVALID>"

        s = sql

        if self.lowercase:
            s = s.lower()

        if self.collapse_whitespace:
            s = re.sub(r"\s+", " ", s).strip()

        if self.strip_aliases:
            # Remove single-letter table aliases: "c.", "o.", "p.", "oi."
            s = re.sub(r"\b[a-z]{1,2}\.", "", s)
            # Remove AS alias clauses: "as alias_name"
            s = re.sub(r"\bas\s+\w+", "", s)
            s = re.sub(r"\s+", " ", s).strip()

        if self.sort_select_cols:
            s = self._sort_select_columns(s)

        return s

    @staticmethod
    def _sort_select_columns(sql: str) -> str:
        """
        Sort the SELECT column list alphabetically.
        This makes column ordering invariant, which is correct semantics
        since SQL SELECT projection ordering is arbitrary for our purposes.

        Only sorts the outermost SELECT; subquery SELECTs are left as-is.
        """
        m = re.match(r"^(select\s+)(.*?)(\s+from\b.*)", sql, re.IGNORECASE | re.DOTALL)
        if not m:
            return sql
        select_kw = m.group(1)
        col_str = m.group(2)
        rest = m.group(3)

        cols = [c.strip() for c in col_str.split(",")]
        cols_sorted = sorted(cols)
        return select_kw + ", ".join(cols_sorted) + rest

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------
    @classmethod
    def from_config(cls, cfg: dict) -> "ExactMatchReward":
        em_cfg = cfg.get("exact_match", {})
        return cls(
            lowercase=em_cfg.get("lowercase", True),
            strip_aliases=em_cfg.get("strip_aliases", True),
            sort_select_cols=em_cfg.get("sort_select_cols", True),
            collapse_whitespace=em_cfg.get("collapse_whitespace", True),
        )
