"""
partial_credit.py — R3: Component-wise partial credit reward.

Design rationale (see research_notes/design_decisions.md §R3):

Partial credit decomposes SQL into semantic components and scores each
independently. The final score is a weighted sum in [0.0, 1.0].

Components and weights (default from reward_config.yaml):
    tables       0.20 — correct FROM/JOIN tables
    columns      0.25 — correct projected columns
    where        0.20 — correct WHERE column + operator
    aggregation  0.15 — correct aggregate function
    group_by     0.10 — correct GROUP BY column
    order_by     0.10 — ORDER BY column + direction

Overlap metric: JACCARD (|∩|/|∪|) for set-valued components (columns, tables).
Critical implementation detail: the denominator is |∪|, not |predicted|.
Using |predicted| (recall) creates Hacking Scenario H3.1 (column flooding).
The Jaccard metric penalises over-selection, preventing the attack.

Vulnerability to reward hacking:
    H3.1 — Column flooding: selecting ALL columns maximises |∩| but Jaccard
            still penalises via large |∪|. Documented with exact calculation.
    H3.2 — WHERE operator gaming: "WHERE col >= 0" for numeric columns always
            matches a filter condition, earning 0.5 of the where_score.
            Mitigated by tracking operator diversity (hacking_detector.py).

Use case: dense training signal in early episodes; ablation comparisons.

Parser strategy: sqlparse for structured extraction, regex fallback.

Author: Kartik Munjal
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional


def _extract_tables(sql: str) -> set[str]:
    """Extract table names from FROM and JOIN clauses."""
    tables = set()
    # FROM table [alias]
    for m in re.finditer(r"\bFROM\s+(\w+)", sql, re.IGNORECASE):
        tables.add(m.group(1).lower())
    # JOIN table [alias]
    for m in re.finditer(r"\bJOIN\s+(\w+)", sql, re.IGNORECASE):
        tables.add(m.group(1).lower())
    return tables


def _extract_columns(sql: str) -> set[str]:
    """
    Extract column names from the SELECT clause.
    Strips table aliases and AS aliases.
    Returns lowercase set of bare column names.
    """
    m = re.match(r"SELECT\s+(.*?)\s+FROM\b", sql, re.IGNORECASE | re.DOTALL)
    if not m:
        return set()
    col_str = m.group(1)
    cols = set()
    for raw_col in col_str.split(","):
        raw_col = raw_col.strip()
        # Remove AS alias
        raw_col = re.sub(r"\s+AS\s+\w+", "", raw_col, flags=re.IGNORECASE).strip()
        # Remove aggregate wrapper: COUNT(*), SUM(col), etc.
        agg_m = re.match(r"\w+\((.+)\)", raw_col)
        if agg_m:
            raw_col = agg_m.group(1).strip()
        # Remove table alias prefix
        if "." in raw_col:
            raw_col = raw_col.split(".")[-1]
        if raw_col and raw_col != "*":
            cols.add(raw_col.lower())
    return cols


def _extract_where(sql: str) -> Optional[tuple[str, str]]:
    """
    Extract the (column, operator) from the WHERE clause.
    Returns None if no WHERE clause found.
    Handles both "WHERE col op val" and "WHERE col IN (...)".
    """
    m = re.search(
        r"\bWHERE\s+(\w+(?:\.\w+)?)\s*(=|!=|<>|>=|<=|>|<|LIKE|IN|NOT\s+IN)\b",
        sql,
        re.IGNORECASE,
    )
    if not m:
        return None
    col = m.group(1).lower()
    if "." in col:
        col = col.split(".")[-1]
    op = m.group(2).upper().replace("  ", " ")
    return col, op


def _extract_aggregation(sql: str) -> Optional[str]:
    """Extract the primary aggregate function used (COUNT, SUM, AVG, MAX, MIN)."""
    m = re.search(r"\b(COUNT|SUM|AVG|MAX|MIN)\s*\(", sql, re.IGNORECASE)
    return m.group(1).upper() if m else None


def _extract_group_by(sql: str) -> Optional[str]:
    """Extract the GROUP BY column."""
    m = re.search(r"\bGROUP\s+BY\s+(\w+(?:\.\w+)?)", sql, re.IGNORECASE)
    if not m:
        return None
    col = m.group(1).lower()
    return col.split(".")[-1] if "." in col else col


def _extract_order_by(sql: str) -> Optional[tuple[str, str]]:
    """Extract (column, direction) from ORDER BY clause."""
    m = re.search(
        r"\bORDER\s+BY\s+(\w+(?:\.\w+)?)\s*(ASC|DESC)?",
        sql,
        re.IGNORECASE,
    )
    if not m:
        return None
    col = m.group(1).lower().split(".")[-1]
    direction = (m.group(2) or "ASC").upper()
    return col, direction


def _jaccard(set_a: set, set_b: set) -> float:
    """
    Jaccard similarity: |A ∩ B| / |A ∪ B|.

    CRITICAL: denominator is |A ∪ B|, not |A| (precision) or |B| (recall).
    Using recall would create Hacking Scenario H3.1 (column flooding).
    See reward_hacking_report.md §H3.1 for the mathematical proof.
    """
    if not set_a and not set_b:
        return 1.0
    union = len(set_a | set_b)
    if union == 0:
        return 0.0
    return len(set_a & set_b) / union


@dataclass
class ComponentWeights:
    tables: float = 0.20
    columns: float = 0.25
    where: float = 0.20
    aggregation: float = 0.15
    group_by: float = 0.10
    order_by: float = 0.10

    def validate(self) -> None:
        total = sum([
            self.tables, self.columns, self.where,
            self.aggregation, self.group_by, self.order_by
        ])
        assert abs(total - 1.0) < 1e-6, f"Component weights must sum to 1.0, got {total:.3f}"


@dataclass
class PartialCreditReward:
    weights: ComponentWeights = field(default_factory=ComponentWeights)

    def __call__(
        self,
        predicted_sql: str,
        gold_sql: str,
        db_path=None,
        schema=None,
    ) -> tuple[float, dict]:
        if predicted_sql.startswith("<INVALID"):
            return 0.0, {"partial_credit": 0.0, "reason": "invalid_sql"}

        components = self._score_all(predicted_sql, gold_sql)
        w = self.weights
        total = (
            w.tables * components["tables"]
            + w.columns * components["columns"]
            + w.where * components["where"]
            + w.aggregation * components["aggregation"]
            + w.group_by * components["group_by"]
            + w.order_by * components["order_by"]
        )
        return total, {"partial_credit": total, **components}

    def _score_all(self, pred: str, gold: str) -> dict[str, float]:
        """Score each SQL component independently."""
        scores: dict[str, float] = {}

        # Tables (Jaccard)
        pred_tables = _extract_tables(pred)
        gold_tables = _extract_tables(gold)
        scores["tables"] = _jaccard(pred_tables, gold_tables)

        # Columns (Jaccard)
        pred_cols = _extract_columns(pred)
        gold_cols = _extract_columns(gold)
        scores["columns"] = _jaccard(pred_cols, gold_cols)

        # WHERE condition
        pred_where = _extract_where(pred)
        gold_where = _extract_where(gold)
        if gold_where is None:
            # Gold has no WHERE; we reward the agent for also having no WHERE
            scores["where"] = 1.0 if pred_where is None else 0.0
        elif pred_where is None:
            scores["where"] = 0.0
        else:
            col_match = float(pred_where[0] == gold_where[0])
            op_match = float(pred_where[1] == gold_where[1])
            # Column is more important than operator
            scores["where"] = 0.6 * col_match + 0.4 * op_match

        # Aggregation
        pred_agg = _extract_aggregation(pred)
        gold_agg = _extract_aggregation(gold)
        if gold_agg is None:
            scores["aggregation"] = 1.0 if pred_agg is None else 0.0
        else:
            scores["aggregation"] = 1.0 if pred_agg == gold_agg else 0.0

        # GROUP BY
        pred_grp = _extract_group_by(pred)
        gold_grp = _extract_group_by(gold)
        if gold_grp is None:
            scores["group_by"] = 1.0 if pred_grp is None else 0.0
        else:
            scores["group_by"] = 1.0 if pred_grp == gold_grp else 0.0

        # ORDER BY
        pred_ord = _extract_order_by(pred)
        gold_ord = _extract_order_by(gold)
        if gold_ord is None:
            scores["order_by"] = 1.0 if pred_ord is None else 0.0
        elif pred_ord is None:
            scores["order_by"] = 0.0
        else:
            col_match = float(pred_ord[0] == gold_ord[0])
            dir_match = float(pred_ord[1] == gold_ord[1])
            scores["order_by"] = 0.7 * col_match + 0.3 * dir_match

        return scores

    @classmethod
    def from_config(cls, cfg: dict) -> "PartialCreditReward":
        pc_cfg = cfg.get("partial_credit", {})
        w_cfg = pc_cfg.get("weights", {})
        weights = ComponentWeights(
            tables=w_cfg.get("tables", 0.20),
            columns=w_cfg.get("columns", 0.25),
            where=w_cfg.get("where", 0.20),
            aggregation=w_cfg.get("aggregation", 0.15),
            group_by=w_cfg.get("group_by", 0.10),
            order_by=w_cfg.get("order_by", 0.10),
        )
        weights.validate()
        return cls(weights=weights)
