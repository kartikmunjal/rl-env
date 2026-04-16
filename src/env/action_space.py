"""
action_space.py — Hierarchical action space with schema-grounded masking.

Core design decisions (see research_notes/design_decisions.md §D4):

1. The action space is FLAT: a single Gymnasium Discrete(N_MAX_ACTIONS) space.
   The meaning of each action integer is phase-dependent (re-indexed per phase).
   This avoids a MultiDiscrete space, which Gymnasium handles less cleanly.

2. Action masking is done at the ENVIRONMENT level, not the agent level.
   Valid actions are provided via `info["action_mask"]` (standard convention).
   Selecting a masked action raises AssertionError — this surfaces bugs early.

3. Schema-grounding: masks are derived directly from the Schema object.
   - FROM_TABLE phase: only tables in schema are valid
   - WHERE_COL phase: only columns of the current FROM + JOIN tables
   - JOIN_KEY_LEFT/RIGHT: only FK-compatible column pairs
   This eliminates an entire class of reward hacking (schema-invalid SQL).

Phase ordering matches task_config.yaml. Not all phases are active for every task.
Active phases are determined by SQLTask.active_phases at task load time.

Author: Kartik Munjal
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
from typing import Optional

import numpy as np

from .schema import Schema, ColumnInfo, DType


# ---------------------------------------------------------------------------
# Phase definitions
# ---------------------------------------------------------------------------
class BuildPhase(IntEnum):
    """
    All possible build phases across all 5 tasks.
    Tasks activate a subset of these phases.
    Phase values are contiguous integers for easy indexing into partial_slots.
    """
    SELECT_COLS         = 0
    FROM_TABLE          = 1
    JOIN_TABLE          = 2
    JOIN_KEY_LEFT       = 3
    JOIN_KEY_RIGHT      = 4
    WHERE_COL           = 5
    WHERE_OP            = 6
    WHERE_VAL           = 7
    AGG_FN              = 8
    GROUP_BY_COL        = 9
    HAVING_OP           = 10
    HAVING_VAL          = 11
    ORDER_BY_COL        = 12
    ORDER_DIR           = 13
    WINDOW_FN           = 14
    WINDOW_PARTITION_COL = 15
    WINDOW_ORDER_COL    = 16
    WINDOW_ORDER_DIR    = 17
    SUBQ_SELECT_COL     = 18
    SUBQ_FROM_TABLE     = 19
    SUBQ_JOIN_TABLE     = 20
    SUBQ_JOIN_KEY_LEFT  = 21
    SUBQ_JOIN_KEY_RIGHT = 22
    SUBQ_WHERE_COL      = 23
    WHERE_OP_IN         = 24   # Task 4: fixed IN operator
    DONE                = 25


N_PHASES = len(BuildPhase)
N_MAX_ACTIONS = 32    # Maximum actions per phase (padded)

# ---------------------------------------------------------------------------
# Fixed-vocabulary action lists (phase → ordered list of string tokens)
# ---------------------------------------------------------------------------
WHERE_OPERATORS = ["=", "!=", ">", "<", ">=", "<=", "LIKE", "IN", "NOT IN"]
AGG_FUNCTIONS = ["COUNT", "SUM", "AVG", "MIN", "MAX"]
ORDER_DIRS = ["ASC", "DESC"]
WINDOW_FUNCTIONS = ["RANK", "ROW_NUMBER", "DENSE_RANK"]
BOOLEAN_VALUES = ["true", "false"]  # for "is there a WHERE?" style phases

# Numeric threshold values available for HAVING_VAL and WHERE_VAL (numeric)
NUMERIC_THRESHOLDS = [1, 2, 3, 5, 10, 20, 50, 100, 200, 500]


# ---------------------------------------------------------------------------
# Action metadata
# ---------------------------------------------------------------------------
@dataclass
class ActionSpec:
    """
    Describes the legal actions for a given phase in a given task context.

    tokens: human-readable labels for each action index (used for decoding)
    mask:   binary np.ndarray of shape (N_MAX_ACTIONS,); 1 = valid, 0 = masked
    n_valid: number of valid actions (sum of mask)
    """
    phase: BuildPhase
    tokens: list[str]
    mask: np.ndarray

    @property
    def n_valid(self) -> int:
        return int(self.mask.sum())

    def decode(self, action_idx: int) -> str:
        """Map a raw action integer to its token string."""
        if action_idx < 0 or action_idx >= len(self.tokens):
            raise ValueError(f"Action {action_idx} out of range [0, {len(self.tokens)})")
        if self.mask[action_idx] == 0:
            raise ValueError(f"Action {action_idx} is masked (invalid) for phase {self.phase.name}")
        return self.tokens[action_idx]


# ---------------------------------------------------------------------------
# Action space builder
# ---------------------------------------------------------------------------
class HierarchicalActionSpace:
    """
    Builds per-phase ActionSpecs grounded in a Schema and current partial state.

    The key invariant:
        For any phase, get_action_spec() returns exactly the legal actions
        given the schema AND the slots already filled in earlier phases.

    This means the mask changes as the episode progresses:
        - After FROM_TABLE = 'customers', WHERE_COL shows only customers columns
        - After JOIN_TABLE = 'orders', WHERE_COL adds orders columns too
    """

    def __init__(self, schema: Schema) -> None:
        self.schema = schema
        self._table_names = schema.table_names()

    def get_action_spec(
        self,
        phase: BuildPhase,
        partial_slots: dict,
        task_active_phases: list[BuildPhase],
    ) -> ActionSpec:
        """
        Build the ActionSpec for `phase` given the current partial_slots.

        partial_slots maps BuildPhase → chosen value (or None if not yet filled).
        """
        builders = {
            BuildPhase.SELECT_COLS:          self._select_cols_spec,
            BuildPhase.FROM_TABLE:           self._from_table_spec,
            BuildPhase.JOIN_TABLE:           self._join_table_spec,
            BuildPhase.JOIN_KEY_LEFT:        self._join_key_left_spec,
            BuildPhase.JOIN_KEY_RIGHT:       self._join_key_right_spec,
            BuildPhase.WHERE_COL:            self._where_col_spec,
            BuildPhase.WHERE_OP:             self._where_op_spec,
            BuildPhase.WHERE_VAL:            self._where_val_spec,
            BuildPhase.WHERE_OP_IN:          self._where_op_in_spec,
            BuildPhase.AGG_FN:               self._agg_fn_spec,
            BuildPhase.GROUP_BY_COL:         self._group_by_col_spec,
            BuildPhase.HAVING_OP:            self._having_op_spec,
            BuildPhase.HAVING_VAL:           self._having_val_spec,
            BuildPhase.ORDER_BY_COL:         self._order_by_col_spec,
            BuildPhase.ORDER_DIR:            self._order_dir_spec,
            BuildPhase.WINDOW_FN:            self._window_fn_spec,
            BuildPhase.WINDOW_PARTITION_COL: self._window_partition_spec,
            BuildPhase.WINDOW_ORDER_COL:     self._window_order_col_spec,
            BuildPhase.WINDOW_ORDER_DIR:     self._order_dir_spec,
            BuildPhase.SUBQ_SELECT_COL:      self._subq_select_col_spec,
            BuildPhase.SUBQ_FROM_TABLE:      self._subq_from_table_spec,
            BuildPhase.SUBQ_JOIN_TABLE:      self._subq_join_table_spec,
            BuildPhase.SUBQ_JOIN_KEY_LEFT:   self._subq_join_key_left_spec,
            BuildPhase.SUBQ_JOIN_KEY_RIGHT:  self._subq_join_key_right_spec,
            BuildPhase.SUBQ_WHERE_COL:       self._subq_where_col_spec,
        }
        builder = builders.get(phase)
        if builder is None:
            raise ValueError(f"No action spec builder for phase {phase}")
        return builder(partial_slots)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _make_spec(self, phase: BuildPhase, tokens: list[str]) -> ActionSpec:
        mask = np.zeros(N_MAX_ACTIONS, dtype=np.float32)
        for i in range(min(len(tokens), N_MAX_ACTIONS)):
            mask[i] = 1.0
        padded = tokens[:N_MAX_ACTIONS] + ["<PAD>"] * max(0, N_MAX_ACTIONS - len(tokens))
        return ActionSpec(phase=phase, tokens=padded, mask=mask)

    def _cols_for_tables(self, *table_names: str) -> list[str]:
        """Qualified column names (table.column) for given tables."""
        cols = []
        for t in table_names:
            for c in self.schema.columns_of(t):
                cols.append(f"{t}.{c.column}")
        return cols

    # ------------------------------------------------------------------
    # Phase builders
    # ------------------------------------------------------------------
    def _select_cols_spec(self, slots: dict) -> ActionSpec:
        """SELECT_COLS: all columns of the from_table (+ join_table if present)."""
        from_table = slots.get(BuildPhase.FROM_TABLE) or self._table_names[0]
        join_table = slots.get(BuildPhase.JOIN_TABLE)
        tables = [from_table] + ([join_table] if join_table else [])
        cols = self._cols_for_tables(*tables)
        # Add '*' shorthand — only valid if schema allows it for this task
        tokens = ["*"] + cols
        return self._make_spec(BuildPhase.SELECT_COLS, tokens)

    def _from_table_spec(self, slots: dict) -> ActionSpec:
        return self._make_spec(BuildPhase.FROM_TABLE, self._table_names)

    def _join_table_spec(self, slots: dict) -> ActionSpec:
        """JOIN_TABLE: tables that share an FK with the FROM_TABLE."""
        from_table = slots.get(BuildPhase.FROM_TABLE)
        if not from_table:
            return self._make_spec(BuildPhase.JOIN_TABLE, self._table_names)
        joinable = [
            t for t in self._table_names
            if t != from_table and self.schema.are_joinable(from_table, t)
        ]
        return self._make_spec(BuildPhase.JOIN_TABLE, joinable or self._table_names)

    def _join_key_left_spec(self, slots: dict) -> ActionSpec:
        """JOIN_KEY_LEFT: FK columns in FROM_TABLE that link to JOIN_TABLE."""
        from_table = slots.get(BuildPhase.FROM_TABLE)
        join_table = slots.get(BuildPhase.JOIN_TABLE)
        if from_table and join_table:
            pairs = self.schema.join_keys(from_table, join_table)
            cols = [p[0] for p in pairs] or [c.column for c in self.schema.columns_of(from_table)]
        else:
            cols = [c.column for c in self.schema.columns_of(from_table or self._table_names[0])]
        return self._make_spec(BuildPhase.JOIN_KEY_LEFT, cols)

    def _join_key_right_spec(self, slots: dict) -> ActionSpec:
        """JOIN_KEY_RIGHT: matching FK columns in JOIN_TABLE."""
        from_table = slots.get(BuildPhase.FROM_TABLE)
        join_table = slots.get(BuildPhase.JOIN_TABLE)
        if from_table and join_table:
            pairs = self.schema.join_keys(from_table, join_table)
            cols = [p[1] for p in pairs] or [c.column for c in self.schema.columns_of(join_table)]
        else:
            cols = [c.column for c in self.schema.columns_of(join_table or self._table_names[0])]
        return self._make_spec(BuildPhase.JOIN_KEY_RIGHT, cols)

    def _where_col_spec(self, slots: dict) -> ActionSpec:
        """WHERE_COL: columns across FROM + JOIN tables (type-safe join masking)."""
        from_table = slots.get(BuildPhase.FROM_TABLE)
        join_table = slots.get(BuildPhase.JOIN_TABLE)
        tables = ([from_table] if from_table else []) + ([join_table] if join_table else [])
        if not tables:
            tables = self._table_names[:1]
        # "NONE" allows skipping the WHERE clause
        cols = ["<NONE>"] + self._cols_for_tables(*tables)
        return self._make_spec(BuildPhase.WHERE_COL, cols)

    def _where_op_spec(self, slots: dict) -> ActionSpec:
        """WHERE_OP: operator selection; restrict to = for text columns."""
        where_col_raw = slots.get(BuildPhase.WHERE_COL)
        if where_col_raw and "." in where_col_raw:
            table, col_name = where_col_raw.split(".", 1)
            col_info = self.schema.get_table(table)
            if col_info:
                ci = col_info.get_column(col_name)
                if ci and ci.dtype == DType.TEXT:
                    # Text columns: restrict to =, !=, LIKE (no >, <)
                    return self._make_spec(BuildPhase.WHERE_OP, ["=", "!=", "LIKE"])
        return self._make_spec(BuildPhase.WHERE_OP, WHERE_OPERATORS[:7])

    def _where_val_spec(self, slots: dict) -> ActionSpec:
        """WHERE_VAL: distinct domain values for the chosen WHERE column."""
        where_col_raw = slots.get(BuildPhase.WHERE_COL)
        if not where_col_raw or where_col_raw == "<NONE>":
            return self._make_spec(BuildPhase.WHERE_VAL, ["<SKIP>"])
        if "." in where_col_raw:
            table, col_name = where_col_raw.split(".", 1)
            col_info_tbl = self.schema.get_table(table)
            if col_info_tbl:
                ci = col_info_tbl.get_column(col_name)
                if ci and ci.dtype in (DType.INT, DType.FLOAT):
                    return self._make_spec(
                        BuildPhase.WHERE_VAL,
                        [str(v) for v in NUMERIC_THRESHOLDS],
                    )
                else:
                    vals = self.schema.distinct_values(table, col_name)
                    return self._make_spec(BuildPhase.WHERE_VAL, [str(v) for v in vals])
        return self._make_spec(BuildPhase.WHERE_VAL, ["<UNKNOWN>"])

    def _where_op_in_spec(self, slots: dict) -> ActionSpec:
        """WHERE_OP_IN: fixed IN operator (Task 4 only)."""
        return self._make_spec(BuildPhase.WHERE_OP_IN, ["IN"])

    def _agg_fn_spec(self, slots: dict) -> ActionSpec:
        return self._make_spec(BuildPhase.AGG_FN, AGG_FUNCTIONS)

    def _group_by_col_spec(self, slots: dict) -> ActionSpec:
        from_table = slots.get(BuildPhase.FROM_TABLE)
        cols = ["<NONE>"] + [c.column for c in self.schema.columns_of(from_table or "")]
        return self._make_spec(BuildPhase.GROUP_BY_COL, cols)

    def _having_op_spec(self, slots: dict) -> ActionSpec:
        return self._make_spec(BuildPhase.HAVING_OP, ["<NONE>"] + WHERE_OPERATORS[:6])

    def _having_val_spec(self, slots: dict) -> ActionSpec:
        return self._make_spec(
            BuildPhase.HAVING_VAL,
            ["<NONE>"] + [str(v) for v in NUMERIC_THRESHOLDS],
        )

    def _order_by_col_spec(self, slots: dict) -> ActionSpec:
        from_table = slots.get(BuildPhase.FROM_TABLE)
        cols = ["<NONE>", "agg_val"] + [c.column for c in self.schema.columns_of(from_table or "")]
        return self._make_spec(BuildPhase.ORDER_BY_COL, cols)

    def _order_dir_spec(self, slots: dict) -> ActionSpec:
        return self._make_spec(BuildPhase.ORDER_DIR, ORDER_DIRS)

    def _window_fn_spec(self, slots: dict) -> ActionSpec:
        return self._make_spec(BuildPhase.WINDOW_FN, WINDOW_FUNCTIONS)

    def _window_partition_spec(self, slots: dict) -> ActionSpec:
        from_table = slots.get(BuildPhase.FROM_TABLE)
        cols = [c.column for c in self.schema.columns_of(from_table or "")]
        return self._make_spec(BuildPhase.WINDOW_PARTITION_COL, cols)

    def _window_order_col_spec(self, slots: dict) -> ActionSpec:
        from_table = slots.get(BuildPhase.FROM_TABLE)
        cols = [c.column for c in self.schema.columns_of(from_table or "")]
        return self._make_spec(BuildPhase.WINDOW_ORDER_COL, cols)

    # Subquery phases mirror outer phases but use SUBQ_ slot keys
    def _subq_select_col_spec(self, slots: dict) -> ActionSpec:
        subq_from = slots.get(BuildPhase.SUBQ_FROM_TABLE)
        if subq_from:
            cols = [c.column for c in self.schema.columns_of(subq_from)]
        else:
            # Fallback: all columns across all tables (SUBQ_FROM_TABLE not yet filled)
            cols = [c.column for t in self._table_names for c in self.schema.columns_of(t)]
        cols = list(dict.fromkeys(cols)) or ["customer_id"]  # deduplicate, ensure non-empty
        return self._make_spec(BuildPhase.SUBQ_SELECT_COL, cols)

    def _subq_from_table_spec(self, slots: dict) -> ActionSpec:
        return self._make_spec(BuildPhase.SUBQ_FROM_TABLE, self._table_names)

    def _subq_join_table_spec(self, slots: dict) -> ActionSpec:
        subq_from = slots.get(BuildPhase.SUBQ_FROM_TABLE)
        joinable = [
            t for t in self._table_names
            if t != subq_from and self.schema.are_joinable(subq_from or "", t)
        ] if subq_from else self._table_names
        return self._make_spec(BuildPhase.SUBQ_JOIN_TABLE, joinable or self._table_names)

    def _subq_join_key_left_spec(self, slots: dict) -> ActionSpec:
        subq_from = slots.get(BuildPhase.SUBQ_FROM_TABLE)
        subq_join = slots.get(BuildPhase.SUBQ_JOIN_TABLE)
        if subq_from and subq_join:
            pairs = self.schema.join_keys(subq_from, subq_join)
            cols = [p[0] for p in pairs] or [c.column for c in self.schema.columns_of(subq_from)]
        else:
            cols = [c.column for c in self.schema.columns_of(subq_from or self._table_names[0])]
        return self._make_spec(BuildPhase.SUBQ_JOIN_KEY_LEFT, cols)

    def _subq_join_key_right_spec(self, slots: dict) -> ActionSpec:
        subq_from = slots.get(BuildPhase.SUBQ_FROM_TABLE)
        subq_join = slots.get(BuildPhase.SUBQ_JOIN_TABLE)
        if subq_from and subq_join:
            pairs = self.schema.join_keys(subq_from, subq_join)
            cols = [p[1] for p in pairs] or [c.column for c in self.schema.columns_of(subq_join)]
        else:
            cols = [c.column for c in self.schema.columns_of(subq_join or self._table_names[0])]
        return self._make_spec(BuildPhase.SUBQ_JOIN_KEY_RIGHT, cols)

    def _subq_where_col_spec(self, slots: dict) -> ActionSpec:
        subq_from = slots.get(BuildPhase.SUBQ_FROM_TABLE)
        subq_join = slots.get(BuildPhase.SUBQ_JOIN_TABLE)
        tables = ([subq_from] if subq_from else []) + ([subq_join] if subq_join else [])
        cols = ["<NONE>"] + self._cols_for_tables(*tables)
        return self._make_spec(BuildPhase.SUBQ_WHERE_COL, cols)
