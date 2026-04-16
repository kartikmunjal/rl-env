"""
schema.py — Schema loader and column registry for the ecommerce SQLite database.

The Schema class is the authoritative source of truth for:
  - Which tables / columns exist
  - Column data types (int, float, text, date)
  - Foreign key relationships (used for join key masking)
  - Schema encoding as a numpy array (for RL observations)

All action masking logic in action_space.py is grounded in Schema.

Author: Kartik Munjal
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional

import numpy as np


# ---------------------------------------------------------------------------
# Type system
# ---------------------------------------------------------------------------
class DType(Enum):
    INT = 0
    FLOAT = 1
    TEXT = 2
    DATE = 3

    @classmethod
    def from_sqlite(cls, affinity: str) -> "DType":
        a = affinity.upper()
        if "INT" in a:
            return cls.INT
        if "REAL" in a or "FLOAT" in a or "NUMERIC" in a:
            return cls.FLOAT
        if "DATE" in a or "TIME" in a:
            return cls.DATE
        return cls.TEXT


@dataclass(frozen=True)
class ColumnInfo:
    table: str
    column: str
    dtype: DType
    is_pk: bool = False
    is_fk: bool = False
    fk_references: Optional[tuple[str, str]] = None  # (ref_table, ref_col)

    @property
    def qualified_name(self) -> str:
        return f"{self.table}.{self.column}"

    def feature_vector(self, table_index: int, n_tables: int = 4) -> np.ndarray:
        """
        8-dimensional feature vector per column:
          [0:n_tables]  — one-hot table index
          [n_tables:]   — one-hot dtype (int, float, text, date)

        Design decision: keep encoding small (8 dims) so the agent must
        learn from the NL query rather than memorise schema structure.
        """
        vec = np.zeros(n_tables + 4, dtype=np.float32)
        if 0 <= table_index < n_tables:
            vec[table_index] = 1.0
        vec[n_tables + self.dtype.value] = 1.0
        return vec


@dataclass
class TableInfo:
    name: str
    index: int                            # 0-based index into schema
    columns: list[ColumnInfo] = field(default_factory=list)
    primary_key: Optional[str] = None

    def get_column(self, col_name: str) -> Optional[ColumnInfo]:
        for c in self.columns:
            if c.column == col_name:
                return c
        return None

    def col_names(self) -> list[str]:
        return [c.column for c in self.columns]


# ---------------------------------------------------------------------------
# Foreign key graph
# ---------------------------------------------------------------------------
@dataclass
class ForeignKey:
    from_table: str
    from_col: str
    to_table: str
    to_col: str


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------
class Schema:
    """
    Loaded from a live SQLite database.
    Provides the ground truth for all action masking operations.
    """

    # Maximum columns for the padded encoding (schema_max_cols in env_config)
    MAX_COLS = 20
    N_TABLES = 4   # customers, products, orders, order_items

    def __init__(self, db_path: str) -> None:
        self.db_path = db_path
        self.tables: dict[str, TableInfo] = {}
        self.foreign_keys: list[ForeignKey] = []
        self._col_flat: list[ColumnInfo] = []   # flat list in stable order
        self._load()

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------
    def _load(self) -> None:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row

        # Get table names in a stable order
        tables_raw = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        ).fetchall()
        table_names = [r["name"] for r in tables_raw]

        for idx, tname in enumerate(table_names):
            cols_raw = conn.execute(f"PRAGMA table_info('{tname}')").fetchall()
            fk_raw = conn.execute(f"PRAGMA foreign_key_list('{tname}')").fetchall()

            fk_map: dict[str, tuple[str, str]] = {
                r["from"]: (r["table"], r["to"]) for r in fk_raw
            }

            pk_col = None
            columns: list[ColumnInfo] = []
            for row in cols_raw:
                col_name = row["name"]
                dtype = DType.from_sqlite(row["type"])
                is_pk = bool(row["pk"])
                is_fk = col_name in fk_map
                fk_ref = fk_map.get(col_name)
                if is_pk:
                    pk_col = col_name
                col = ColumnInfo(
                    table=tname,
                    column=col_name,
                    dtype=dtype,
                    is_pk=is_pk,
                    is_fk=is_fk,
                    fk_references=fk_ref,
                )
                columns.append(col)
                self._col_flat.append(col)

            ti = TableInfo(name=tname, index=idx, columns=columns, primary_key=pk_col)
            self.tables[tname] = ti

            # Build FK graph
            for col_name, (ref_table, ref_col) in fk_map.items():
                self.foreign_keys.append(
                    ForeignKey(
                        from_table=tname,
                        from_col=col_name,
                        to_table=ref_table,
                        to_col=ref_col,
                    )
                )

        conn.close()

    # ------------------------------------------------------------------
    # Query helpers
    # ------------------------------------------------------------------
    def table_names(self) -> list[str]:
        return list(self.tables.keys())

    def get_table(self, name: str) -> Optional[TableInfo]:
        return self.tables.get(name)

    def columns_of(self, table: str) -> list[ColumnInfo]:
        t = self.tables.get(table)
        return t.columns if t else []

    def join_keys(self, table_a: str, table_b: str) -> list[tuple[str, str]]:
        """
        Return list of (col_in_a, col_in_b) FK pairs connecting table_a and table_b.
        Checks both directions of the FK relationship.
        """
        pairs = []
        for fk in self.foreign_keys:
            if fk.from_table == table_a and fk.to_table == table_b:
                pairs.append((fk.from_col, fk.to_col))
            elif fk.from_table == table_b and fk.to_table == table_a:
                pairs.append((fk.to_col, fk.from_col))
        return pairs

    def are_joinable(self, table_a: str, table_b: str) -> bool:
        return len(self.join_keys(table_a, table_b)) > 0

    def type_compatible(self, col_a: ColumnInfo, col_b: ColumnInfo) -> bool:
        """Two columns can be joined if their dtypes are compatible."""
        if col_a.dtype == col_b.dtype:
            return True
        # INT and FLOAT are compatible
        numeric = {DType.INT, DType.FLOAT}
        return col_a.dtype in numeric and col_b.dtype in numeric

    # ------------------------------------------------------------------
    # Observation encoding
    # ------------------------------------------------------------------
    def encode(self) -> np.ndarray:
        """
        Returns a (MAX_COLS, 8) float32 array.
        Each row encodes one column (in stable order).
        Rows beyond the actual column count are zero-padded.

        Design decision: fixed-size encoding enables a static observation
        space, which is required by Gymnasium's Dict space definition.
        """
        n_tables = len(self.tables)
        feat_dim = n_tables + 4
        encoding = np.zeros((self.MAX_COLS, feat_dim), dtype=np.float32)

        for i, col in enumerate(self._col_flat):
            if i >= self.MAX_COLS:
                break
            tbl = self.tables.get(col.table)
            table_idx = tbl.index if tbl else 0
            encoding[i] = col.feature_vector(table_idx, n_tables=n_tables)

        return encoding

    # ------------------------------------------------------------------
    # Domain value extraction (for WHERE_VAL masking)
    # ------------------------------------------------------------------
    def distinct_values(self, table: str, column: str, limit: int = 20) -> list:
        """
        Return up to `limit` distinct values for a column.
        Used to build the WHERE_VAL action space for categorical/text columns.
        Numeric columns use a pre-defined range set instead.
        """
        conn = sqlite3.connect(self.db_path)
        rows = conn.execute(
            f"SELECT DISTINCT {column} FROM {table} ORDER BY {column} LIMIT {limit}"
        ).fetchall()
        conn.close()
        return [r[0] for r in rows if r[0] is not None]

    def __repr__(self) -> str:
        parts = []
        for tname, tinfo in self.tables.items():
            col_strs = ", ".join(f"{c.column}({c.dtype.name})" for c in tinfo.columns)
            parts.append(f"  {tname}: [{col_strs}]")
        return "Schema(\n" + "\n".join(parts) + "\n)"
