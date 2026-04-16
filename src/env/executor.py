"""
executor.py — Safe SQLite query execution with timeout and row-count guards.

Design decisions documented here:
  - Timeout is enforced via a threading interrupt (SQLite has no native timeout)
  - Row count for efficiency penalty is computed on a COUNT(*) wrapper,
    BEFORE any LIMIT clause — this prevents H4.2 (LIMIT-based penalty suppression)
  - Syntax errors and timeouts yield (None, ExecutionError) not exceptions,
    so callers can distinguish "ran and produced result" from "failed to run"

Author: Kartik Munjal
"""

from __future__ import annotations

import sqlite3
import threading
from dataclasses import dataclass
from enum import Enum, auto
from typing import Optional

import pandas as pd


class ExecutionStatus(Enum):
    OK = auto()
    SYNTAX_ERROR = auto()
    TIMEOUT = auto()
    EMPTY_RESULT = auto()
    ROW_LIMIT_EXCEEDED = auto()


@dataclass
class ExecutionResult:
    status: ExecutionStatus
    rows: Optional[pd.DataFrame] = None
    row_count_no_limit: int = 0    # COUNT(*) of un-LIMITed query
    error_msg: str = ""

    @property
    def success(self) -> bool:
        return self.status == ExecutionStatus.OK

    @property
    def n_rows(self) -> int:
        return len(self.rows) if self.rows is not None else 0


class SQLExecutor:
    """
    Thread-safe SQLite executor with configurable timeout and row limits.

    Usage:
        executor = SQLExecutor("data/ecommerce.db", timeout=2.0, max_rows=10000)
        result = executor.run("SELECT * FROM customers WHERE tier='gold'")
    """

    def __init__(
        self,
        db_path: str,
        timeout_seconds: float = 2.0,
        max_rows: int = 10_000,
    ) -> None:
        self.db_path = db_path
        self.timeout_seconds = timeout_seconds
        self.max_rows = max_rows

    def run(self, sql: str) -> ExecutionResult:
        """
        Execute `sql` and return an ExecutionResult.
        Never raises — all errors are captured in the result.
        """
        result_holder: list[ExecutionResult] = []
        error_holder: list[str] = []

        def _target():
            try:
                conn = sqlite3.connect(self.db_path)
                conn.row_factory = sqlite3.Row

                # Execute main query
                cursor = conn.execute(sql)
                column_names = [d[0] for d in cursor.description] if cursor.description else []
                raw_rows = cursor.fetchmany(self.max_rows + 1)

                if len(raw_rows) > self.max_rows:
                    result_holder.append(ExecutionResult(
                        status=ExecutionStatus.ROW_LIMIT_EXCEEDED,
                        error_msg=f"Query returned more than {self.max_rows} rows",
                    ))
                    conn.close()
                    return

                df = pd.DataFrame(
                    [dict(zip(column_names, r)) for r in raw_rows],
                    columns=column_names,
                )

                # Count rows WITHOUT limit (counter-measure for H4.2)
                count_no_limit = self._count_without_limit(conn, sql)

                conn.close()

                status = ExecutionStatus.EMPTY_RESULT if len(df) == 0 else ExecutionStatus.OK
                result_holder.append(ExecutionResult(
                    status=status,
                    rows=df,
                    row_count_no_limit=count_no_limit,
                ))

            except sqlite3.OperationalError as e:
                error_holder.append(str(e))
            except Exception as e:
                error_holder.append(f"Unexpected error: {e}")

        t = threading.Thread(target=_target, daemon=True)
        t.start()
        t.join(timeout=self.timeout_seconds)

        if t.is_alive():
            return ExecutionResult(
                status=ExecutionStatus.TIMEOUT,
                error_msg=f"Query timed out after {self.timeout_seconds}s",
            )
        if error_holder:
            return ExecutionResult(
                status=ExecutionStatus.SYNTAX_ERROR,
                error_msg=error_holder[0],
            )
        if result_holder:
            return ExecutionResult(
                status=result_holder[0].status,
                rows=result_holder[0].rows,
                row_count_no_limit=result_holder[0].row_count_no_limit,
                error_msg=result_holder[0].error_msg,
            )

        return ExecutionResult(
            status=ExecutionStatus.SYNTAX_ERROR,
            error_msg="No result produced (unknown error)",
        )

    def _count_without_limit(self, conn: sqlite3.Connection, sql: str) -> int:
        """
        Wrap the query in a COUNT(*) subquery after stripping any LIMIT clause.
        This gives the true result-set size before LIMIT truncation.

        Counter-measure for Hacking Scenario H4.2 (LIMIT-based penalty suppression):
        an agent that adds LIMIT 100 to suppress efficiency penalty will still be
        charged based on the full COUNT(*).
        """
        try:
            stripped = self._strip_limit(sql)
            count_sql = f"SELECT COUNT(*) FROM ({stripped}) _subq"
            row = conn.execute(count_sql).fetchone()
            return row[0] if row else 0
        except Exception:
            return 0

    @staticmethod
    def _strip_limit(sql: str) -> str:
        """Remove trailing LIMIT N [OFFSET M] clause from SQL."""
        import re
        return re.sub(r"\bLIMIT\s+\d+(\s+OFFSET\s+\d+)?\s*$", "", sql, flags=re.IGNORECASE).strip()

    def compare_results(
        self,
        result_a: ExecutionResult,
        result_b: ExecutionResult,
    ) -> tuple[bool, float]:
        """
        Compare two execution results as frozensets of row tuples (order-invariant).

        Returns:
            (exact_match: bool, jaccard_overlap: float)

        The Jaccard overlap is used by execution_match reward for partial scoring.
        Order-invariance is intentional: SQL ORDER BY is stylistic in set semantics.
        """
        if not result_a.success or not result_b.success:
            return False, 0.0

        def to_frozenset(df: pd.DataFrame) -> frozenset:
            return frozenset(
                tuple(row) for row in df.itertuples(index=False, name=None)
            )

        set_a = to_frozenset(result_a.rows)
        set_b = to_frozenset(result_b.rows)

        if set_a == set_b:
            return True, 1.0

        intersection = len(set_a & set_b)
        union = len(set_a | set_b)
        jaccard = intersection / union if union > 0 else 0.0
        return False, jaccard
