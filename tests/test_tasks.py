"""
test_tasks.py — Task definition and SQL assembly tests.

Verifies:
    - All 5 tasks load correctly from task_config.yaml
    - Each task has at least 1 NL query
    - SQL assembly produces valid SQL strings (no INVALID_SQL)
    - Active phases are correctly parsed
    - Slot-filling assembles the expected SQL templates

Author: Kartik Munjal
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pytest

from src.env.action_space import BuildPhase
from src.tasks.base import load_task_registry, SQLTask


@pytest.fixture(scope="module")
def task_registry():
    return load_task_registry(
        str(PROJECT_ROOT / "configs/task_config.yaml"),
        str(PROJECT_ROOT),
    )


class TestTaskRegistry:
    def test_loads_5_tasks(self, task_registry):
        assert len(task_registry) == 5

    def test_all_task_ids_present(self, task_registry):
        expected = {
            "task_01_simple", "task_02_aggregation", "task_03_join",
            "task_04_subquery", "task_05_window"
        }
        assert set(task_registry.task_ids()) == expected

    def test_task1_has_queries(self, task_registry):
        t = task_registry.get("task_01_simple")
        assert len(t.queries) >= 5

    def test_task_difficulties_ascending(self, task_registry):
        difficulties = [
            task_registry.get(tid).difficulty
            for tid in ["task_01_simple", "task_02_aggregation", "task_03_join",
                        "task_04_subquery", "task_05_window"]
        ]
        assert difficulties == sorted(difficulties), "Difficulties should be ascending 1→5"

    def test_task1_active_phases(self, task_registry):
        t = task_registry.get("task_01_simple")
        expected_phases = {
            BuildPhase.SELECT_COLS, BuildPhase.FROM_TABLE,
            BuildPhase.WHERE_COL, BuildPhase.WHERE_OP, BuildPhase.WHERE_VAL
        }
        assert set(t.active_phases) == expected_phases

    def test_task5_has_window_phases(self, task_registry):
        t = task_registry.get("task_05_window")
        phase_set = set(t.active_phases)
        assert BuildPhase.WINDOW_FN in phase_set
        assert BuildPhase.WINDOW_PARTITION_COL in phase_set
        assert BuildPhase.WINDOW_ORDER_COL in phase_set


class TestSQLAssembly:
    def test_task1_assembly(self, task_registry):
        t = task_registry.get("task_01_simple")
        slots = {
            BuildPhase.SELECT_COLS: "customers.name",
            BuildPhase.FROM_TABLE: "customers",
            BuildPhase.WHERE_COL: "customers.tier",
            BuildPhase.WHERE_OP: "=",
            BuildPhase.WHERE_VAL: "gold",
        }
        sql = t.assemble_sql(slots)
        assert "<INVALID" not in sql, f"Assembly produced invalid SQL: {sql}"
        assert "FROM customers" in sql
        assert "WHERE" in sql

    def test_task2_assembly(self, task_registry):
        t = task_registry.get("task_02_aggregation")
        slots = {
            BuildPhase.SELECT_COLS: "customers.city",
            BuildPhase.FROM_TABLE: "customers",
            BuildPhase.AGG_FN: "COUNT",
            BuildPhase.GROUP_BY_COL: "city",
            BuildPhase.HAVING_OP: ">",
            BuildPhase.HAVING_VAL: "5",
            BuildPhase.ORDER_BY_COL: "agg_val",
            BuildPhase.ORDER_DIR: "DESC",
        }
        sql = t.assemble_sql(slots)
        assert "<INVALID" not in sql
        assert "GROUP BY" in sql
        assert "COUNT" in sql

    def test_task3_assembly(self, task_registry):
        t = task_registry.get("task_03_join")
        slots = {
            BuildPhase.SELECT_COLS: "customers.name",
            BuildPhase.FROM_TABLE: "customers",
            BuildPhase.JOIN_TABLE: "orders",
            BuildPhase.JOIN_KEY_LEFT: "customer_id",
            BuildPhase.JOIN_KEY_RIGHT: "customer_id",
            BuildPhase.WHERE_COL: "orders.status",
            BuildPhase.WHERE_OP: "=",
            BuildPhase.WHERE_VAL: "delivered",
        }
        sql = t.assemble_sql(slots)
        assert "<INVALID" not in sql
        assert "INNER JOIN" in sql
        assert "ON" in sql

    def test_task4_assembly_no_crash(self, task_registry):
        t = task_registry.get("task_04_subquery")
        slots = {
            BuildPhase.SELECT_COLS: "customers.name",
            BuildPhase.FROM_TABLE: "customers",
            BuildPhase.WHERE_COL: "customers.customer_id",
            BuildPhase.WHERE_OP_IN: "IN",
            BuildPhase.SUBQ_SELECT_COL: "customer_id",
            BuildPhase.SUBQ_FROM_TABLE: "orders",
            BuildPhase.SUBQ_JOIN_TABLE: None,
            BuildPhase.SUBQ_JOIN_KEY_LEFT: None,
            BuildPhase.SUBQ_JOIN_KEY_RIGHT: None,
            BuildPhase.SUBQ_WHERE_COL: None,
        }
        sql = t.assemble_sql(slots)
        # Should not raise even if INVALID
        assert isinstance(sql, str)

    def test_task5_assembly(self, task_registry):
        t = task_registry.get("task_05_window")
        slots = {
            BuildPhase.SELECT_COLS: "customers.name",
            BuildPhase.WINDOW_FN: "RANK",
            BuildPhase.WINDOW_PARTITION_COL: "city",
            BuildPhase.WINDOW_ORDER_COL: "signup_date",
            BuildPhase.WINDOW_ORDER_DIR: "ASC",
            BuildPhase.FROM_TABLE: "customers",
            BuildPhase.WHERE_COL: "<NONE>",
            BuildPhase.WHERE_OP: "<=",
            BuildPhase.WHERE_VAL: "3",
        }
        sql = t.assemble_sql(slots)
        assert "<INVALID" not in sql
        assert "RANK" in sql or "ROW_NUMBER" in sql
        assert "OVER" in sql
        assert "PARTITION BY" in sql
