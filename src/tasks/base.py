"""
base.py — SQLTask dataclass and TaskRegistry.

A SQLTask represents one of the 5 benchmark tasks. Each task specifies:
  - Which phases are active (an ordered list)
  - A pool of NL queries with their gold SQL
  - Episode length limit
  - Complexity rating (1-5)

The TaskRegistry loads all tasks from task_config.yaml and their NL queries
from the JSON files in configs/tasks/.

Author: Kartik Munjal
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import yaml

from src.env.action_space import BuildPhase


# ---------------------------------------------------------------------------
# Query item
# ---------------------------------------------------------------------------
@dataclass
class QueryItem:
    """One NL question with its ground-truth SQL and slot annotations."""
    nl: str
    gold_sql: str
    metadata: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# SQLTask
# ---------------------------------------------------------------------------
@dataclass
class SQLTask:
    """
    Specifies everything the environment needs to run episodes for one task.

    Attributes:
        task_id:        Unique identifier, matches key in task_config.yaml
        name:           Human-readable name
        description:    Research motivation for including this task
        active_phases:  Ordered list of BuildPhase objects active in this task
        queries:        Pool of QueryItem objects (NL → gold SQL)
        max_steps:      Episode length limit
        difficulty:     Integer 1–5

    Episode protocol:
        1. reset() → sample a QueryItem from self.queries
        2. step() for each phase in active_phases
        3. assemble_sql() → render partial_slots to SQL string
        4. Compute reward against QueryItem.gold_sql
    """

    task_id: str
    name: str
    description: str
    active_phases: list[BuildPhase]
    queries: list[QueryItem]
    max_steps: int
    difficulty: int

    def sample_query(self, rng: random.Random) -> QueryItem:
        """Sample a random NL query from the pool."""
        return rng.choice(self.queries)

    def assemble_sql(self, slots: dict[BuildPhase, Optional[object]]) -> str:
        """
        Render the filled slot dictionary to a SQL string.
        Returns '<INVALID_SQL>' if required slots are missing.

        This is the single source of truth for SQL assembly — all reward
        functions receive the output of this method.
        """
        try:
            return self._assemble(slots)
        except Exception as e:
            return f"<INVALID_SQL: {e}>"

    def _assemble(self, slots: dict) -> str:
        from src.env.action_space import BuildPhase as P

        task_assemblers = {
            "task_01_simple":      self._assemble_task1,
            "task_02_aggregation": self._assemble_task2,
            "task_03_join":        self._assemble_task3,
            "task_04_subquery":    self._assemble_task4,
            "task_05_window":      self._assemble_task5,
        }
        assembler = task_assemblers.get(self.task_id)
        if assembler is None:
            raise ValueError(f"No assembler for task {self.task_id}")
        return assembler(slots)

    def _assemble_task1(self, slots: dict) -> str:
        """SELECT {cols} FROM {table} WHERE {col} {op} {val}"""
        from src.env.action_space import BuildPhase as P

        cols = slots.get(P.SELECT_COLS, "*")
        table = slots.get(P.FROM_TABLE, "customers")
        where_col = slots.get(P.WHERE_COL, "<NONE>")
        where_op = slots.get(P.WHERE_OP, "=")
        where_val = slots.get(P.WHERE_VAL, "")

        if "." in str(cols):
            cols = str(cols).split(".")[-1]

        sql = f"SELECT {cols} FROM {table}"
        if where_col and where_col != "<NONE>":
            col_name = str(where_col).split(".")[-1]
            val = self._format_val(where_val)
            sql += f" WHERE {col_name} {where_op} {val}"
        return sql

    def _assemble_task2(self, slots: dict) -> str:
        """SELECT {group_col}, {agg_fn}({agg_col}) ... GROUP BY ... HAVING ... ORDER BY"""
        from src.env.action_space import BuildPhase as P

        table = slots.get(P.FROM_TABLE, "customers")
        agg_fn = slots.get(P.AGG_FN, "COUNT")
        group_col = slots.get(P.GROUP_BY_COL, "<NONE>")
        having_op = slots.get(P.HAVING_OP, "<NONE>")
        having_val = slots.get(P.HAVING_VAL, "<NONE>")
        order_col = slots.get(P.ORDER_BY_COL, "<NONE>")
        order_dir = slots.get(P.ORDER_DIR, "DESC")

        if group_col and group_col != "<NONE>":
            if "." in str(group_col):
                group_col = str(group_col).split(".")[-1]
            select_col = f"{group_col}"
            agg_col = "*" if agg_fn == "COUNT" else group_col
            sql = (
                f"SELECT {select_col}, {agg_fn}({agg_col}) AS agg_val "
                f"FROM {table} GROUP BY {select_col}"
            )
            if having_op and having_op != "<NONE>" and having_val and having_val != "<NONE>":
                sql += f" HAVING {agg_fn}({agg_col}) {having_op} {having_val}"
            if order_col and order_col != "<NONE>":
                sql += f" ORDER BY agg_val {order_dir}"
        else:
            agg_col = "*" if agg_fn == "COUNT" else "price"
            sql = f"SELECT {agg_fn}({agg_col}) AS agg_val FROM {table}"
        return sql

    def _assemble_task3(self, slots: dict) -> str:
        """SELECT {cols} FROM {from} INNER JOIN {join} ON ... WHERE ..."""
        from src.env.action_space import BuildPhase as P

        from_table = slots.get(P.FROM_TABLE, "customers")
        join_table = slots.get(P.JOIN_TABLE, "orders")
        left_key = slots.get(P.JOIN_KEY_LEFT, "customer_id")
        right_key = slots.get(P.JOIN_KEY_RIGHT, "customer_id")
        where_col = slots.get(P.WHERE_COL, "<NONE>")
        where_op = slots.get(P.WHERE_OP, "=")
        where_val = slots.get(P.WHERE_VAL, "")

        # Derive alias
        fa = from_table[0]
        ja = join_table[0]
        sql = (
            f"SELECT {fa}.*, {ja}.* FROM {from_table} {fa} "
            f"INNER JOIN {join_table} {ja} ON {fa}.{left_key} = {ja}.{right_key}"
        )
        if where_col and where_col != "<NONE>":
            col_qualified = str(where_col)
            if "." not in col_qualified:
                col_qualified = f"{fa}.{col_qualified}"
            val = self._format_val(where_val)
            sql += f" WHERE {col_qualified} {where_op} {val}"
        return sql

    def _assemble_task4(self, slots: dict) -> str:
        """SELECT {cols} FROM {table} WHERE {col} IN (subquery)"""
        from src.env.action_space import BuildPhase as P

        outer_table = slots.get(P.FROM_TABLE, "customers")
        outer_col = slots.get(P.WHERE_COL, "<NONE>")
        subq_from = slots.get(P.SUBQ_FROM_TABLE, "orders")
        subq_col = slots.get(P.SUBQ_SELECT_COL, "customer_id")
        subq_join = slots.get(P.SUBQ_JOIN_TABLE)
        left_key = slots.get(P.SUBQ_JOIN_KEY_LEFT, "order_id")
        right_key = slots.get(P.SUBQ_JOIN_KEY_RIGHT, "order_id")
        subq_where = slots.get(P.SUBQ_WHERE_COL, "<NONE>")

        if outer_col and "." in str(outer_col):
            outer_col = str(outer_col).split(".")[-1]

        inner_sql = f"SELECT DISTINCT {subq_col} FROM {subq_from}"
        if subq_join:
            inner_sql += (
                f" INNER JOIN {subq_join} ON {subq_from}.{left_key} = {subq_join}.{right_key}"
            )
        if subq_where and subq_where != "<NONE>":
            if "." in str(subq_where):
                subq_where = str(subq_where).split(".")[-1]
            inner_sql += f" WHERE {subq_where} IS NOT NULL"

        ot = outer_table
        sql = f"SELECT name FROM {ot}"
        if outer_col and outer_col != "<NONE>":
            sql += f" WHERE {outer_col} IN ({inner_sql})"
        return sql

    def _assemble_task5(self, slots: dict) -> str:
        """SELECT cols, RANK() OVER (PARTITION BY ... ORDER BY ...) FROM ..."""
        from src.env.action_space import BuildPhase as P

        from_table = slots.get(P.FROM_TABLE, "customers")
        window_fn = slots.get(P.WINDOW_FN, "RANK")
        partition_col = slots.get(P.WINDOW_PARTITION_COL, "city")
        order_col = slots.get(P.WINDOW_ORDER_COL, "signup_date")
        order_dir = slots.get(P.WINDOW_ORDER_DIR, "ASC")
        where_col = slots.get(P.WHERE_COL, "<NONE>")
        where_op = slots.get(P.WHERE_OP, "<=")
        where_val = slots.get(P.WHERE_VAL, "3")

        sql = (
            f"SELECT *, {window_fn}() OVER "
            f"(PARTITION BY {partition_col} ORDER BY {order_col} {order_dir}) AS rnk "
            f"FROM {from_table}"
        )
        if where_col and where_col != "<NONE>":
            # Wrap in subquery for outer filter on rank
            sql = f"SELECT * FROM ({sql}) t WHERE rnk {where_op} {where_val}"
        return sql

    @staticmethod
    def _format_val(val) -> str:
        """Quote strings; leave numbers as-is."""
        if val is None or val == "":
            return "''"
        try:
            float(val)
            return str(val)
        except (ValueError, TypeError):
            v = str(val).strip("'\"")
            return f"'{v}'"


# ---------------------------------------------------------------------------
# TaskRegistry
# ---------------------------------------------------------------------------
class TaskRegistry:
    """
    Loads and holds all SQLTask objects.
    Provides O(1) lookup by task_id.
    """

    def __init__(self, tasks: list[SQLTask]) -> None:
        self._tasks = {t.task_id: t for t in tasks}

    def get(self, task_id: str) -> SQLTask:
        if task_id not in self._tasks:
            raise KeyError(f"Unknown task: {task_id}")
        return self._tasks[task_id]

    def all_tasks(self) -> list[SQLTask]:
        return list(self._tasks.values())

    def task_ids(self) -> list[str]:
        return list(self._tasks.keys())

    def __len__(self) -> int:
        return len(self._tasks)


def load_task_registry(
    task_config_path: str = "configs/task_config.yaml",
    project_root: Optional[str] = None,
) -> TaskRegistry:
    """
    Parse task_config.yaml and load all NL query JSON files.
    Returns a populated TaskRegistry.
    """
    root = Path(project_root) if project_root else Path.cwd()
    cfg = yaml.safe_load((root / task_config_path).read_text())

    phase_map = {p.name: p for p in BuildPhase}
    tasks = []
    for task_cfg in cfg["tasks"]:
        active_phases = [phase_map[p] for p in task_cfg["active_phases"]]
        queries_path = root / task_cfg["nl_queries_file"]
        queries_raw = json.loads(queries_path.read_text())
        queries = [
            QueryItem(
                nl=q["nl"],
                gold_sql=q["gold_sql"],
                metadata={k: v for k, v in q.items() if k not in ("nl", "gold_sql")},
            )
            for q in queries_raw
        ]
        task = SQLTask(
            task_id=task_cfg["id"],
            name=task_cfg["name"],
            description=task_cfg.get("description", ""),
            active_phases=active_phases,
            queries=queries,
            max_steps=task_cfg["max_episode_steps"],
            difficulty=task_cfg["difficulty"],
        )
        tasks.append(task)

    return TaskRegistry(tasks)
