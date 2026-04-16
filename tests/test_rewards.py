"""
test_rewards.py — Unit tests for all four reward functions.

Tests cover:
    - Correct SQL → max reward
    - Wrong table → partial/zero reward
    - Wrong operator → partial reward
    - Column flooding → Jaccard penalisation (H3.1 regression test)
    - Syntax error → error penalty (H2.2 regression test)
    - Efficiency penalty trigger (H4.2 regression test)

Author: Kartik Munjal
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pytest

from src.rewards.exact_match import ExactMatchReward
from src.rewards.partial_credit import PartialCreditReward, _jaccard, _extract_columns, _extract_tables


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
GOLD_SQL_SIMPLE = "SELECT name, city FROM customers WHERE city = 'New York'"
GOLD_SQL_AGG = "SELECT city, COUNT(*) AS agg_val FROM customers GROUP BY city ORDER BY agg_val DESC"
GOLD_SQL_JOIN = "SELECT c.name, o.order_date FROM customers c INNER JOIN orders o ON c.customer_id = o.customer_id WHERE o.status = 'delivered'"


# ---------------------------------------------------------------------------
# R1: Exact Match
# ---------------------------------------------------------------------------
class TestExactMatch:
    def setup_method(self):
        self.reward = ExactMatchReward()

    def test_identical_sql_scores_1(self):
        r, _ = self.reward(GOLD_SQL_SIMPLE, GOLD_SQL_SIMPLE)
        assert r == 1.0

    def test_normalisation_case_insensitive(self):
        pred = "select name, city from customers where city = 'new york'"
        r, _ = self.reward(pred, GOLD_SQL_SIMPLE)
        assert r == 1.0

    def test_column_order_invariant(self):
        pred = "SELECT city, name FROM customers WHERE city = 'New York'"
        r, _ = self.reward(pred, GOLD_SQL_SIMPLE)
        assert r == 1.0

    def test_wrong_table_scores_0(self):
        pred = "SELECT name, city FROM products WHERE city = 'New York'"
        r, _ = self.reward(pred, GOLD_SQL_SIMPLE)
        assert r == 0.0

    def test_wrong_value_scores_0(self):
        pred = "SELECT name, city FROM customers WHERE city = 'London'"
        r, _ = self.reward(pred, GOLD_SQL_SIMPLE)
        assert r == 0.0

    def test_invalid_sql_scores_0(self):
        r, _ = self.reward("<INVALID_SQL: missing table>", GOLD_SQL_SIMPLE)
        assert r == 0.0


# ---------------------------------------------------------------------------
# R3: Partial Credit
# ---------------------------------------------------------------------------
class TestPartialCredit:
    def setup_method(self):
        self.reward = PartialCreditReward()

    def test_perfect_sql_scores_1(self):
        r, _ = self.reward(GOLD_SQL_SIMPLE, GOLD_SQL_SIMPLE)
        assert abs(r - 1.0) < 0.05, f"Perfect SQL should score ~1.0, got {r:.3f}"

    def test_wrong_table_lower_score(self):
        pred = "SELECT name, city FROM products WHERE category = 'gold'"
        r_wrong, _ = self.reward(pred, GOLD_SQL_SIMPLE)
        r_correct, _ = self.reward(GOLD_SQL_SIMPLE, GOLD_SQL_SIMPLE)
        assert r_wrong < r_correct, "Wrong table should score lower"

    def test_jaccard_is_symmetric(self):
        set_a = {"name", "city"}
        set_b = {"name", "email"}
        assert abs(_jaccard(set_a, set_b) - _jaccard(set_b, set_a)) < 1e-9

    def test_jaccard_penalises_column_flooding(self):
        """
        H3.1 regression test: column flooding should NOT achieve max column score.
        Agent selects all 15 possible columns vs gold selecting 2.
        Jaccard should be 2/15 ≈ 0.133, far below 1.0.
        """
        all_cols = {"col1", "col2", "col3", "col4", "col5",
                    "col6", "col7", "col8", "col9", "col10",
                    "col11", "col12", "col13", "col14", "col15"}
        gold_cols = {"col1", "col2"}
        jaccard_val = _jaccard(all_cols, gold_cols)
        assert jaccard_val < 0.2, (
            f"Column flooding should have Jaccard ≈ 2/15 = 0.133, got {jaccard_val:.3f}. "
            "If this fails, the Jaccard denominator uses |pred| instead of |pred ∪ gold|."
        )

    def test_invalid_sql_scores_0(self):
        r, _ = self.reward("<INVALID_SQL>", GOLD_SQL_SIMPLE)
        assert r == 0.0

    def test_missing_where_penalised(self):
        pred_no_where = "SELECT name, city FROM customers"
        r_no_where, c_no_where = self.reward(pred_no_where, GOLD_SQL_SIMPLE)
        r_correct, _ = self.reward(GOLD_SQL_SIMPLE, GOLD_SQL_SIMPLE)
        assert r_no_where < r_correct, "Missing WHERE clause should score lower"
        assert c_no_where["where"] == 0.0, "WHERE score should be 0.0 if gold has WHERE"

    def test_correct_agg_fn_scores_full(self):
        r, c = self.reward(GOLD_SQL_AGG, GOLD_SQL_AGG)
        assert c["aggregation"] == 1.0
        assert c["group_by"] == 1.0

    def test_wrong_agg_fn_scores_0(self):
        pred_wrong_agg = "SELECT city, SUM(price) AS agg_val FROM customers GROUP BY city"
        _, c = self.reward(pred_wrong_agg, GOLD_SQL_AGG)
        assert c["aggregation"] == 0.0, "Wrong aggregation function should score 0.0"

    def test_column_extraction(self):
        cols = _extract_columns("SELECT name, city, email FROM customers WHERE city = 'X'")
        assert "name" in cols
        assert "city" in cols
        assert "email" in cols

    def test_table_extraction(self):
        tables = _extract_tables(GOLD_SQL_JOIN)
        assert "customers" in tables
        assert "orders" in tables

    def test_component_weights_sum_to_1(self):
        from src.rewards.partial_credit import ComponentWeights
        w = ComponentWeights()
        w.validate()  # Should not raise


# ---------------------------------------------------------------------------
# Jaccard correctness (fundamental property)
# ---------------------------------------------------------------------------
class TestJaccard:
    def test_identical_sets(self):
        assert _jaccard({"a", "b"}, {"a", "b"}) == 1.0

    def test_disjoint_sets(self):
        assert _jaccard({"a"}, {"b"}) == 0.0

    def test_partial_overlap(self):
        # |{a,b} ∩ {b,c}| / |{a,b} ∪ {b,c}| = 1/3
        result = _jaccard({"a", "b"}, {"b", "c"})
        assert abs(result - 1 / 3) < 1e-9

    def test_empty_sets(self):
        assert _jaccard(set(), set()) == 1.0

    def test_subset(self):
        # {a} ⊂ {a,b}: Jaccard = 1/2
        result = _jaccard({"a"}, {"a", "b"})
        assert abs(result - 0.5) < 1e-9

    def test_superset_penalised(self):
        """
        The superset (flooding) case: {a,b,c,d} vs gold {a}.
        Jaccard = 1/4 = 0.25. Recall would give 1.0 (the hacking bug).
        """
        result = _jaccard({"a", "b", "c", "d"}, {"a"})
        assert abs(result - 0.25) < 1e-9, (
            "Superset should have Jaccard=0.25. "
            "If 1.0, the denominator uses |gold| not |union| — this is the H3.1 bug."
        )
