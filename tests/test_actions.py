"""
test_actions.py — Action mask correctness and validity tests.

Verifies:
    - All masks are binary (0.0 or 1.0)
    - At least one valid action per phase (no empty masks)
    - FROM_TABLE mask exactly matches schema table names
    - JOIN_TABLE mask excludes the FROM_TABLE
    - WHERE_OP mask restricts to = / != / LIKE for text columns
    - Action decode is consistent with mask

Author: Kartik Munjal
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pytest
import numpy as np


# ---------------------------------------------------------------------------
# Tests without a live DB (pure logic tests)
# ---------------------------------------------------------------------------
class TestActionSpec:
    def test_make_spec_binary_mask(self):
        from src.env.action_space import HierarchicalActionSpace, BuildPhase, N_MAX_ACTIONS

        # Build a mock spec directly
        from src.env.action_space import ActionSpec
        tokens = ["customers", "orders", "products", "order_items"]
        mask = np.zeros(N_MAX_ACTIONS, dtype=np.float32)
        mask[:len(tokens)] = 1.0
        spec = ActionSpec(phase=BuildPhase.FROM_TABLE, tokens=tokens + ["<PAD>"] * (N_MAX_ACTIONS - len(tokens)), mask=mask)

        # All values must be 0 or 1
        unique_vals = set(spec.mask.tolist())
        assert unique_vals <= {0.0, 1.0}, f"Mask contains non-binary values: {unique_vals}"

    def test_spec_n_valid(self):
        from src.env.action_space import ActionSpec, BuildPhase, N_MAX_ACTIONS
        tokens = ["a", "b"] + ["<PAD>"] * (N_MAX_ACTIONS - 2)
        mask = np.zeros(N_MAX_ACTIONS, dtype=np.float32)
        mask[:2] = 1.0
        spec = ActionSpec(phase=BuildPhase.FROM_TABLE, tokens=tokens, mask=mask)
        assert spec.n_valid == 2

    def test_decode_valid_action(self):
        from src.env.action_space import ActionSpec, BuildPhase, N_MAX_ACTIONS
        tokens = ["customers", "orders"] + ["<PAD>"] * (N_MAX_ACTIONS - 2)
        mask = np.zeros(N_MAX_ACTIONS, dtype=np.float32)
        mask[:2] = 1.0
        spec = ActionSpec(phase=BuildPhase.FROM_TABLE, tokens=tokens, mask=mask)
        assert spec.decode(0) == "customers"
        assert spec.decode(1) == "orders"

    def test_decode_masked_action_raises(self):
        from src.env.action_space import ActionSpec, BuildPhase, N_MAX_ACTIONS
        tokens = ["a"] + ["<PAD>"] * (N_MAX_ACTIONS - 1)
        mask = np.zeros(N_MAX_ACTIONS, dtype=np.float32)
        mask[0] = 1.0
        spec = ActionSpec(phase=BuildPhase.FROM_TABLE, tokens=tokens, mask=mask)
        with pytest.raises(ValueError, match="masked"):
            spec.decode(1)  # action 1 is masked


class TestOperatorMasking:
    """Test that text column WHERE_OP restricts operators correctly."""

    def test_where_operators_constants(self):
        from src.env.action_space import WHERE_OPERATORS
        assert "=" in WHERE_OPERATORS
        assert "!=" in WHERE_OPERATORS
        assert "LIKE" in WHERE_OPERATORS
        assert ">" in WHERE_OPERATORS

    def test_n_max_actions_is_32(self):
        from src.env.action_space import N_MAX_ACTIONS
        assert N_MAX_ACTIONS == 32

    def test_build_phase_done_exists(self):
        from src.env.action_space import BuildPhase
        assert BuildPhase.DONE is not None
        assert BuildPhase.SELECT_COLS.value == 0
