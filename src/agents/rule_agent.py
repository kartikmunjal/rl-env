"""
rule_agent.py — Keyword-matching rule-based agent (non-learning upper bound).

The rule agent uses a hand-crafted keyword lookup table to deterministically
map NL query tokens to SQL slot selections. It represents the maximum
performance achievable WITHOUT any learning — purely from pattern matching.

Research role:
    The rule agent performs well on Tasks 1-2 (simple SELECT and aggregation)
    where keyword patterns reliably indicate the correct slot values.
    It fails systematically on Tasks 3-5 (join, subquery, window) because:
    - Join key selection requires relational reasoning, not keyword matching
    - Subquery structure requires understanding nested semantics
    - Window functions have no obvious NL keyword mapping

    Documenting these failure modes motivates the need for RL-based agents
    that can learn relational reasoning beyond keyword lookup.

Keyword tables are loaded from agent_config.yaml.

Author: Kartik Munjal
"""

from __future__ import annotations

import re
from typing import Optional

import numpy as np
import yaml
from pathlib import Path

from .base import BaseAgent
from src.env.action_space import BuildPhase


class RuleAgent(BaseAgent):
    """
    Deterministic keyword-matching agent.

    At each phase, it scans the NL query for keyword patterns and selects
    the corresponding action. Falls back to the first valid action if no
    pattern matches.
    """

    def __init__(
        self,
        table_keywords: dict[str, str],
        agg_keywords: dict[str, str],
        op_keywords: dict[str, str],
        order_keywords: dict[str, str],
    ) -> None:
        self.table_keywords = {k.lower(): v for k, v in table_keywords.items()}
        self.agg_keywords = {k.lower(): v for k, v in agg_keywords.items()}
        self.op_keywords = {k.lower(): v for k, v in op_keywords.items()}
        self.order_keywords = {k.lower(): v for k, v in order_keywords.items()}
        self._nl_query: str = ""

    # ------------------------------------------------------------------
    # Phase-specific slot selection
    # ------------------------------------------------------------------
    def act(self, obs: dict, action_mask: np.ndarray) -> int:
        # Extract current phase from obs
        phase_idx = int(obs["current_phase"][0])

        # We need the NL query to do keyword matching.
        # The NL is encoded in nl_embedding (bag-of-words); we store it at reset.
        valid = np.where(action_mask == 1.0)[0]
        if len(valid) == 0:
            raise RuntimeError("No valid actions")

        # For each phase, try to select the best action by keyword matching
        # Phase-specific handlers return the preferred action index or None
        handlers = {
            0:  self._handle_select_cols,
            1:  self._handle_from_table,
            2:  self._handle_join_table,
            8:  self._handle_agg_fn,
            9:  self._handle_group_by_col,
            12: self._handle_order_by_col,
            13: self._handle_order_dir,
            14: self._handle_window_fn,
        }

        handler = handlers.get(phase_idx)
        if handler:
            preferred = handler(valid)
            if preferred is not None:
                return preferred

        # Default: first valid action
        return int(valid[0])

    def _match_keyword(self, keyword_map: dict[str, str], target: str) -> Optional[str]:
        """Return the mapped value for the first keyword found in NL query."""
        nl = self._nl_query.lower()
        for kw, val in keyword_map.items():
            if re.search(r"\b" + re.escape(kw) + r"\b", nl):
                return val
        return None

    def _find_action_for_token(self, valid: np.ndarray, token: str, tokens_list) -> Optional[int]:
        """Find the action index in `valid` whose token matches `token`."""
        token_lower = token.lower()
        for idx in valid:
            if idx < len(tokens_list) and tokens_list[idx].lower() == token_lower:
                return int(idx)
        return None

    def _handle_from_table(self, valid: np.ndarray) -> Optional[int]:
        matched = self._match_keyword(self.table_keywords, self._nl_query)
        if matched and len(valid) > 0:
            # Try to find the matched table in valid actions (we don't have tokens here)
            # Use position heuristic: tables are [customers=0, order_items=1, orders=2, products=3]
            TABLE_ORDER = ["customers", "order_items", "orders", "products"]
            if matched in TABLE_ORDER:
                preferred_idx = TABLE_ORDER.index(matched)
                if preferred_idx in valid:
                    return preferred_idx
        return int(valid[0]) if len(valid) > 0 else None

    def _handle_select_cols(self, valid: np.ndarray) -> Optional[int]:
        # Default: don't select *, prefer first real column (action 1)
        non_star = [a for a in valid if a != 0]
        return int(non_star[0]) if non_star else int(valid[0])

    def _handle_join_table(self, valid: np.ndarray) -> Optional[int]:
        # Match secondary table keyword
        nl = self._nl_query.lower()
        TABLE_ORDER = ["customers", "order_items", "orders", "products"]
        for kw, tbl in self.table_keywords.items():
            if kw in nl and tbl in TABLE_ORDER:
                idx = TABLE_ORDER.index(tbl)
                if idx in valid:
                    return idx
        return int(valid[0]) if len(valid) > 0 else None

    def _handle_agg_fn(self, valid: np.ndarray) -> Optional[int]:
        matched = self._match_keyword(self.agg_keywords, self._nl_query)
        if matched:
            AGG_ORDER = ["COUNT", "SUM", "AVG", "MIN", "MAX"]
            if matched in AGG_ORDER:
                idx = AGG_ORDER.index(matched)
                if idx in valid:
                    return idx
        return None

    def _handle_group_by_col(self, valid: np.ndarray) -> Optional[int]:
        # Heuristic: if we detect a grouping keyword, pick first non-NONE action
        nl = self._nl_query.lower()
        grouping_kws = ["each", "per", "by", "group"]
        if any(k in nl for k in grouping_kws):
            non_none = [a for a in valid if a != 0]
            return int(non_none[0]) if non_none else int(valid[0])
        return int(valid[0])  # <NONE>

    def _handle_order_by_col(self, valid: np.ndarray) -> Optional[int]:
        nl = self._nl_query.lower()
        ordering_kws = ["order", "sort", "rank", "top", "highest", "lowest"]
        if any(k in nl for k in ordering_kws):
            non_none = [a for a in valid if a != 0]
            return int(non_none[0]) if non_none else int(valid[0])
        return int(valid[0])  # <NONE>

    def _handle_order_dir(self, valid: np.ndarray) -> Optional[int]:
        matched = self._match_keyword(self.order_keywords, self._nl_query)
        if matched == "DESC":
            return int(valid[1]) if len(valid) > 1 else int(valid[0])
        return int(valid[0])  # ASC default

    def _handle_window_fn(self, valid: np.ndarray) -> Optional[int]:
        nl = self._nl_query.lower()
        if "row number" in nl or "row_number" in nl:
            return int(valid[2]) if len(valid) > 2 else int(valid[0])
        if "dense" in nl:
            return int(valid[2]) if len(valid) > 2 else int(valid[0])
        return int(valid[0])  # RANK default

    def reset_episode(self) -> None:
        self._nl_query = ""

    def set_nl_query(self, nl_query: str) -> None:
        """Must be called before each episode with the NL query text."""
        self._nl_query = nl_query

    @classmethod
    def from_config(cls, cfg_path: str = "configs/agent_config.yaml") -> "RuleAgent":
        cfg = yaml.safe_load(Path(cfg_path).read_text())
        rule_cfg = cfg.get("rule_agent", {})
        return cls(
            table_keywords=rule_cfg.get("table_keywords", {}),
            agg_keywords=rule_cfg.get("agg_keywords", {}),
            op_keywords=rule_cfg.get("op_keywords", {}),
            order_keywords=rule_cfg.get("order_keywords", {}),
        )
