"""
random_agent.py — Uniform random agent (lower-bound baseline).

The random agent samples uniformly from the VALID (non-masked) actions at
each step. Since the environment enforces schema-grounded masking, the random
agent always produces syntactically and schema-valid SQL — but with random
semantic content.

Research role:
    Establishes the floor performance across all tasks and reward signals.
    Because the action space is masked, the random agent's expected performance
    on Task 1 (5 phases, ~3 choices/phase) is approximately 1/(3^5) ≈ 0.4%
    execution match, purely from random slot selection.

    This baseline is important: it quantifies how much the RL agent must
    learn beyond random exploration to achieve non-trivial performance.

Author: Kartik Munjal
"""

from __future__ import annotations

import random

import numpy as np

from .base import BaseAgent


class RandomAgent(BaseAgent):
    """
    Selects uniformly at random from valid (unmasked) actions.
    Fully deterministic given a seed — reproducible baselines.
    """

    def __init__(self, seed: int = 42) -> None:
        self._rng = random.Random(seed)

    def act(self, obs: dict, action_mask: np.ndarray) -> int:
        valid_actions = np.where(action_mask == 1.0)[0].tolist()
        if not valid_actions:
            raise RuntimeError("No valid actions available — check action mask.")
        return self._rng.choice(valid_actions)

    def reset_episode(self) -> None:
        pass
