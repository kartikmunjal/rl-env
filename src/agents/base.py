"""
base.py — Abstract base class for all agents.

All agents interact with SQLQueryEnv via the standard Gymnasium loop:
    obs, info = env.reset()
    while not terminated:
        action = agent.act(obs, info["action_mask"])
        obs, reward, terminated, truncated, info = env.step(action)

The action_mask is ALWAYS passed to act() — agents must respect it.
Selecting a masked action raises AssertionError in the environment.

Author: Kartik Munjal
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class BaseAgent(ABC):
    """
    Abstract agent base class.

    Subclasses must implement:
        act(obs, action_mask) -> int
        reset_episode() -> None  (called at the start of each episode)
    """

    @abstractmethod
    def act(self, obs: dict, action_mask: np.ndarray) -> int:
        """
        Select an action given the current observation and action mask.

        Parameters:
            obs:         dict observation from SQLQueryEnv
            action_mask: float32 array of shape (N_MAX_ACTIONS,)
                         1.0 = valid action, 0.0 = masked (illegal)

        Returns:
            action: int in [0, N_MAX_ACTIONS), must satisfy action_mask[action] == 1.0
        """
        raise NotImplementedError

    def reset_episode(self) -> None:
        """Called at the start of each episode. Override for stateful agents."""
        pass

    def on_episode_end(self, episode_reward: float, info: dict) -> None:
        """Called at the end of each episode. Override for learning agents."""
        pass
