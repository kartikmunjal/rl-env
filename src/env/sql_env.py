"""
sql_env.py — Gymnasium-compatible RL environment for SQL query generation.

This is the central contract of the project. All agents interact exclusively
through this interface. Reward functions, tasks, and the schema are injected
at construction time, making the environment fully configurable.

Gymnasium API compliance:
    - observation_space: Dict(schema, nl_embedding, partial_sql, current_phase)
    - action_space:      Discrete(N_MAX_ACTIONS)
    - reset() -> (obs, info)   where info["action_mask"] is always present
    - step() -> (obs, reward, terminated, truncated, info)

Episode lifecycle:
    1. reset(): sample a task + NL query, initialise SQLState
    2. step(action): fill the current slot, check terminal condition
    3. On terminal: assemble SQL, execute, compute reward, return
    4. info always includes: action_mask, sql_so_far, phase_name, reward_components

See research_notes/design_decisions.md for rationale behind architectural choices.

Author: Kartik Munjal
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Optional

import gymnasium as gym
import numpy as np
import yaml

from .action_space import (
    BuildPhase,
    HierarchicalActionSpace,
    N_MAX_ACTIONS,
    N_PHASES,
    ActionSpec,
)
from .executor import SQLExecutor
from .schema import Schema
from .state import NLEncoder, SQLState
from src.tasks.base import SQLTask, TaskRegistry, load_task_registry  # noqa: E402


class SQLQueryEnv(gym.Env):
    """
    Gymnasium environment for slot-filling SQL query generation.

    Parameters:
        db_path:       Path to the seeded SQLite database
        task_registry: TaskRegistry with all 5 tasks loaded
        reward_fn:     Callable(predicted_sql, gold_sql, db_path, schema) -> float
        nl_vocab_path: Path to the NL vocabulary JSON
        seed:          RNG seed for reproducibility
        task_id:       If set, always uses this task; otherwise samples uniformly
        render_mode:   "human" prints episode progress to stdout

    Observation space:
        schema        (MAX_COLS, 8)     float32
        nl_embedding  (128,)            float32
        partial_sql   (N_PHASES, 32)    float32
        current_phase (1,)              int32

    Action space:
        Discrete(N_MAX_ACTIONS=32)
        The valid subset is communicated via info["action_mask"] at every step.
    """

    metadata = {"render_modes": ["human", "ansi"]}

    def __init__(
        self,
        db_path: str,
        task_registry: TaskRegistry,
        reward_fn,
        nl_vocab_path: str = "configs/nl_vocab.json",
        seed: int = 42,
        task_id: Optional[str] = None,
        render_mode: Optional[str] = None,
        max_episode_steps: int = 15,
    ) -> None:
        super().__init__()

        self.db_path = db_path
        self.task_registry = task_registry
        self.reward_fn = reward_fn
        self.fixed_task_id = task_id
        self.render_mode = render_mode
        self.max_episode_steps = max_episode_steps

        # Load schema (single shared instance for the lifetime of the env)
        self.schema = Schema(db_path)
        self.action_space_builder = HierarchicalActionSpace(self.schema)
        self.executor = SQLExecutor(db_path)

        # NL encoder
        vocab = json.loads(Path(nl_vocab_path).read_text())
        self.nl_encoder = NLEncoder(vocab, dim=128)

        # RNG
        self._rng = random.Random(seed)

        # Gymnasium spaces
        n_tables = len(self.schema.tables)
        schema_feat_dim = n_tables + 4
        self.observation_space = gym.spaces.Dict({
            "schema":        gym.spaces.Box(0.0, 1.0, shape=(Schema.MAX_COLS, schema_feat_dim), dtype=np.float32),
            "nl_embedding":  gym.spaces.Box(0.0, 1.0, shape=(128,), dtype=np.float32),
            "partial_sql":   gym.spaces.Box(0.0, 1.0, shape=(N_PHASES, N_MAX_ACTIONS), dtype=np.float32),
            "current_phase": gym.spaces.Box(0, N_PHASES, shape=(1,), dtype=np.int32),
        })
        self.action_space = gym.spaces.Discrete(N_MAX_ACTIONS)

        # Episode state (initialised by reset())
        self._state: Optional[SQLState] = None
        self._current_task: Optional[SQLTask] = None
        self._current_query_item = None
        self._schema_encoding = self.schema.encode()   # cached; schema never changes
        self._current_action_specs: dict[BuildPhase, ActionSpec] = {}
        self._episode_count = 0

    # ------------------------------------------------------------------
    # Gymnasium API
    # ------------------------------------------------------------------
    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> tuple[dict, dict]:
        if seed is not None:
            self._rng = random.Random(seed)

        # Select task
        if self.fixed_task_id:
            task = self.task_registry.get(self.fixed_task_id)
        else:
            task = self._rng.choice(self.task_registry.all_tasks())
        self._current_task = task

        # Sample NL query
        query_item = task.sample_query(self._rng)
        self._current_query_item = query_item

        # Initialise state
        self._state = SQLState(
            task_id=task.task_id,
            nl_query=query_item.nl,
            active_phases=task.active_phases,
        )

        # Build action specs for the initial phase
        self._current_action_specs = {}
        self._rebuild_action_spec()

        self._episode_count += 1

        obs = self._make_obs()
        info = self._make_info(reward=0.0, reward_components={})
        return obs, info

    def step(self, action: int) -> tuple[dict, float, bool, bool, dict]:
        assert self._state is not None, "Call reset() before step()"
        assert not self._state.is_terminal, "Episode already terminated; call reset()"

        current_phase = self._state.current_phase
        spec = self._current_action_specs.get(current_phase)

        # Validate action against mask (bugs surface as AssertionError)
        assert spec is not None, f"No ActionSpec for phase {current_phase}"
        assert 0 <= action < N_MAX_ACTIONS, f"Action {action} out of range"
        assert spec.mask[action] == 1.0, (
            f"Action {action} is masked (invalid) for phase {current_phase.name}. "
            f"Valid actions: {np.where(spec.mask == 1.0)[0].tolist()}"
        )

        # Decode action to token value
        chosen_value = spec.decode(action)
        self._state.advance(current_phase, chosen_value)

        reward = 0.0
        reward_components: dict[str, float] = {}
        terminated = False
        truncated = False

        if self._state.is_terminal:
            # Assemble SQL and compute reward
            sql = self._current_task.assemble_sql(self._state.partial_slots)
            self._state.assembled_sql = sql
            reward, reward_components = self._compute_reward(sql)
            terminated = True
        elif self._state.episode_step >= self.max_episode_steps:
            # Safety truncation
            truncated = True

        # Rebuild action spec for the new current phase
        if not terminated and not truncated:
            self._rebuild_action_spec()

        obs = self._make_obs()
        info = self._make_info(reward=reward, reward_components=reward_components)

        if self.render_mode == "human":
            self.render()

        return obs, reward, terminated, truncated, info

    def render(self) -> Optional[str]:
        if self._state is None:
            return None
        lines = [
            f"\n=== Episode {self._episode_count} ===",
            f"Task: {self._state.task_id}",
            f"NL:   {self._state.nl_query}",
            f"Phase [{self._state.current_phase_idx}/{len(self._state.active_phases)}]: "
            f"{self._state.current_phase.name}",
        ]
        filled = {p.name: v for p, v in self._state.partial_slots.items() if v is not None}
        if filled:
            lines.append(f"Slots: {filled}")
        if self._state.assembled_sql:
            lines.append(f"SQL:  {self._state.assembled_sql}")
        output = "\n".join(lines)
        if self.render_mode == "human":
            print(output)
        return output

    def close(self) -> None:
        pass

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _rebuild_action_spec(self) -> None:
        """Rebuild the ActionSpec for the current phase."""
        if self._state is None or self._state.is_terminal:
            return
        phase = self._state.current_phase
        spec = self.action_space_builder.get_action_spec(
            phase=phase,
            partial_slots=self._state.partial_slots,
            task_active_phases=self._state.active_phases,
        )
        self._current_action_specs[phase] = spec

    def _compute_reward(self, predicted_sql: str) -> tuple[float, dict]:
        """Delegate to the injected reward function."""
        gold_sql = self._current_query_item.gold_sql
        return self.reward_fn(
            predicted_sql=predicted_sql,
            gold_sql=gold_sql,
            db_path=self.db_path,
            schema=self.schema,
        )

    def _make_obs(self) -> dict:
        if self._state is None:
            raise RuntimeError("State not initialised")
        return self._state.to_observation(
            schema_encoding=self._schema_encoding,
            nl_encoder=self.nl_encoder,
            action_specs=self._current_action_specs,
        )

    def _make_info(self, reward: float, reward_components: dict) -> dict:
        """Build the info dict with action_mask (standard convention)."""
        current_phase = self._state.current_phase if self._state else BuildPhase.DONE
        spec = self._current_action_specs.get(current_phase)
        action_mask = spec.mask.copy() if spec else np.zeros(N_MAX_ACTIONS, dtype=np.float32)

        return {
            "action_mask": action_mask,
            "phase_name": current_phase.name,
            "task_id": self._state.task_id if self._state else "",
            "nl_query": self._state.nl_query if self._state else "",
            "sql_so_far": self._state.assembled_sql or "",
            "reward_components": reward_components,
            "episode_step": self._state.episode_step if self._state else 0,
        }

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------
    @classmethod
    def from_config(
        cls,
        env_config_path: str = "configs/env_config.yaml",
        reward_config_path: str = "configs/reward_config.yaml",
        task_config_path: str = "configs/task_config.yaml",
        project_root: Optional[str] = None,
    ) -> "SQLQueryEnv":
        """
        Convenience factory that loads all configs and returns a ready env.
        """
        root = Path(project_root) if project_root else Path.cwd()
        env_cfg = yaml.safe_load((root / env_config_path).read_text())
        reward_cfg = yaml.safe_load((root / reward_config_path).read_text())

        task_registry = load_task_registry(task_config_path, str(root))

        from src.rewards.composite import CompositeReward
        reward_fn = CompositeReward.from_config(reward_cfg, str(root / env_cfg["db_path"]))

        return cls(
            db_path=str(root / env_cfg["db_path"]),
            task_registry=task_registry,
            reward_fn=reward_fn,
            nl_vocab_path=str(root / env_cfg.get("nl_vocab_path", "configs/nl_vocab.json")),
            seed=env_cfg.get("seed", 42),
            max_episode_steps=env_cfg.get("max_episode_steps", 15),
        )
