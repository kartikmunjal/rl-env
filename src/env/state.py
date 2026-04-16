"""
state.py — SQLState dataclass and observation encoding.

The SQLState encodes the full Markov state of an episode at any given timestep.
It is designed so that to_observation() is deterministic and pure (no side effects).

Observation space (gymnasium.spaces.Dict):
    schema:          float32 (MAX_COLS, 8)     — schema column feature matrix
    nl_embedding:    float32 (128,)            — bag-of-words NL query encoding
    partial_sql:     float32 (N_PHASES, 32)    — one-hot slot fillings so far
    current_phase:   int32   (1,)              — phase index

Design decisions:
    - partial_sql uses one-hot encoding per phase, not raw token indices.
      This gives the agent a richer gradient signal than a raw integer.
    - nl_embedding is a TF-IDF bag-of-words vector, not a learned embedding.
      This is deliberate: we want the reward hacking analysis to not conflate
      representation learning with reward signal issues.
    - All components are fixed-size to support a static Gymnasium observation space.

Author: Kartik Munjal
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np

from .action_space import BuildPhase, N_MAX_ACTIONS, N_PHASES


# ---------------------------------------------------------------------------
# NL Encoder (bag-of-words)
# ---------------------------------------------------------------------------
class NLEncoder:
    """
    Encodes natural language queries as bag-of-words vectors.
    Uses a pre-built vocabulary (loaded from configs/nl_vocab.json).
    """

    def __init__(self, vocab: dict[str, int], dim: int = 128) -> None:
        self.vocab = vocab
        self.dim = dim
        self._unk_idx = vocab.get("<UNK>", 1)

    def encode(self, nl_query: str) -> np.ndarray:
        vec = np.zeros(self.dim, dtype=np.float32)
        tokens = nl_query.lower().split()
        for tok in tokens:
            tok = tok.strip(".,?!;:")
            idx = self.vocab.get(tok, self._unk_idx)
            if 0 <= idx < self.dim:
                vec[idx] += 1.0
        # L1 normalise
        total = vec.sum()
        if total > 0:
            vec /= total
        return vec

    @classmethod
    def from_file(cls, vocab_path: str, dim: int = 128) -> "NLEncoder":
        import json
        from pathlib import Path
        vocab = json.loads(Path(vocab_path).read_text())
        return cls(vocab, dim)


# ---------------------------------------------------------------------------
# SQLState
# ---------------------------------------------------------------------------
@dataclass
class SQLState:
    """
    Full Markov state of a SQL generation episode.

    Fields:
        task_id         — identifier string from task_config.yaml
        nl_query        — natural language question to answer
        active_phases   — ordered list of BuildPhase for this task
        partial_slots   — BuildPhase → chosen value (str/int/list) or None
        current_phase_idx — index into active_phases (not the raw BuildPhase int)
        episode_step    — number of steps taken so far
        is_terminal     — True when all active phases are filled

    NOT stored: schema encoding and NL embedding. These are computed lazily
    in to_observation() to keep the state itself lightweight.
    """

    task_id: str
    nl_query: str
    active_phases: list[BuildPhase]
    partial_slots: dict[BuildPhase, Optional[Any]] = field(default_factory=dict)
    current_phase_idx: int = 0
    episode_step: int = 0
    is_terminal: bool = False

    # Held after assembly at terminal step
    assembled_sql: Optional[str] = None
    reward_components: dict = field(default_factory=dict)

    @property
    def current_phase(self) -> BuildPhase:
        if self.current_phase_idx >= len(self.active_phases):
            return BuildPhase.DONE
        return self.active_phases[self.current_phase_idx]

    def advance(self, phase: BuildPhase, value: Any) -> None:
        """Fill the current slot and advance to the next phase."""
        self.partial_slots[phase] = value
        self.current_phase_idx += 1
        self.episode_step += 1
        if self.current_phase_idx >= len(self.active_phases):
            self.is_terminal = True

    def to_observation(
        self,
        schema_encoding: np.ndarray,
        nl_encoder: NLEncoder,
        action_specs: dict,
    ) -> dict[str, np.ndarray]:
        """
        Encode the state into a gymnasium-compatible observation dict.

        Parameters:
            schema_encoding: (MAX_COLS, 8) float32 from Schema.encode()
            nl_encoder:      NLEncoder for this environment instance
            action_specs:    {BuildPhase: ActionSpec} for encoding partial_sql

        Returns dict with keys: schema, nl_embedding, partial_sql, current_phase
        """
        nl_emb = nl_encoder.encode(self.nl_query)

        # partial_sql: for each phase, one-hot over the N_MAX_ACTIONS slots
        partial_sql = np.zeros((N_PHASES, N_MAX_ACTIONS), dtype=np.float32)
        for phase, value in self.partial_slots.items():
            if value is None:
                continue
            spec = action_specs.get(phase)
            if spec is None:
                continue
            # Find index of the chosen value in the spec tokens
            try:
                idx = spec.tokens.index(str(value))
                if idx < N_MAX_ACTIONS:
                    partial_sql[int(phase), idx] = 1.0
            except ValueError:
                pass

        return {
            "schema": schema_encoding,
            "nl_embedding": nl_emb,
            "partial_sql": partial_sql,
            "current_phase": np.array([self.current_phase_idx], dtype=np.int32),
        }

    def __repr__(self) -> str:
        filled = {p.name: v for p, v in self.partial_slots.items() if v is not None}
        return (
            f"SQLState(task={self.task_id}, phase={self.current_phase.name}, "
            f"step={self.episode_step}, slots={filled})"
        )
