"""
reinforce_agent.py — REINFORCE with learned linear baseline.

Architecture:
    PolicyNetwork: Linear(obs_dim, 256) -> ReLU -> Dropout -> Linear(256, 128)
                   -> ReLU -> Linear(128, N_MAX_ACTIONS) -> masked softmax
    BaselineNetwork: Linear(obs_dim, 128) -> ReLU -> Linear(128, 1)

Training:
    For each episode:
        1. Collect trajectory (obs, action, log_prob, mask) at each step
        2. At episode end, receive terminal reward G
        3. Compute advantage: delta = G - V(s) for each step
        4. Policy loss: -log_pi(a|s) * delta (summed over steps)
        5. Baseline loss: (G - V(s))^2 (MSE)
        6. Entropy bonus: -sum(pi * log_pi) (encourages exploration)
        7. Total loss = policy_loss + 0.5 * baseline_loss - entropy_coeff * entropy

Design decisions (see research_notes/design_decisions.md §D3):
    gamma = 1.0: No discounting. All steps within an episode are treated equally.
    Rationale: Reward is only issued at episode termination; discounting would
    create an artificial preference for shorter episodes (Tasks 1-2 over Tasks 4-5).

    REINFORCE over PPO: REINFORCE is the correct choice at this scale.
    The environment has a 590-dim observation, not 175B parameters.
    REINFORCE's failure modes (high variance) are directly observable and
    documentable, which is the research goal.

Observation flattening:
    obs["schema"]        (MAX_COLS, 8)   -> flatten to (MAX_COLS*8,)
    obs["nl_embedding"]  (128,)
    obs["partial_sql"]   (N_PHASES, 32)  -> flatten to (N_PHASES*32,)
    obs["current_phase"] (1,)
    Total input dim: 4*8*20 + 128 + 26*32 + 1 = 640 + 128 + 832 + 1 = ...
    Computed dynamically at first forward pass.

Author: Kartik Munjal
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

from src.env.action_space import N_MAX_ACTIONS
from .base import BaseAgent


# ---------------------------------------------------------------------------
# Networks
# ---------------------------------------------------------------------------
class PolicyNetwork(nn.Module):
    def __init__(self, obs_dim: int, hidden: int = 256, hidden2: int = 128, dropout: float = 0.1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden2),
            nn.ReLU(),
            nn.Linear(hidden2, N_MAX_ACTIONS),
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns (action_probs, log_probs) with invalid actions masked to near-zero.
        Masked softmax: set logit of invalid actions to -1e9 before softmax.
        """
        logits = self.net(x)
        # Mask invalid actions: -1e9 effectively zeroes their probability
        logits = logits + (1.0 - mask) * (-1e9)
        probs = F.softmax(logits, dim=-1)
        log_probs = F.log_softmax(logits, dim=-1)
        return probs, log_probs


class BaselineNetwork(nn.Module):
    def __init__(self, obs_dim: int, hidden: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


# ---------------------------------------------------------------------------
# Trajectory storage
# ---------------------------------------------------------------------------
@dataclass
class Transition:
    obs_flat: torch.Tensor
    action: int
    log_prob: torch.Tensor
    mask: torch.Tensor
    value: torch.Tensor


# ---------------------------------------------------------------------------
# REINFORCE Agent
# ---------------------------------------------------------------------------
class REINFORCEAgent(BaseAgent):
    """
    REINFORCE with learned linear baseline.
    Implements the BaseAgent interface for drop-in use with the training loop.
    """

    def __init__(
        self,
        obs_dim: int,
        hidden_dim: int = 256,
        hidden_dim2: int = 128,
        dropout: float = 0.10,
        learning_rate: float = 3e-4,
        baseline_lr: float = 1e-3,
        gamma: float = 1.0,
        entropy_coeff: float = 0.01,
        grad_clip: float = 0.5,
        device: str = "cpu",
        seed: int = 42,
    ) -> None:
        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)

        self.gamma = gamma
        self.entropy_coeff = entropy_coeff
        self.grad_clip = grad_clip
        self.device = torch.device(device)

        self.policy = PolicyNetwork(obs_dim, hidden_dim, hidden_dim2, dropout).to(self.device)
        self.baseline = BaselineNetwork(obs_dim, hidden_dim2).to(self.device)

        self.policy_optim = torch.optim.Adam(self.policy.parameters(), lr=learning_rate)
        self.baseline_optim = torch.optim.Adam(self.baseline.parameters(), lr=baseline_lr)

        # Episode buffer
        self._trajectory: list[Transition] = []
        self._training = True

        # Metrics
        self.episode_count = 0
        self.total_policy_loss = 0.0
        self.total_baseline_loss = 0.0

    @property
    def obs_dim(self) -> int:
        return self.policy.net[0].in_features

    def train_mode(self) -> None:
        self._training = True
        self.policy.train()
        self.baseline.train()

    def eval_mode(self) -> None:
        self._training = False
        self.policy.eval()
        self.baseline.eval()

    def _flatten_obs(self, obs: dict) -> torch.Tensor:
        """Flatten all observation components into a single 1D tensor."""
        parts = [
            obs["schema"].flatten(),
            obs["nl_embedding"].flatten(),
            obs["partial_sql"].flatten(),
            obs["current_phase"].flatten().astype(np.float32) / 25.0,  # normalise
        ]
        flat = np.concatenate(parts)
        return torch.tensor(flat, dtype=torch.float32, device=self.device)

    def act(self, obs: dict, action_mask: np.ndarray) -> int:
        obs_t = self._flatten_obs(obs).unsqueeze(0)
        mask_t = torch.tensor(action_mask, dtype=torch.float32, device=self.device).unsqueeze(0)

        with torch.set_grad_enabled(self._training):
            probs, log_probs = self.policy(obs_t, mask_t)
            value = self.baseline(obs_t)

            dist = Categorical(probs)
            action_t = dist.sample()
            log_prob = log_probs[0, action_t.item()]

        action = int(action_t.item())

        if self._training:
            self._trajectory.append(Transition(
                obs_flat=obs_t.squeeze(0),
                action=action,
                log_prob=log_prob,
                mask=mask_t.squeeze(0),
                value=value.squeeze(),
            ))

        return action

    def reset_episode(self) -> None:
        self._trajectory = []

    def on_episode_end(self, episode_reward: float, info: dict) -> None:
        if not self._training or not self._trajectory:
            return
        self._update(episode_reward)
        self.episode_count += 1

    def _update(self, G: float) -> None:
        """
        REINFORCE update:
            For all steps t in episode: delta_t = G - V(s_t)
            L_policy = -sum_t [ log pi(a_t | s_t) * delta_t ]
            L_baseline = sum_t [ (G - V(s_t))^2 ]
            L_entropy = -sum_t [ sum_a pi(a|s_t) * log pi(a|s_t) ]
            L_total = L_policy + 0.5 * L_baseline - entropy_coeff * L_entropy

        Design decision: gamma=1.0 (no discounting).
        See research_notes/design_decisions.md §D2 for rationale.
        """
        G_t = torch.tensor(G, dtype=torch.float32, device=self.device)

        policy_losses = []
        baseline_losses = []
        entropies = []

        for transition in self._trajectory:
            delta = G_t - transition.value.detach()
            policy_losses.append(-transition.log_prob * delta)
            baseline_losses.append((G_t - transition.value) ** 2)

            # Entropy for exploration bonus
            obs_t = transition.obs_flat.unsqueeze(0)
            mask_t = transition.mask.unsqueeze(0)
            probs, log_probs = self.policy(obs_t, mask_t)
            entropy = -(probs * log_probs).sum()
            entropies.append(entropy)

        policy_loss = torch.stack(policy_losses).sum()
        baseline_loss = torch.stack(baseline_losses).sum()
        entropy_loss = torch.stack(entropies).sum()

        total_loss = (
            policy_loss
            + 0.5 * baseline_loss
            - self.entropy_coeff * entropy_loss
        )

        self.policy_optim.zero_grad()
        self.baseline_optim.zero_grad()
        total_loss.backward()
        nn.utils.clip_grad_norm_(self.policy.parameters(), self.grad_clip)
        nn.utils.clip_grad_norm_(self.baseline.parameters(), self.grad_clip)
        self.policy_optim.step()
        self.baseline_optim.step()

        self.total_policy_loss += float(policy_loss.item())
        self.total_baseline_loss += float(baseline_loss.item())

    def save(self, path: str) -> None:
        torch.save({
            "policy_state_dict": self.policy.state_dict(),
            "baseline_state_dict": self.baseline.state_dict(),
            "policy_optim_state_dict": self.policy_optim.state_dict(),
            "baseline_optim_state_dict": self.baseline_optim.state_dict(),
            "episode_count": self.episode_count,
        }, path)

    def load(self, path: str) -> None:
        ckpt = torch.load(path, map_location=self.device)
        self.policy.load_state_dict(ckpt["policy_state_dict"])
        self.baseline.load_state_dict(ckpt["baseline_state_dict"])
        self.policy_optim.load_state_dict(ckpt["policy_optim_state_dict"])
        self.baseline_optim.load_state_dict(ckpt["baseline_optim_state_dict"])
        self.episode_count = ckpt.get("episode_count", 0)

    @classmethod
    def from_config(cls, cfg_path: str = "configs/agent_config.yaml") -> "REINFORCEAgent":
        import yaml
        from pathlib import Path
        cfg = yaml.safe_load(Path(cfg_path).read_text())
        rc = cfg.get("reinforce_agent", {})

        # Compute obs_dim dynamically from env specs
        from src.env.action_space import N_PHASES, N_MAX_ACTIONS
        from src.env.schema import Schema
        # MAX_COLS=20, schema_feat=8, nl_dim=128, partial=(N_PHASES*N_MAX_ACTIONS), phase=1
        obs_dim = 20 * 8 + 128 + N_PHASES * N_MAX_ACTIONS + 1

        return cls(
            obs_dim=obs_dim,
            hidden_dim=rc.get("hidden_dim", 256),
            hidden_dim2=rc.get("hidden_dim2", 128),
            dropout=rc.get("dropout", 0.10),
            learning_rate=rc.get("learning_rate", 3e-4),
            baseline_lr=rc.get("baseline_lr", 1e-3),
            gamma=rc.get("gamma", 1.0),
            entropy_coeff=rc.get("entropy_coeff", 0.01),
            grad_clip=rc.get("grad_clip", 0.5),
            device=rc.get("device", "cpu"),
            seed=rc.get("seed", 42),
        )
