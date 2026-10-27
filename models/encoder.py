"""Observation-only hidden latent encoder: z^h_t = E(o_{t-k:t})."""

from __future__ import annotations

import torch
from torch import Tensor, nn


class ObservationEncoder(nn.Module):
    def __init__(
        self,
        observation_dim: int,
        context_length: int,
        hidden_dim: int,
        latent_dim: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Flatten(start_dim=-2),
            nn.Linear(observation_dim * context_length, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, latent_dim),
        )

    def forward(self, observation_context: Tensor) -> Tensor:
        """Accept [..., context_length, observation_dim], return [..., latent_dim]."""
        return self.network(observation_context)
