"""Action-conditioned latent predictor: z^p_t = P(z^h_t, a_{t:t+H})."""

from __future__ import annotations

import torch
from torch import Tensor, nn


class ActionConditionedPredictor(nn.Module):
    def __init__(
        self,
        latent_dim: int,
        action_dim: int,
        action_chunk_length: int,
        hidden_dim: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        input_dim = latent_dim + action_dim * action_chunk_length
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, latent_dim),
        )

    def forward(self, hidden_latent: Tensor, action_chunk: Tensor) -> Tensor:
        """Action conditioning occurs only here, never in ObservationEncoder."""
        flattened_actions = torch.flatten(action_chunk, start_dim=-2)
        return self.network(torch.cat([hidden_latent, flattened_actions], dim=-1))
