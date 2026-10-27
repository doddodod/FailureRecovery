"""Failure detector D over either predicted or hidden latent sequences."""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn


class FailureDetector(nn.Module):
    def __init__(
        self,
        latent_dim: int,
        backbone: str,
        hidden_dim: int,
        num_layers: int = 1,
        num_heads: int = 2,
        dropout: float = 0.0,
        max_sequence_length: int = 128,
    ) -> None:
        super().__init__()
        self.backbone_name = backbone
        if backbone == "mlp":
            self.backbone = nn.Sequential(
                nn.Linear(latent_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout)
            )
        elif backbone == "lstm":
            self.backbone = nn.LSTM(
                latent_dim,
                hidden_dim,
                num_layers=num_layers,
                dropout=dropout if num_layers > 1 else 0.0,
                batch_first=True,
            )
        elif backbone == "transformer":
            if hidden_dim % num_heads:
                raise ValueError("detector.hidden_dim must be divisible by num_heads")
            self.input_projection = nn.Linear(latent_dim, hidden_dim)
            self.position_embedding = nn.Parameter(
                torch.zeros(1, max_sequence_length, hidden_dim)
            )
            layer = nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=num_heads,
                dim_feedforward=hidden_dim * 4,
                dropout=dropout,
                batch_first=True,
            )
            self.backbone = nn.TransformerEncoder(layer, num_layers=num_layers)
        else:
            raise ValueError(f"Unknown detector backbone: {backbone}")
        self.output = nn.Linear(hidden_dim, 1)

    def forward(self, latent_sequence: Tensor) -> Tensor:
        """Return per-time-step failure logits with shape [batch, sequence]."""
        if self.backbone_name == "lstm":
            features, _ = self.backbone(latent_sequence)
        elif self.backbone_name == "transformer":
            sequence_length = latent_sequence.shape[1]
            if sequence_length > self.position_embedding.shape[1]:
                raise ValueError("sequence exceeds configured max_sequence_length")
            features = self.input_projection(latent_sequence)
            features = features * math.sqrt(features.shape[-1])
            features = features + self.position_embedding[:, :sequence_length]
            features = self.backbone(features)
        else:
            features = self.backbone(latent_sequence)
        return self.output(features).squeeze(-1)
