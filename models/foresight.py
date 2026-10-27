"""Composition of the three paper-level modules."""

from __future__ import annotations

from typing import Dict

from torch import Tensor, nn

from .detector import FailureDetector
from .encoder import ObservationEncoder
from .predictor import ActionConditionedPredictor


class ForesightModel(nn.Module):
    def __init__(
        self,
        encoder: ObservationEncoder,
        predictor: ActionConditionedPredictor,
        detector: FailureDetector,
        detector_input_path: str = "predicted",
    ) -> None:
        super().__init__()
        if detector_input_path not in {"predicted", "hidden"}:
            raise ValueError("detector_input_path must be 'predicted' or 'hidden'")
        self.encoder = encoder
        self.predictor = predictor
        self.detector = detector
        self.detector_input_path = detector_input_path

    def forward(self, observations: Tensor, actions: Tensor) -> Dict[str, Tensor]:
        hidden_latent = self.encoder(observations)  # z^h: observation-only
        predicted_latent = self.predictor(hidden_latent, actions)  # z^p: action-conditioned
        detector_latent = (
            predicted_latent
            if self.detector_input_path == "predicted"
            else hidden_latent
        )
        logits = self.detector(detector_latent)
        return {
            "hidden_latent": hidden_latent,
            "predicted_latent": predicted_latent,
            "logits": logits,
            "probabilities": logits.sigmoid(),
        }


def build_model(config: dict) -> ForesightModel:
    data = config["data"]
    model = config["model"]
    detector = config["detector"]
    return ForesightModel(
        ObservationEncoder(
            observation_dim=data["observation_dim"],
            context_length=data["observation_context_length"],
            hidden_dim=model["encoder_hidden_dim"],
            latent_dim=model["latent_dim"],
            dropout=model["dropout"],
        ),
        ActionConditionedPredictor(
            latent_dim=model["latent_dim"],
            action_dim=data["action_dim"],
            action_chunk_length=data["action_chunk_length"],
            hidden_dim=model["predictor_hidden_dim"],
            dropout=model["dropout"],
        ),
        FailureDetector(
            latent_dim=model["latent_dim"],
            backbone=detector["backbone"],
            hidden_dim=detector["hidden_dim"],
            num_layers=detector["num_layers"],
            num_heads=detector["num_heads"],
            dropout=detector["dropout"],
            max_sequence_length=detector["max_sequence_length"],
        ),
        detector_input_path=detector["input_path"],
    )
