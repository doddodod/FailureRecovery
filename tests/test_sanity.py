"""End-to-end sanity tests; these do not represent experimental results."""

from __future__ import annotations

from copy import deepcopy

import pytest
import torch
from torch.optim import Adam

from config import load_config
from evaluation.eval import evaluate_loader
from models import build_model
from training.train import (
    build_loader,
    build_mock_dataset,
    compute_loss,
    load_checkpoint,
    save_checkpoint,
    set_seed,
)


@pytest.mark.parametrize("latent_path", ["predicted", "hidden"])
def test_forward_backward_and_evaluation(latent_path: str) -> None:
    config = load_config("configs/default.yaml")
    config["detector"]["input_path"] = latent_path
    config["detector"]["backbone"] = "lstm"
    set_seed(config["seed"])
    dataset = build_mock_dataset(config, "val")
    loader = build_loader(dataset, config, training=False)
    batch = next(iter(loader))
    model = build_model(config)

    outputs = model(batch["observations"], batch["actions"])
    batch_size = batch["observations"].shape[0]
    sequence_length = config["data"]["detector_sequence_length"]
    latent_dim = config["model"]["latent_dim"]
    assert outputs["hidden_latent"].shape == (batch_size, sequence_length, latent_dim)
    assert outputs["predicted_latent"].shape == (batch_size, sequence_length, latent_dim)
    assert outputs["logits"].shape == (batch_size, sequence_length)

    loss = compute_loss(outputs, batch, config)
    loss.backward()
    assert model.encoder.network[1].weight.grad is not None
    if latent_path == "predicted":
        assert model.predictor.network[0].weight.grad is not None

    results = evaluate_loader(model, loader, config)
    assert len(results["trajectory_predictions"]) == config["data"]["mock_val_trajectories"]
    assert len(results["confusion_matrix"]) == 2
    assert "balanced_accuracy" in results


def test_checkpoint_round_trip(tmp_path) -> None:
    config = load_config("configs/default.yaml")
    set_seed(config["seed"])
    model = build_model(config)
    optimizer = Adam(model.parameters(), lr=1e-3)
    batch = next(iter(build_loader(build_mock_dataset(config, "val"), config, False)))
    optimizer.zero_grad()
    loss = compute_loss(
        model(batch["observations"], batch["actions"]), batch, config
    )
    loss.backward()
    optimizer.step()

    checkpoint_path = tmp_path / "checkpoint.pt"
    save_checkpoint(checkpoint_path, model, optimizer, epoch=1, config=config)
    restored = build_model(deepcopy(config))
    restored_optimizer = Adam(restored.parameters(), lr=1e-3)
    checkpoint = load_checkpoint(
        checkpoint_path, restored, restored_optimizer, map_location="cpu"
    )
    assert checkpoint["epoch"] == 1
    for expected, actual in zip(model.parameters(), restored.parameters()):
        assert torch.equal(expected, actual)
