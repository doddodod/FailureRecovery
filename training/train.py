"""Minimal train/validation loop for the Foresight skeleton."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Dict, Iterable, Optional

import numpy as np
import torch
from torch import Tensor, nn
from torch.optim import Adam
from torch.utils.data import DataLoader, WeightedRandomSampler

from config import config_parser, load_config
from data import SlidingWindowTrajectoryDataset, make_mock_trajectories
from models import ForesightModel, build_model


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_mock_dataset(config: dict, split: str) -> SlidingWindowTrajectoryDataset:
    data = config["data"]
    count = data[f"mock_{split}_trajectories"]
    trajectories = make_mock_trajectories(
        count=count,
        trajectory_length=data["mock_trajectory_length"],
        observation_dim=data["observation_dim"],
        action_dim=data["action_dim"],
        seed=config["seed"] + (0 if split == "train" else 10_000),
        include_step_labels=data["mock_include_step_labels"],
    )
    return SlidingWindowTrajectoryDataset(
        trajectories,
        observation_context_length=data["observation_context_length"],
        action_chunk_length=data["action_chunk_length"],
        detector_sequence_length=data["detector_sequence_length"],
        stride=data["window_stride"],
    )


def build_loader(
    dataset: SlidingWindowTrajectoryDataset,
    config: dict,
    training: bool,
) -> DataLoader:
    sampler = None
    shuffle = training
    if training and config["training"]["balanced_sampler"]:
        labels = dataset.window_labels()
        counts = torch.bincount(labels, minlength=2).clamp_min(1)
        weights = (1.0 / counts.float())[labels]
        sampler = WeightedRandomSampler(weights, len(weights), replacement=True)
        shuffle = False
    return DataLoader(
        dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=shuffle,
        sampler=sampler,
        num_workers=config["training"]["num_workers"],
    )


def aggregate_logits(logits: Tensor, method: str) -> Tensor:
    if method == "max":
        return logits.max(dim=-1).values
    if method == "mean":
        return logits.mean(dim=-1)
    if method == "last":
        return logits[..., -1]
    raise ValueError(f"Unknown trajectory aggregation: {method}")


def early_detection_weights(labels: Tensor, enabled: bool) -> Tensor:
    # TODO(paper): replace with the paper's exact temporal weighting, if specified.
    return torch.ones_like(labels)


def compute_loss(outputs: Dict[str, Tensor], batch: dict, config: dict) -> Tensor:
    training = config["training"]
    logits = outputs["logits"]
    if training["target_level"] == "step":
        targets = batch["step_labels"].to(logits.device)
    elif training["target_level"] == "trajectory":
        logits = aggregate_logits(logits, training["trajectory_aggregation"])
        targets = batch["trajectory_label"].to(logits.device)
    else:
        raise ValueError("training.target_level must be 'step' or 'trajectory'")

    pos_weight = training["pos_weight"]
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=None
        if pos_weight is None
        else torch.tensor(float(pos_weight), device=logits.device),
        reduction="none",
    )
    losses = criterion(logits, targets)
    weights = early_detection_weights(
        targets, enabled=training["early_detection_weighting"]
    )
    return (losses * weights).mean()


def run_epoch(
    model: ForesightModel,
    loader: Iterable[dict],
    config: dict,
    optimizer: Optional[torch.optim.Optimizer] = None,
) -> float:
    is_training = optimizer is not None
    model.train(is_training)
    total_loss = 0.0
    total_samples = 0
    device = torch.device(config["device"])
    for batch in loader:
        observations = batch["observations"].to(device)
        actions = batch["actions"].to(device)
        if is_training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(is_training):
            outputs = model(observations, actions)
            loss = compute_loss(outputs, batch, config)
            if is_training:
                loss.backward()
                optimizer.step()
        batch_size = observations.shape[0]
        total_loss += float(loss.detach()) * batch_size
        total_samples += batch_size
    return total_loss / max(total_samples, 1)


def save_checkpoint(
    path: str | Path,
    model: ForesightModel,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    config: dict,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "config": config,
        },
        path,
    )


def load_checkpoint(
    path: str | Path,
    model: ForesightModel,
    optimizer: Optional[torch.optim.Optimizer] = None,
    map_location: str | torch.device = "cpu",
) -> dict:
    # Keep compatibility with older PyTorch releases that predate weights_only.
    # Checkpoints are pickle-based; only load files from trusted sources.
    checkpoint = torch.load(path, map_location=map_location)
    model.load_state_dict(checkpoint["model"])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer"])
    return checkpoint


def train(config: dict) -> tuple[ForesightModel, dict]:
    set_seed(config["seed"])
    if config["data"]["source"] != "mock":
        raise NotImplementedError("TODO: connect a real trajectory loader")
    train_loader = build_loader(build_mock_dataset(config, "train"), config, True)
    val_loader = build_loader(build_mock_dataset(config, "val"), config, False)
    model = build_model(config).to(config["device"])
    optimizer = Adam(
        model.parameters(),
        lr=config["training"]["learning_rate"],
        weight_decay=config["training"]["weight_decay"],
    )
    history = {"train_loss": [], "val_loss": []}
    for epoch in range(1, config["training"]["epochs"] + 1):
        train_loss = run_epoch(model, train_loader, config, optimizer)
        val_loss = run_epoch(model, val_loader, config)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        print(f"epoch={epoch} train_loss={train_loss:.4f} val_loss={val_loss:.4f}")
        save_checkpoint(
            config["training"]["checkpoint_path"], model, optimizer, epoch, config
        )
    return model, history


def main() -> None:
    args = config_parser("Train the Foresight skeleton on mock data").parse_args()
    train(load_config(args.config, args.set))


if __name__ == "__main__":
    main()
