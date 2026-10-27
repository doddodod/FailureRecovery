"""Evaluation flow with temporal scores and trajectory-level metrics."""

from __future__ import annotations

import json
from collections import defaultdict
from typing import Dict, Iterable

import torch

from config import config_parser, load_config
from evaluation.metrics import binary_metrics
from models import ForesightModel, build_model
from training.train import build_loader, build_mock_dataset, load_checkpoint, set_seed


@torch.no_grad()
def evaluate_loader(
    model: ForesightModel,
    loader: Iterable[dict],
    config: dict,
) -> Dict[str, object]:
    model.eval()
    device = torch.device(config["device"])
    # Overlapping dataset windows may score the same absolute step. Average those
    # duplicate scores before trajectory aggregation.
    score_buckets: dict[int, dict[int, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    labels: dict[int, int] = {}

    for batch in loader:
        outputs = model(
            batch["observations"].to(device),
            batch["actions"].to(device),
        )
        probabilities = outputs["probabilities"].cpu()
        for row in range(probabilities.shape[0]):
            trajectory_index = int(batch["trajectory_index"][row])
            start_t = int(batch["start_t"][row])
            labels[trajectory_index] = int(batch["trajectory_label"][row])
            for offset, score in enumerate(probabilities[row].tolist()):
                score_buckets[trajectory_index][start_t + offset].append(score)

    threshold = float(config["evaluation"]["threshold"])
    aggregation = config["evaluation"]["trajectory_aggregation"]
    temporal_scores = {}
    trajectory_scores = {}
    for trajectory_index, step_buckets in sorted(score_buckets.items()):
        merged = [
            (step, sum(values) / len(values))
            for step, values in sorted(step_buckets.items())
        ]
        temporal_scores[str(trajectory_index)] = [
            {"t": step, "failure_probability": score} for step, score in merged
        ]
        values = [score for _, score in merged]
        if aggregation == "max":
            trajectory_score = max(values)
        elif aggregation == "mean":
            trajectory_score = sum(values) / len(values)
        elif aggregation == "last":
            trajectory_score = values[-1]
        else:
            raise ValueError(f"Unknown trajectory aggregation: {aggregation}")
        trajectory_scores[str(trajectory_index)] = trajectory_score

    ordered_indices = sorted(labels)
    label_tensor = torch.tensor([labels[index] for index in ordered_indices])
    prediction_tensor = torch.tensor(
        [
            trajectory_scores[str(index)] >= threshold
            for index in ordered_indices
        ]
    )
    metrics = binary_metrics(label_tensor, prediction_tensor)
    return {
        "temporal_scores": temporal_scores,
        "trajectory_scores": trajectory_scores,
        "trajectory_predictions": {
            str(index): int(trajectory_scores[str(index)] >= threshold)
            for index in ordered_indices
        },
        **metrics,
    }


def evaluate(config: dict, checkpoint_path: str | None = None) -> Dict[str, object]:
    set_seed(config["seed"])
    dataset = build_mock_dataset(config, "val")
    loader = build_loader(dataset, config, training=False)
    model = build_model(config).to(config["device"])
    load_checkpoint(
        checkpoint_path or config["training"]["checkpoint_path"],
        model,
        map_location=config["device"],
    )
    return evaluate_loader(model, loader, config)


def main() -> None:
    parser = config_parser("Evaluate a Foresight skeleton checkpoint")
    parser.add_argument("--checkpoint", default=None)
    args = parser.parse_args()
    results = evaluate(load_config(args.config, args.set), args.checkpoint)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
