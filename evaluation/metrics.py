"""Binary failure-detection metrics without external metric dependencies."""

from __future__ import annotations

from typing import Dict

import torch
from torch import Tensor


def binary_metrics(labels: Tensor, predictions: Tensor) -> Dict[str, object]:
    labels = labels.long().cpu()
    predictions = predictions.long().cpu()
    tn = int(((labels == 0) & (predictions == 0)).sum())
    fp = int(((labels == 0) & (predictions == 1)).sum())
    fn = int(((labels == 1) & (predictions == 0)).sum())
    tp = int(((labels == 1) & (predictions == 1)).sum())

    success_accuracy = tn / (tn + fp) if tn + fp else float("nan")
    failure_accuracy = tp / (tp + fn) if tp + fn else float("nan")
    valid = [value for value in (success_accuracy, failure_accuracy) if value == value]
    balanced_accuracy = sum(valid) / len(valid) if valid else float("nan")
    return {
        "confusion_matrix": [[tn, fp], [fn, tp]],
        "per_class_accuracy": {
            "success": success_accuracy,
            "failure": failure_accuracy,
        },
        # Success and failure receive equal weight.
        "balanced_accuracy": balanced_accuracy,
    }


def aggregate_scores(scores: Tensor, method: str) -> Tensor:
    if method == "max":
        return scores.max(dim=-1).values
    if method == "mean":
        return scores.mean(dim=-1)
    if method == "last":
        return scores[..., -1]
    raise ValueError(f"Unknown trajectory aggregation: {method}")
