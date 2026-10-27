"""Compare same-task success/failure rollouts with the repository latent modules.

No checkpoint is loaded and no optimization is performed.  Repeating fixed random
initializations makes explicit how much of the observed trend is seed-dependent.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import torch

from config import load_config
from models import build_model


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_rollout(path: Path) -> tuple[np.ndarray, np.ndarray, dict]:
    with h5py.File(path, "r") as handle:
        observations = np.asarray(handle["proprio"], dtype=np.float32)
        actions = np.asarray(handle["actions"], dtype=np.float32)
        description = handle.attrs["task_description"]
        if isinstance(description, bytes):
            description = description.decode("utf-8")
        metadata = {
            "path": str(path),
            "steps": int(len(observations)),
            "success": bool(handle.attrs["success"]),
            "task_description": str(description),
            "sha256": sha256(path),
        }
    return observations, actions, metadata


def standardize_pair(first: np.ndarray, second: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    joined = np.concatenate([first, second], axis=0)
    mean = joined.mean(axis=0, keepdims=True)
    std = joined.std(axis=0, keepdims=True)
    std[std < 1e-6] = 1.0
    return (first - mean) / std, (second - mean) / std


def make_windows(
    observations: np.ndarray, actions: np.ndarray, context: int, horizon: int
) -> tuple[torch.Tensor, torch.Tensor]:
    last_t = min(len(observations) - 1, len(actions) - horizon)
    timestamps = range(context - 1, last_t + 1)
    obs = np.stack([observations[t - context + 1 : t + 1] for t in timestamps])
    act = np.stack([actions[t : t + horizon] for t in timestamps])
    return torch.from_numpy(obs).float(), torch.from_numpy(act).float()


def resample_progress(values: np.ndarray, bins: int) -> np.ndarray:
    indices = np.linspace(0, len(values) - 1, bins).round().astype(int)
    return values[indices]


def latent_pair(model, obs: torch.Tensor, act: torch.Tensor, bins: int) -> tuple[np.ndarray, np.ndarray]:
    with torch.inference_mode():
        # The detector expects a [batch, sequence, latent] tensor. Here every
        # row is an independently aligned temporal window, so invoke only the
        # two repository modules whose latent geometry is under study.
        hidden_tensor = model.encoder(obs)
        predicted_tensor = model.predictor(hidden_tensor, act)
    hidden = resample_progress(hidden_tensor.cpu().numpy(), bins)
    predicted = resample_progress(predicted_tensor.cpu().numpy(), bins)
    return hidden, predicted


def prefix_metrics(success: np.ndarray, failure: np.ndarray, prefix_steps: list[int]) -> list[dict]:
    rows = []
    for steps in prefix_steps:
        success_prefix = success[:steps]
        failure_prefix = failure[:steps]
        aligned_l2 = np.linalg.norm(success_prefix - failure_prefix, axis=1)
        success_mean = success_prefix.mean(axis=0)
        failure_mean = failure_prefix.mean(axis=0)
        denominator = np.linalg.norm(success_mean) * np.linalg.norm(failure_mean)
        cosine = 1.0 - float(success_mean @ failure_mean / denominator) if denominator > 1e-12 else 0.0
        rows.append({
            "normalized_steps": steps,
            "progress_fraction": steps / len(success),
            "mean_aligned_l2": float(aligned_l2.mean()),
            "last_aligned_l2": float(aligned_l2[-1]),
            "prefix_mean_l2": float(np.linalg.norm(success_mean - failure_mean)),
            "prefix_mean_cosine_distance": cosine,
        })
    return rows


def aggregate(per_seed: list[dict], path: str, prefix_steps: list[int]) -> list[dict]:
    metric_names = ["mean_aligned_l2", "last_aligned_l2", "prefix_mean_l2", "prefix_mean_cosine_distance"]
    rows = []
    for index, steps in enumerate(prefix_steps):
        row = {"normalized_steps": steps, "progress_fraction": per_seed[0][path][index]["progress_fraction"]}
        for metric in metric_names:
            values = np.asarray([seed[path][index][metric] for seed in per_seed])
            row[f"{metric}_mean"] = float(values.mean())
            row[f"{metric}_std"] = float(values.std(ddof=1))
        rows.append(row)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--success", type=Path, required=True)
    parser.add_argument("--failure", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bins", type=int, default=64)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(10)))
    args = parser.parse_args()

    success_obs, success_act, success_meta = load_rollout(args.success)
    failure_obs, failure_act, failure_meta = load_rollout(args.failure)
    if success_meta["task_description"] != failure_meta["task_description"]:
        raise ValueError("Rollouts do not belong to the same task")
    if not success_meta["success"] or failure_meta["success"]:
        raise ValueError("Expected one successful and one failed rollout")

    success_obs, failure_obs = standardize_pair(success_obs, failure_obs)
    success_act, failure_act = standardize_pair(success_act, failure_act)
    config = load_config(args.config)
    config["data"]["observation_dim"] = int(success_obs.shape[1])
    config["data"]["action_dim"] = int(success_act.shape[1])
    config["model"]["dropout"] = 0.0
    config["detector"]["dropout"] = 0.0
    context = int(config["data"]["observation_context_length"])
    horizon = int(config["data"]["action_chunk_length"])
    success_inputs = make_windows(success_obs, success_act, context, horizon)
    failure_inputs = make_windows(failure_obs, failure_act, context, horizon)
    prefix_steps = sorted(set([1, 2, 4, 8, 16, 32, args.bins]))

    per_seed = []
    for seed in args.seeds:
        torch.manual_seed(seed)
        np.random.seed(seed)
        model = build_model(config).eval()
        success_hidden, success_predicted = latent_pair(model, *success_inputs, args.bins)
        failure_hidden, failure_predicted = latent_pair(model, *failure_inputs, args.bins)
        per_seed.append({
            "seed": seed,
            "hidden": prefix_metrics(success_hidden, failure_hidden, prefix_steps),
            "predicted": prefix_metrics(success_predicted, failure_predicted, prefix_steps),
        })

    hidden = aggregate(per_seed, "hidden", prefix_steps)
    predicted = aggregate(per_seed, "predicted", prefix_steps)
    result = {
        "experiment": "same_task_repo_latent_timeline_untrained",
        "status": "exploratory_pair_only",
        "weights": "none; repository modules use fresh PyTorch random initialization for every listed seed",
        "training": "none; no optimizer, loss, checkpoint, or label is used by the model",
        "model_path": "models.build_model -> ObservationEncoder + ActionConditionedPredictor",
        "inputs": "jointly standardized proprio[9] contexts and action[7] chunks",
        "normalization": f"each valid-window latent sequence resampled to {args.bins} normalized progress bins",
        "seeds": args.seeds,
        "config": {
            "observation_context_length": context,
            "action_chunk_length": horizon,
            "latent_dim": int(config["model"]["latent_dim"]),
            "encoder_hidden_dim": int(config["model"]["encoder_hidden_dim"]),
            "predictor_hidden_dim": int(config["model"]["predictor_hidden_dim"]),
        },
        "warning": "An untrained random feature map has no learned failure semantics. This pair tests geometric divergence only and cannot establish generalization.",
        "success_rollout": success_meta,
        "failure_rollout": failure_meta,
        "aggregate": {"hidden": hidden, "predicted": predicted},
        "per_seed": per_seed,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "results.json").write_text(json.dumps(result, indent=2), encoding="utf-8")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    x = np.asarray(prefix_steps)
    for axis, rows, title in zip(axes, [hidden, predicted], ["Observation hidden latent $z^h$", "Action-conditioned latent $z^p$"]) :
        for metric, label, marker in [("mean_aligned_l2", "mean aligned L2", "o"), ("prefix_mean_l2", "prefix-centroid L2", "s")]:
            mean = np.asarray([row[f"{metric}_mean"] for row in rows])
            std = np.asarray([row[f"{metric}_std"] for row in rows])
            axis.plot(x, mean, marker=marker, label=label)
            axis.fill_between(x, mean - std, mean + std, alpha=0.18)
        axis.set(xscale="log", xlabel="Observed normalized progress bins", ylabel="Success-failure latent distance", title=title)
        axis.grid(alpha=0.25)
        axis.legend()
    fig.suptitle("Same LIBERO-Long task: untrained repository latents (mean ± SD over seeds)")
    fig.tight_layout()
    fig.savefig(args.output_dir / "prefix_latent_separation.png", dpi=180)
    print(json.dumps(result["aggregate"], indent=2))


if __name__ == "__main__":
    main()
