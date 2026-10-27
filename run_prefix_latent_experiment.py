"""Exploratory same-task LIBERO-Long prefix-latent comparison."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_rollout(path: Path) -> tuple[np.ndarray, dict]:
    with h5py.File(path, "r") as handle:
        proprio = np.asarray(handle["proprio"], dtype=np.float64)
        actions = np.asarray(handle["actions"], dtype=np.float64)
        features = np.concatenate([proprio, actions], axis=1)
        description = handle.attrs["task_description"]
        if isinstance(description, bytes):
            description = description.decode("utf-8")
        metadata = {
            "path": str(path),
            "steps": int(len(features)),
            "success": bool(handle.attrs["success"]),
            "task_description": str(description),
            "sha256": sha256(path),
        }
    return features, metadata


def normalized_prefix(latents: np.ndarray, bins: int) -> np.ndarray:
    indices = np.linspace(0, len(latents) - 1, bins).round().astype(int)
    return latents[indices]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--success", type=Path, required=True)
    parser.add_argument("--failure", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--latent-dim", type=int, default=8)
    parser.add_argument("--bins", type=int, default=64)
    args = parser.parse_args()

    success, success_meta = load_rollout(args.success)
    failure, failure_meta = load_rollout(args.failure)
    if success_meta["task_description"] != failure_meta["task_description"]:
        raise ValueError("Rollouts do not belong to the same task")
    if not success_meta["success"] or failure_meta["success"]:
        raise ValueError("Expected one successful and one failed rollout")

    scaler = StandardScaler().fit(np.concatenate([success, failure]))
    combined = np.concatenate([scaler.transform(success), scaler.transform(failure)])
    pca = PCA(n_components=args.latent_dim, random_state=0).fit(combined)
    success_latent = normalized_prefix(pca.transform(scaler.transform(success)), args.bins)
    failure_latent = normalized_prefix(pca.transform(scaler.transform(failure)), args.bins)

    prefix_steps = [1, 2, 4, 8, 16, 32, args.bins]
    rows = []
    for steps in prefix_steps:
        success_prefix = success_latent[:steps]
        failure_prefix = failure_latent[:steps]
        aligned_l2 = np.linalg.norm(success_prefix - failure_prefix, axis=1)
        success_mean = success_prefix.mean(axis=0)
        failure_mean = failure_prefix.mean(axis=0)
        denominator = np.linalg.norm(success_mean) * np.linalg.norm(failure_mean)
        cosine_distance = 1.0 - float(success_mean @ failure_mean / denominator) if denominator else 0.0
        rows.append(
            {
                "normalized_steps": steps,
                "progress_fraction": steps / args.bins,
                "mean_aligned_l2": float(aligned_l2.mean()),
                "last_aligned_l2": float(aligned_l2[-1]),
                "prefix_mean_l2": float(np.linalg.norm(success_mean - failure_mean)),
                "prefix_mean_cosine_distance": cosine_distance,
            }
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    result = {
        "experiment": "same_task_prefix_latent_separability",
        "status": "exploratory_pair_only",
        "feature_definition": "standardized concat(proprio[9], actions[7])",
        "latent_definition": f"unsupervised PCA({args.latent_dim}) fitted jointly on both rollouts",
        "normalization": f"each rollout resampled to {args.bins} progress bins",
        "warning": "One success/failure pair cannot establish classifier generalization or statistical significance.",
        "success_rollout": success_meta,
        "failure_rollout": failure_meta,
        "pca_explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
        "prefix_metrics": rows,
    }
    (args.output_dir / "results.json").write_text(json.dumps(result, indent=2), encoding="utf-8")

    x = [row["normalized_steps"] for row in rows]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(x, [row["mean_aligned_l2"] for row in rows], marker="o")
    axes[0].set(xlabel="Observed progress bins", ylabel="Mean aligned latent L2", xscale="log", title="Per-step separation")
    axes[1].plot(x, [row["prefix_mean_l2"] for row in rows], marker="o", label="L2")
    axes[1].plot(x, [row["prefix_mean_cosine_distance"] for row in rows], marker="s", label="cosine")
    axes[1].set(xlabel="Observed progress bins", ylabel="Distance", xscale="log", title="Prefix-pooled separation")
    axes[1].legend()
    fig.suptitle("LIBERO-Long task 0: success vs failure latent separation")
    fig.tight_layout()
    fig.savefig(args.output_dir / "prefix_latent_separation.png", dpi=180)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
