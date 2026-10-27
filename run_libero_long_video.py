"""Run one public LIBERO-Long rollout through the Foresight skeleton.

This is deliberately a smoke adapter: the public project page provides a video
but no policy action chunks or released Foresight checkpoint. Frame statistics
form observations and temporal frame differences form an explicitly labelled
action proxy so the real rollout can exercise the complete tensor path.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from torch import nn

from config import load_config
from data import SlidingWindowTrajectoryDataset, Trajectory
from models import build_model


def read_video_features(path: Path, max_frames: int) -> tuple[torch.Tensor, torch.Tensor, dict]:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {path}")

    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    stride = max(frame_count // max_frames, 1)
    observations = []
    action_proxies = []
    previous_mean = None
    frame_index = 0

    while len(observations) < max_frames:
        ok, frame = capture.read()
        if not ok:
            break
        if frame_index % stride:
            frame_index += 1
            continue

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        small = cv2.resize(rgb, (32, 32), interpolation=cv2.INTER_AREA)
        channel_mean = small.mean(axis=(0, 1))
        channel_std = small.std(axis=(0, 1))
        gray = small.mean(axis=2)
        quadrant_mean = np.array(
            [
                gray[:16, :16].mean(),
                gray[:16, 16:].mean(),
                gray[16:, :16].mean(),
                gray[16:, 16:].mean(),
            ],
            dtype=np.float32,
        )
        gradients = np.gradient(gray)
        gradient_stats = np.array(
            [np.abs(gradients[0]).mean(), np.abs(gradients[1]).mean()],
            dtype=np.float32,
        )
        observations.append(
            np.concatenate([channel_mean, channel_std, quadrant_mean, gradient_stats])
        )

        delta = np.zeros(3, dtype=np.float32) if previous_mean is None else channel_mean - previous_mean
        action_proxies.append(np.concatenate([delta, [np.abs(delta).mean()]]))
        previous_mean = channel_mean
        frame_index += 1

    capture.release()
    if len(observations) < 10:
        raise RuntimeError(f"Video yielded only {len(observations)} sampled frames")

    metadata = {
        "source_frames": frame_count,
        "sampled_frames": len(observations),
        "fps": fps,
        "resolution": [width, height],
        "sampling_stride": stride,
    }
    return (
        torch.from_numpy(np.stack(observations)),
        torch.from_numpy(np.stack(action_proxies)),
        metadata,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-frames", type=int, default=64)
    parser.add_argument("--task-index", type=int, required=True)
    parser.add_argument("--episode-index", type=int, required=True)
    parser.add_argument("--trajectory-label", type=int, choices=[0, 1], required=True)
    args = parser.parse_args()

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")

    config = load_config("configs/default.yaml", [])
    config["device"] = args.device
    config["data"]["observation_dim"] = 12
    config["data"]["action_dim"] = 4

    observations, actions, video_metadata = read_video_features(args.video, args.max_frames)
    trajectory = Trajectory(
        observations, actions, trajectory_label=args.trajectory_label
    )
    dataset = SlidingWindowTrajectoryDataset(
        [trajectory],
        observation_context_length=config["data"]["observation_context_length"],
        action_chunk_length=config["data"]["action_chunk_length"],
        detector_sequence_length=config["data"]["detector_sequence_length"],
    )
    sample = dataset[0]
    batch_observations = sample["observations"].unsqueeze(0).to(args.device)
    batch_actions = sample["actions"].unsqueeze(0).to(args.device)

    torch.manual_seed(config["seed"])
    model = build_model(config).to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    if args.device.startswith("cuda"):
        torch.cuda.synchronize()
    start = time.perf_counter()
    outputs = model(batch_observations, batch_actions)
    targets = torch.full_like(outputs["logits"], float(args.trajectory_label))
    loss = nn.BCEWithLogitsLoss()(outputs["logits"], targets)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    if args.device.startswith("cuda"):
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    result = {
        "status": "ok",
        "run_kind": "smoke_test_not_paper_reproduction",
        "sample": {
            "benchmark": "LIBERO-Long",
            "task_index": args.task_index,
            "episode_index": args.episode_index,
            "trajectory_label": args.trajectory_label,
            "label": "failure / true positive" if args.trajectory_label else "success / true negative",
            "video": str(args.video),
            **video_metadata,
        },
        "input_note": "real video observations; frame-difference action proxy (policy actions unavailable)",
        "model_note": "randomly initialized skeleton; scores are not scientific metrics",
        "device": str(args.device),
        "cuda_device": torch.cuda.get_device_name(0) if args.device.startswith("cuda") else None,
        "window_count": len(dataset),
        "observation_tensor_shape": list(batch_observations.shape),
        "action_tensor_shape": list(batch_actions.shape),
        "logits_shape": list(outputs["logits"].shape),
        "failure_probabilities_before_update": outputs["probabilities"].detach().cpu().flatten().tolist(),
        "one_step_bce_loss": float(loss.detach().cpu()),
        "forward_backward_seconds": elapsed,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
