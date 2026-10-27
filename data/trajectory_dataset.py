"""Unified trajectory format and sliding-window dataset for Foresight."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import torch
from torch import Tensor
from torch.utils.data import Dataset


@dataclass
class Trajectory:
    """One successful (0) or failed (1) manipulation trajectory."""

    observations: Tensor  # [T, observation_dim]
    actions: Tensor  # [T, action_dim]
    trajectory_label: int  # 0=success, 1=failure
    step_label: Optional[Tensor] = None  # [T], optional 0/1 labels

    def validate(self) -> None:
        if self.observations.ndim != 2 or self.actions.ndim != 2:
            raise ValueError("observations and actions must both have shape [T, D]")
        if len(self.observations) != len(self.actions):
            raise ValueError("observations and actions must have equal length")
        if self.trajectory_label not in (0, 1):
            raise ValueError("trajectory_label must be 0 (success) or 1 (failure)")
        if self.step_label is not None and self.step_label.shape != (len(self.actions),):
            raise ValueError("step_label must have shape [T]")


class SlidingWindowTrajectoryDataset(Dataset):
    """Produces fixed detector sequences made of context/action windows.

    For each detector position t:
      observations contains o_{t-k:t}
      actions contains a configurable action block beginning at t
    The model, rather than this dataset, computes z^h and z^p.
    """

    def __init__(
        self,
        trajectories: Sequence[Trajectory],
        observation_context_length: int,
        action_chunk_length: int,
        detector_sequence_length: int,
        stride: int = 1,
    ) -> None:
        self.trajectories = list(trajectories)
        self.context_length = observation_context_length
        self.action_chunk_length = action_chunk_length
        self.sequence_length = detector_sequence_length
        self.stride = stride
        self.indices: List[tuple[int, int]] = []

        if min(observation_context_length, action_chunk_length, detector_sequence_length, stride) < 1:
            raise ValueError("all window lengths and stride must be positive")

        for trajectory_index, trajectory in enumerate(self.trajectories):
            trajectory.validate()
            first_t = self.context_length - 1
            last_t = (
                len(trajectory.actions)
                - self.action_chunk_length
                - self.sequence_length
                + 1
            )
            for start_t in range(first_t, last_t + 1, self.stride):
                self.indices.append((trajectory_index, start_t))

        if not self.indices:
            raise ValueError("No valid windows: trajectories are too short for configured lengths")

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int) -> Dict[str, Tensor]:
        trajectory_index, start_t = self.indices[index]
        trajectory = self.trajectories[trajectory_index]
        observation_windows = []
        action_windows = []
        for t in range(start_t, start_t + self.sequence_length):
            observation_windows.append(
                trajectory.observations[t - self.context_length + 1 : t + 1]
            )
            action_windows.append(
                trajectory.actions[t : t + self.action_chunk_length]
            )

        has_step_labels = trajectory.step_label is not None
        if has_step_labels:
            step_labels = trajectory.step_label[start_t : start_t + self.sequence_length]
        else:
            # A trajectory label is a usable weak target when dense labels are absent.
            step_labels = torch.full(
                (self.sequence_length,), float(trajectory.trajectory_label)
            )

        return {
            "observations": torch.stack(observation_windows).float(),
            "actions": torch.stack(action_windows).float(),
            "trajectory_label": torch.tensor(float(trajectory.trajectory_label)),
            "step_labels": step_labels.float(),
            "has_step_labels": torch.tensor(has_step_labels),
            "trajectory_index": torch.tensor(trajectory_index),
            "start_t": torch.tensor(start_t),
        }

    def window_labels(self) -> Tensor:
        """Labels used by the optional balanced sampler."""
        return torch.tensor(
            [self.trajectories[i].trajectory_label for i, _ in self.indices],
            dtype=torch.long,
        )
