"""Small synthetic trajectories used only for plumbing/sanity checks."""

from __future__ import annotations

from typing import List

import torch

from .trajectory_dataset import Trajectory


def make_mock_trajectories(
    count: int,
    trajectory_length: int,
    observation_dim: int,
    action_dim: int,
    seed: int,
    include_step_labels: bool = True,
) -> List[Trajectory]:
    generator = torch.Generator().manual_seed(seed)
    trajectories = []
    for index in range(count):
        label = index % 2  # Guarantee both classes when count >= 2.
        observations = torch.randn(
            trajectory_length, observation_dim, generator=generator
        )
        actions = torch.randn(trajectory_length, action_dim, generator=generator)

        # A tiny synthetic shift makes metrics deterministic but is not an experiment.
        if label:
            observations[trajectory_length // 2 :] += 0.25

        step_label = None
        if include_step_labels:
            step_label = torch.zeros(trajectory_length)
            if label:
                step_label[trajectory_length // 2 :] = 1.0

        trajectories.append(
            Trajectory(
                observations=observations,
                actions=actions,
                trajectory_label=label,
                step_label=step_label,
            )
        )
    return trajectories
