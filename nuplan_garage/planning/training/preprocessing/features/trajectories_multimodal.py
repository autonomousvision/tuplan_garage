from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

import torch
from nuplan.planning.training.preprocessing.features.abstract_model_feature import (
    AbstractModelFeature,
    FeatureDataType,
    to_tensor,
)
from torch.utils.data.dataloader import default_collate


@dataclass
class MultiModalTrajectories(AbstractModelFeature):
    """
    A feature that contains multiple trajectories
    :param trajectories: either a [num_batches, num_trajectories, num_states, 3] or [num_trajectories, num_states, 3] representing the trajectory
        where se2_state is [x, y, heading] with units [meters, meters, radians].
    :param probabilities: either a [num_batches, num_trajectories] or [num_trajectories] representing the probability for the respective trajectory
    """

    trajectories: FeatureDataType
    probabilities: FeatureDataType

    def __post_init__(self):
        if self.trajectories.ndim != 4 and self.trajectories.ndim != 3:
            raise RuntimeError(
                f"Invalid trajectory array. Expected 3 or 4 dims, got {self.trajectories.ndim}."
            )
        if self.probabilities.ndim != self.trajectories.ndim - 2:
            raise RuntimeError(
                f"Invalid probabilities array. Expected {self.trajectories.ndim-2} dims for {self.trajectories.ndim} dims in trajectory, got {self.probabilities.ndim}."
            )

        if self.probabilities.shape[-1] != self.trajectories.shape[-3]:
            raise RuntimeError(
                "Number of trajectories must match number of probabilities"
            )

    def to_feature_tensor(self) -> MultiModalTrajectories:
        """Implemented. See interface."""
        return MultiModalTrajectories(
            trajectories=to_tensor(self.trajectories),
            probabilities=to_tensor(self.probabilities),
        )

    def to_device(self, device: torch.device) -> MultiModalTrajectories:
        """Implemented. See interface."""
        return MultiModalTrajectories(
            trajectories=self.trajectories.to(device=device),
            probabilities=self.probabilities.to(device=device),
        )

    @classmethod
    def deserialize(cls, data: Dict[str, Any]) -> MultiModalTrajectories:
        """Implemented. See interface."""
        return MultiModalTrajectories(
            trajectories=data["trajectories"],
            probabilities=data["probabilities"],
        )

    @property
    def most_likely_trajectory(self) -> FeatureDataType:
        """
        :return: [num_batches, num_states, 3] trajectory with the highest probability
        """
        trajectories = to_tensor(self.trajectories)
        probabilities = to_tensor(self.probabilities)
        if trajectories.ndim == 3:
            trajectories = trajectories[None]
            probabilities = probabilities[None]

        return trajectories.take_along_dim(
            indices=probabilities.argmax(dim=1, keepdim=True)[..., None, None],
            dim=1,
        ).squeeze(dim=1)

    def unpack(self) -> List[MultiModalTrajectories]:
        """Implemented. See interface."""
        return [
            MultiModalTrajectories(
                trajectories=trajectories[None],
                probabilities=probabilities[None],
            )
            for trajectories, probabilities in zip(
                self.trajectories, self.probabilities
            )
        ]

    @classmethod
    def collate(cls, batch: List[MultiModalTrajectories]) -> MultiModalTrajectories:
        """
        Implemented. See interface.
        Collates a list of features that each have batch size of 1.
        """
        return MultiModalTrajectories(
            trajectories=default_collate([item.trajectories for item in batch]),
            probabilities=default_collate([item.probabilities for item in batch]),
        )
