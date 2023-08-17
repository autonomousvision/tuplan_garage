from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

import torch
from nuplan.planning.script.builders.utils.utils_type import validate_type
from nuplan.planning.training.preprocessing.features.abstract_model_feature import (
    AbstractModelFeature,
    FeatureDataType,
    to_tensor,
)


@dataclass
class PDMFeature(AbstractModelFeature):
    ego_position: FeatureDataType
    ego_velocity: FeatureDataType
    ego_acceleration: FeatureDataType
    planner_centerline: FeatureDataType
    planner_trajectory: FeatureDataType

    def to_feature_tensor(self) -> PDMFeature:
        """
        :return object which will be collated into a batch
        """
        return PDMFeature(
            ego_position=to_tensor(self.ego_position),
            ego_velocity=to_tensor(self.ego_velocity),
            ego_acceleration=to_tensor(self.ego_acceleration),
            planner_centerline=to_tensor(self.planner_centerline),
            planner_trajectory=to_tensor(self.planner_trajectory),
        )

    def to_device(self, device: torch.device) -> PDMFeature:
        """Implemented. See interface."""
        validate_type(self.ego_position, torch.Tensor)
        validate_type(self.ego_velocity, torch.Tensor)
        validate_type(self.ego_acceleration, torch.Tensor)

        validate_type(self.planner_centerline, torch.Tensor)
        validate_type(self.planner_trajectory, torch.Tensor)

        return PDMFeature(
            ego_position=self.ego_position.to(device=device),
            ego_velocity=self.ego_velocity.to(device=device),
            ego_acceleration=self.ego_acceleration.to(device=device),
            planner_centerline=self.planner_centerline.to(device=device),
            planner_trajectory=self.planner_trajectory.to(device=device),
        )

    @classmethod
    def deserialize(cls, data: Dict[str, Any]) -> PDMFeature:
        """
        :return: Return dictionary of data that can be serialized
        """
        return PDMFeature(
            ego_position=data["ego_position"],
            ego_velocity=data["ego_velocity"],
            ego_acceleration=data["ego_acceleration"],
            planner_centerline=data["planner_centerline"],
            planner_trajectory=data["planner_trajectory"],
        )

    def unpack(self) -> List[PDMFeature]:
        """
        :return: Unpack a batched feature to a list of features.
        """
        return [
            PDMFeature(
                ego_position[None],
                ego_velocity[None],
                ego_acceleration[None],
                planner_centerline[None],
                planner_trajectory[None],
            )
            for ego_position, ego_velocity, ego_acceleration, planner_centerline, planner_trajectory in zip(
                self.ego_position,
                self.ego_velocity,
                self.ego_acceleration,
                self.planner_centerline,
                self.planner_trajectory,
            )
        ]

    @property
    def batch_size(self) -> int:
        """
        :return: number of batches
        """
        if len(self.ego_position.shape) == 2:
            return self.ego_position.shape[0]
        else:
            return None

    @classmethod
    def collate(cls, batch: List[PDMFeature]) -> PDMFeature:
        """
        Implemented. See interface.
        Collates a list of features that each have batch size of 1.
        """
        device = batch[0].ego_position.device

        collated_position = torch.stack(
            [item.ego_position for item in batch], dim=0
        ).to(device)

        collated_velocity = torch.stack(
            [item.ego_velocity for item in batch], dim=0
        ).to(device)

        collated_acceleration = torch.stack(
            [item.ego_acceleration for item in batch], dim=0
        ).to(device)

        collated_centerline = torch.stack(
            [item.planner_centerline for item in batch], dim=0
        ).to(device)
        collated_trajectory = torch.stack(
            [item.planner_trajectory for item in batch], dim=0
        ).to(device)

        return PDMFeature(
            ego_position=collated_position,
            ego_velocity=collated_velocity,
            ego_acceleration=collated_acceleration,
            planner_centerline=collated_centerline,
            planner_trajectory=collated_trajectory,
        )
