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
class DTPPFeatures(AbstractModelFeature):
    ego_agent_past: FeatureDataType
    ego_agent_future: FeatureDataType
    first_stage_ego_trajectory: FeatureDataType
    second_stage_ego_trajectory: FeatureDataType
    neighbor_agents_past: FeatureDataType
    neighbor_agents_future: FeatureDataType

    map_lanes: FeatureDataType
    map_crosswalks: FeatureDataType
    route_lanes: FeatureDataType

    expert_error_in_bounds: FeatureDataType

    def to_feature_tensor(self) -> DTPPFeatures:
        """
        :return object which will be collated into a batch
        """
        return DTPPFeatures(
            ego_agent_future=to_tensor(self.ego_agent_future),
            ego_agent_past=to_tensor(self.ego_agent_past),
            first_stage_ego_trajectory=to_tensor(self.first_stage_ego_trajectory),
            second_stage_ego_trajectory=to_tensor(self.second_stage_ego_trajectory),
            neighbor_agents_past=to_tensor(self.neighbor_agents_past),
            neighbor_agents_future=to_tensor(self.neighbor_agents_future),
            map_lanes=to_tensor(self.map_lanes),
            map_crosswalks=to_tensor(self.map_crosswalks),
            route_lanes=to_tensor(self.route_lanes),
            expert_error_in_bounds=to_tensor(self.expert_error_in_bounds),
        )

    def to_device(self, device: torch.device) -> DTPPFeatures:
        """Implemented. See interface."""
        validate_type(self.ego_agent_past, torch.Tensor)
        validate_type(self.ego_agent_future, torch.Tensor)
        validate_type(self.first_stage_ego_trajectory, torch.Tensor)
        validate_type(self.second_stage_ego_trajectory, torch.Tensor)
        validate_type(self.neighbor_agents_past, torch.Tensor)
        validate_type(self.neighbor_agents_future, torch.Tensor)
        validate_type(self.map_lanes, torch.Tensor)
        validate_type(self.map_crosswalks, torch.Tensor)
        validate_type(self.route_lanes, torch.Tensor)
        validate_type(self.expert_error_in_bounds, torch.Tensor)

        return DTPPFeatures(
            ego_agent_past=self.ego_agent_past.to(device=device),
            ego_agent_future=self.ego_agent_future.to(device=device),
            first_stage_ego_trajectory=self.first_stage_ego_trajectory.to(
                device=device
            ),
            second_stage_ego_trajectory=self.second_stage_ego_trajectory.to(
                device=device
            ),
            neighbor_agents_past=self.neighbor_agents_past.to(device=device),
            neighbor_agents_future=self.neighbor_agents_future.to(device=device),
            map_lanes=self.map_lanes.to(device=device),
            map_crosswalks=self.map_crosswalks.to(device=device),
            route_lanes=self.route_lanes.to(device=device),
            expert_error_in_bounds=self.expert_error_in_bounds.to(device=device),
        )

    @classmethod
    def deserialize(cls, data: Dict[str, Any]) -> DTPPFeatures:
        """
        :return: Return dictionary of data that can be serialized
        """
        return DTPPFeatures(
            ego_agent_past=data["ego_agent_past"],
            ego_agent_future=data["ego_agent_future"],
            first_stage_ego_trajectory=data["first_stage_ego_trajectory"],
            second_stage_ego_trajectory=data["second_stage_ego_trajectory"],
            neighbor_agents_past=data["neighbor_agents_past"],
            neighbor_agents_future=data["neighbor_agents_future"],
            map_lanes=data["map_lanes"],
            map_crosswalks=data["map_crosswalks"],
            route_lanes=data["route_lanes"],
            expert_error_in_bounds=data["expert_error_in_bounds"],
        )

    def unpack(self) -> List[DTPPFeatures]:
        """
        :return: Unpack a batched feature to a list of features.
        """
        return [
            DTPPFeatures(
                ego_agent_past[None],
                ego_agent_future[None],
                first_stage_ego_trajectory[None],
                second_stage_ego_trajectory[None],
                neighbor_agents_past[None],
                neighbor_agents_future[None],
                map_lanes[None],
                map_crosswalks[None],
                route_lanes[None],
                expert_error_in_bounds[None],
            )
            for (
                ego_agent_past,
                ego_agent_future,
                first_stage_ego_trajectory,
                second_stage_ego_trajectory,
                neighbor_agents_past,
                neighbor_agents_future,
                map_lanes,
                map_crosswalks,
                route_lanes,
                expert_error_in_bounds,
            ) in zip(
                self.ego_agent_past,
                self.ego_agent_future,
                self.first_stage_ego_trajectory,
                self.second_stage_ego_trajectory,
                self.neighbor_agents_past,
                self.neighbor_agents_future,
                self.map_lanes,
                self.map_crosswalks,
                self.route_lanes,
                self.expert_error_in_bounds,
            )
        ]
