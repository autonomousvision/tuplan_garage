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
class GameFormerFeatures(AbstractModelFeature):
    ego_agent_past: FeatureDataType
    ego_agent_future: FeatureDataType
    neighbor_agents_past: FeatureDataType
    neighbor_agents_future: FeatureDataType
    lanes: FeatureDataType
    crosswalks: FeatureDataType
    route_lanes: FeatureDataType

    def to_feature_tensor(self) -> GameFormerFeatures:
        """
        :return object which will be collated into a batch
        """
        return GameFormerFeatures(
            ego_agent_past=to_tensor(self.ego_agent_past),
            ego_agent_future=to_tensor(self.ego_agent_future),
            neighbor_agents_past=to_tensor(self.neighbor_agents_past),
            neighbor_agents_future=to_tensor(self.neighbor_agents_future),
            lanes=to_tensor(self.lanes),
            crosswalks=to_tensor(self.crosswalks),
            route_lanes=to_tensor(self.route_lanes),
        )

    def to_device(self, device: torch.device) -> GameFormerFeatures:
        """Implemented. See interface."""
        validate_type(self.ego_agent_past, torch.Tensor)
        validate_type(self.ego_agent_future, torch.Tensor)
        validate_type(self.neighbor_agents_past, torch.Tensor)
        validate_type(self.neighbor_agents_future, torch.Tensor)
        validate_type(self.lanes, torch.Tensor)
        validate_type(self.crosswalks, torch.Tensor)
        validate_type(self.route_lanes, torch.Tensor)

        return GameFormerFeatures(
            ego_agent_past=self.ego_agent_past.to(device=device),
            ego_agent_future=self.ego_agent_future.to(device=device),
            neighbor_agents_past=self.neighbor_agents_past.to(device=device),
            neighbor_agents_future=self.neighbor_agents_future.to(device=device),
            lanes=self.lanes.to(device=device),
            crosswalks=self.crosswalks.to(device=device),
            route_lanes=self.route_lanes.to(device=device),
        )

    @classmethod
    def deserialize(cls, data: Dict[str, Any]) -> GameFormerFeatures:
        """
        :return: Return dictionary of data that can be serialized
        """
        return GameFormerFeatures(
            ego_agent_past=data["ego_agent_past"],
            ego_agent_future=data["ego_agent_future"],
            neighbor_agents_past=data["neighbor_agents_past"],
            neighbor_agents_future=data["neighbor_agents_future"],
            lanes=data["lanes"],
            crosswalks=data["crosswalks"],
            route_lanes=data["route_lanes"],
        )

    def unpack(self) -> List[GameFormerFeatures]:
        """
        :return: Unpack a batched feature to a list of features.
        """
        return [
            GameFormerFeatures(
                ego_agent_past[None],
                ego_agent_future[None],
                neighbor_agents_past[None],
                neighbor_agents_future[None],
                lanes[None],
                crosswalks[None],
                route_lanes[None],
            )
            for (
                map_name,
                token,
                ego_agent_past,
                ego_agent_future,
                neighbor_agents_past,
                neighbor_agents_future,
                lanes,
                crosswalks,
                route_lanes,
            ) in zip(
                self.ego_agent_past,
                self.ego_agent_future,
                self.neighbor_agents_past,
                self.neighbor_agents_future,
                self.lanes,
                self.crosswalks,
                self.route_lanes,
            )
        ]

    @classmethod
    def collate(cls, batch: List[GameFormerFeatures]) -> GameFormerFeatures:
        """
        Implemented. See interface.
        Collates a list of features that each have batch size of 1.
        """
        device = batch[0].ego_agent_past.device

        collated_ego_agent_past = torch.stack(
            [item.ego_agent_past for item in batch], dim=0
        ).to(device)

        collated_ego_agent_future = torch.stack(
            [item.ego_agent_future for item in batch], dim=0
        ).to(device)

        collated_neighbor_agents_past = torch.stack(
            [item.neighbor_agents_past for item in batch], dim=0
        ).to(device)

        collated_neighbor_agents_future = torch.stack(
            [item.neighbor_agents_future for item in batch], dim=0
        ).to(device)

        collated_lanes = torch.stack([item.lanes for item in batch], dim=0).to(device)

        collated_crosswalks = torch.stack(
            [item.crosswalks for item in batch], dim=0
        ).to(device)

        collated_route_lanes = torch.stack(
            [item.route_lanes for item in batch], dim=0
        ).to(device)

        return GameFormerFeatures(
            ego_agent_past=collated_ego_agent_past,
            ego_agent_future=collated_ego_agent_future,
            neighbor_agents_past=collated_neighbor_agents_past,
            neighbor_agents_future=collated_neighbor_agents_future,
            lanes=collated_lanes,
            crosswalks=collated_crosswalks,
            route_lanes=collated_route_lanes,
        )
