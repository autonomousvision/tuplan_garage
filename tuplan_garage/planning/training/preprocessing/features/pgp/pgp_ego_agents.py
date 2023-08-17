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
class PGPEgoAgents(AbstractModelFeature):
    """
    Graph Map Structure for pgp model

    :vehicle_agent_feats: shape [max_vehicles, num_poses, nbr_feat_size=5]
    :vehicle_agent_masks: shape [max_vehicles, num_poses, nbr_feat_size=5]
    :pedestrians_agent_feats: shape [max_peds, num_poses, nbr_feat_size=5]
    :pedestrians_agent_masks: shape [max_peds, num_poses, nbr_feat_size=5]
    :ego_feats: shape [1, num_poses, nbr_feat_size=5]
    """

    vehicle_agent_feats: FeatureDataType
    vehicle_agent_masks: FeatureDataType
    pedestrians_agent_feats: FeatureDataType
    pedestrians_agent_masks: FeatureDataType
    ego_feats: FeatureDataType

    def __post_init__(self) -> None:
        n_dim = self.ego_feats.ndim
        same_ndim = (
            (self.vehicle_agent_feats.ndim == n_dim)
            and (self.vehicle_agent_masks.ndim == n_dim)
            and (self.pedestrians_agent_feats.ndim == n_dim)
            and (self.pedestrians_agent_masks.ndim == n_dim)
        )
        if not same_ndim:
            raise AssertionError(
                "All feature tensors need to have the same number of dimensions!"
            )

        num_poses = self.ego_feats.shape[-2]
        same_num_poses = (
            (self.vehicle_agent_feats.shape[-2] == num_poses)
            and (self.vehicle_agent_masks.shape[-2] == num_poses)
            and (self.pedestrians_agent_feats.shape[-2] == num_poses)
            and (self.pedestrians_agent_masks.shape[-2] == num_poses)
        )
        if not same_num_poses:
            raise AssertionError(
                "All feature tensors need to have the same number of poses!"
            )

        if n_dim == 4:
            batch_size = self.ego_feats.shape[0]
            same_batch_size = (
                (self.vehicle_agent_feats.shape[0] == batch_size)
                and (self.vehicle_agent_masks.shape[0] == batch_size)
                and (self.pedestrians_agent_feats.shape[0] == batch_size)
                and (self.pedestrians_agent_masks.shape[0] == batch_size)
            )
            if not same_batch_size:
                raise AssertionError(
                    "All feature tensors need to have the same batch_size!"
                )

    def to_feature_tensor(self) -> PGPEgoAgents:
        """
        :return object which will be collated into a batch
        """
        return PGPEgoAgents(
            vehicle_agent_feats=to_tensor(self.vehicle_agent_feats),
            vehicle_agent_masks=to_tensor(self.vehicle_agent_masks),
            pedestrians_agent_feats=to_tensor(self.pedestrians_agent_feats),
            pedestrians_agent_masks=to_tensor(self.pedestrians_agent_masks),
            ego_feats=to_tensor(self.ego_feats),
        )

    def to_device(self, device: torch.device) -> PGPEgoAgents:
        """Implemented. See interface."""
        validate_type(self.vehicle_agent_feats, torch.Tensor)
        validate_type(self.vehicle_agent_masks, torch.Tensor)
        return PGPEgoAgents(
            vehicle_agent_feats=self.vehicle_agent_feats.to(device=device),
            vehicle_agent_masks=self.vehicle_agent_masks.to(device=device),
            pedestrians_agent_feats=self.pedestrians_agent_feats.to(device=device),
            pedestrians_agent_masks=self.pedestrians_agent_masks.to(device=device),
            ego_feats=self.ego_feats.to(device=device),
        )

    @classmethod
    def deserialize(cls, data: Dict[str, Any]) -> PGPEgoAgents:
        """
        :return: Return dictionary of data that can be serialized
        """
        return PGPEgoAgents(
            vehicle_agent_feats=data["vehicle_agent_feats"],
            vehicle_agent_masks=data["vehicle_agent_masks"],
            pedestrians_agent_feats=data["pedestrians_agent_feats"],
            pedestrians_agent_masks=data["pedestrians_agent_masks"],
            ego_feats=data["ego_feats"],
        )

    def unpack(self) -> List[PGPEgoAgents]:
        """
        :return: Unpack a batched feature to a list of features.
        """
        return [
            PGPEgoAgents(
                vehicle_agent_feats[None],
                vehicle_agent_masks[None],
                pedestrians_agent_feats[None],
                pedestrians_agent_masks[None],
                ego_feats[None],
            )
            for vehicle_agent_feats, vehicle_agent_masks, pedestrians_agent_feats, pedestrians_agent_masks, ego_feats in zip(
                self.vehicle_agent_feats,
                self.vehicle_agent_masks,
                self.pedestrians_agent_feats,
                self.pedestrians_agent_masks,
                self.ego_feats,
            )
        ]

    @staticmethod
    def ego_state_dim() -> int:
        """
        :return: ego state dimension
        """
        return 5

    @staticmethod
    def agents_states_dim() -> int:
        """
        :return: agent state dimension
        """
        return 5

    @property
    def batch_size(self) -> int:
        """
        :return: number of batches
        """
        if len(self.ego_feats.shape) == 4:
            return self.ego_feats.shape[0]
        else:
            return None

    @classmethod
    def collate(cls, batch: List[PGPEgoAgents]) -> PGPEgoAgents:
        """
        Implemented. See interface.
        Collates a list of features that each have batch size of 1.
        """
        assert batch[0].ego_feats.ndim == 3
        device = batch[0].ego_feats.device
        num_poses = batch[0].ego_feats.shape[1]
        max_agents = max([item.vehicle_agent_feats.shape[0] for item in batch])
        max_peds = max([item.pedestrians_agent_feats.shape[0] for item in batch])

        collated_ego_features = torch.stack(
            [item.ego_feats for item in batch], dim=0
        ).to(device)
        collated_vehicle_agent_feats = torch.zeros(
            [len(batch), max_agents, num_poses, 5], device=device
        )
        collated_vehicle_agent_masks = torch.ones_like(
            collated_vehicle_agent_feats, device=device
        )
        collated_pedestrians_agent_feats = torch.zeros(
            [len(batch), max_peds, num_poses, 5], device=device
        )
        collated_pedestrians_agent_masks = torch.ones_like(
            collated_pedestrians_agent_feats, device=device
        )

        for i, item in enumerate(batch):
            num_agents, num_peds = (
                item.vehicle_agent_feats.shape[0],
                item.pedestrians_agent_feats.shape[0],
            )
            collated_vehicle_agent_feats[i, :num_agents, :] = item.vehicle_agent_feats
            collated_vehicle_agent_masks[i, :num_agents, :] = item.vehicle_agent_masks
            collated_pedestrians_agent_feats[
                i, :num_peds, :
            ] = item.pedestrians_agent_feats
            collated_pedestrians_agent_masks[
                i, :num_peds, :
            ] = item.pedestrians_agent_masks

        return PGPEgoAgents(
            ego_feats=collated_ego_features,
            vehicle_agent_feats=collated_vehicle_agent_feats,
            vehicle_agent_masks=collated_vehicle_agent_masks,
            pedestrians_agent_feats=collated_pedestrians_agent_feats,
            pedestrians_agent_masks=collated_pedestrians_agent_masks,
        )
