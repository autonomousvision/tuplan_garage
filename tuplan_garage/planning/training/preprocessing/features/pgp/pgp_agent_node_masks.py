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
class PGPAgentNodeMasks(AbstractModelFeature):
    """

    :vehicle_node_masks:  shape [num_nodes, num_vehicles]
    :pedestrian_node_masks:  shape [num_nodes, num_pedestrians]
    """

    vehicle_node_masks: FeatureDataType
    pedestrian_node_masks: FeatureDataType

    def __post_init__(self) -> None:
        n_dim = self.vehicle_node_masks.ndim
        if not self.pedestrian_node_masks.ndim == n_dim:
            raise AssertionError(
                "All feature tensors need to have the same number of dimensions!"
            )
        if (
            n_dim == 4
            and not self.pedestrian_node_masks.shape[0]
            == self.vehicle_node_masks.shape[0]
        ):
            raise AssertionError(
                "All feature tensors need to have the same batch_size!"
            )

    def to_feature_tensor(self) -> PGPAgentNodeMasks:
        """
        :return object which will be collated into a batch
        """
        return PGPAgentNodeMasks(
            vehicle_node_masks=to_tensor(self.vehicle_node_masks),
            pedestrian_node_masks=to_tensor(self.pedestrian_node_masks),
        )

    def to_device(self, device: torch.device) -> PGPAgentNodeMasks:
        """Implemented. See interface."""
        validate_type(self.vehicle_node_masks, torch.Tensor)
        validate_type(self.pedestrian_node_masks, torch.Tensor)
        return PGPAgentNodeMasks(
            vehicle_node_masks=self.vehicle_node_masks.to(device=device),
            pedestrian_node_masks=self.pedestrian_node_masks.to(device=device),
        )

    @classmethod
    def deserialize(cls, data: Dict[str, Any]) -> PGPAgentNodeMasks:
        """
        :return: Return dictionary of data that can be serialized
        """
        return PGPAgentNodeMasks(
            vehicle_node_masks=data["vehicle_node_masks"],
            pedestrian_node_masks=data["pedestrian_node_masks"],
        )

    def unpack(self) -> List[PGPAgentNodeMasks]:
        """
        :return: Unpack a batched feature to a list of features.
        """
        return [
            PGPAgentNodeMasks(
                vehicle_node_masks=vehicle_node_masks[None],
                pedestrian_node_masks=pedestrian_node_masks[None],
            )
            for vehicle_node_masks, pedestrian_node_masks in zip(
                self.vehicle_node_masks, self.pedestrian_node_masks
            )
        ]

    @property
    def batch_size(self) -> int:
        """
        :return: number of batches
        """
        if len(self.vehicle_node_masks.shape) == 3:
            return self.vehicle_node_masks.shape[0]
        else:
            return None

    @classmethod
    def collate(cls, batch: List[PGPAgentNodeMasks]) -> PGPAgentNodeMasks:
        """
        Implemented. See interface.
        Collates a list of features that each have batch size of 1.
        """
        assert batch[0].vehicle_node_masks.ndim == 2
        device = batch[0].vehicle_node_masks.device
        max_nodes = max([item.vehicle_node_masks.shape[0] for item in batch])
        max_vehicles = max([item.vehicle_node_masks.shape[1] for item in batch])
        max_pedestrians = max([item.pedestrian_node_masks.shape[1] for item in batch])

        collated_vehicle_node_masks = torch.ones(
            [len(batch), max_nodes, max_vehicles], device=device
        )
        collated_pedestrian_node_masks = torch.ones(
            [len(batch), max_nodes, max_pedestrians], device=device
        )

        for i, item in enumerate(batch):
            num_nodes = item.vehicle_node_masks.shape[0]
            num_vehicles = item.vehicle_node_masks.shape[1]
            num_pedestrians = item.pedestrian_node_masks.shape[1]
            collated_vehicle_node_masks[
                i, :num_nodes, :num_vehicles
            ] = item.vehicle_node_masks
            collated_pedestrian_node_masks[
                i, :num_nodes, :num_pedestrians
            ] = item.pedestrian_node_masks

        return PGPAgentNodeMasks(
            vehicle_node_masks=collated_vehicle_node_masks,
            pedestrian_node_masks=collated_pedestrian_node_masks,
        )
