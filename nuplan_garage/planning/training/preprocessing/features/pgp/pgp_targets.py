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
from torch.utils.data.dataloader import default_collate


@dataclass
class PGPTargets(AbstractModelFeature):
    """
    :pi: shape [[bs], num_nodes, num_edges]
    :visited_edges: shape [[bs], num_nodes, num_edges] Note: in the original implementation this is called evf
    :trajectory_probabilities: shape [[bs], K]: score for clustered trajectories (1/rank from clustering)
    :trajectories: shape [[bs], K, num_poses, 2]
    :traversal_coordinates [[bs], num_traversals, num_poses, 2(=xy)]
    """

    pi: FeatureDataType
    trajectory_probabilities: FeatureDataType
    visited_edges: FeatureDataType
    trajectories: FeatureDataType
    traversal_coordinates: FeatureDataType

    def __post_init__(self) -> None:
        if self.pi.ndim == 3:
            assert (
                self.visited_edges.ndim == 3
                and self.trajectory_probabilities.ndim == 2
                and self.trajectories.ndim == 4
            ), f"""if pi has a batch dimension, the other tensors need to have one as well. {self.pi.shape} {self.visited_edges.shape}
                {self.trajectory_probabilities.shape} {self.trajectories.shape}
                """
        elif self.pi.ndim == 2:
            assert (
                self.visited_edges.ndim == 2
                and self.trajectory_probabilities.ndim == 1
                and self.trajectories.ndim == 3
            ), f"""if pi has no batch dimension, the other tensors must not have one as well. {self.pi.shape} {self.visited_edges.shape}
                {self.trajectory_probabilities.shape} {self.trajectories.shape}
                """
        else:
            raise ValueError(
                f"pi has to be either [bs, num_nodes, num_edges] or [num_nodes, num_edges], but {self.pi.shape} was given"
            )
        if self.traversal_coordinates is None:
            # add dummy tensor to make sure member functions work correctly
            if isinstance(self.trajectories, torch.Tensor):
                self.traversal_coordinates = self.trajectories.detach()
            else:
                self.traversal_coordinates = self.trajectories.copy()

    def to_feature_tensor(self) -> PGPTargets:
        """
        :return object which will be collated into a batch
        """
        return PGPTargets(
            pi=to_tensor(self.pi),
            trajectory_probabilities=to_tensor(self.trajectory_probabilities),
            visited_edges=to_tensor(self.visited_edges),
            trajectories=to_tensor(self.trajectories),
            traversal_coordinates=to_tensor(self.traversal_coordinates),
        )

    def to_device(self, device: torch.device) -> PGPTargets:
        """Implemented. See interface."""
        validate_type(self.pi, torch.Tensor)
        validate_type(self.trajectory_probabilities, torch.Tensor)
        validate_type(self.visited_edges, torch.Tensor)
        validate_type(self.trajectories, torch.Tensor)
        validate_type(self.traversal_coordinates, torch.Tensor)
        return PGPTargets(
            pi=self.pi.to(device=device),
            trajectory_probabilities=self.trajectory_probabilities.to(device=device),
            visited_edges=self.visited_edges.to(device=device),
            trajectories=self.trajectories.to(device=device),
            traversal_coordinates=self.traversal_coordinates.to(device=device),
        )

    @classmethod
    def deserialize(cls, data: Dict[str, Any]) -> PGPTargets:
        """
        :return: Return dictionary of data that can be serialized
        """
        return PGPTargets(
            pi=data["pi"],
            trajectory_probabilities=data["trajectory_probabilities"],
            visited_edges=data["visited_edges"],
            trajectories=data["trajectories"],
            traversal_coordinates=data["traversal_coordinates"],
        )

    def unpack(self) -> List[PGPTargets]:
        """
        :return: Unpack a batched feature to a list of features.
        """
        return [
            PGPTargets(
                pi=pi[None],
                trajectory_probabilities=trajectory_probabilities[None],
                visited_edges=visited_edges[None],
                trajectories=trajectories[None],
                traversal_coordinates=traversal_coordinates[None],
            )
            for (
                pi,
                trajectory_probabilities,
                visited_edges,
                trajectories,
                traversal_coordinates,
            ) in zip(
                self.pi,
                self.trajectory_probabilities,
                self.visited_edges,
                self.trajectories,
                self.traversal_coordinates,
            )
        ]

    @classmethod
    def collate(cls, batch: List[PGPTargets]) -> PGPTargets:
        """
        Implemented. See interface.
        Collates a list of features that each have batch size of 1.
        """
        return PGPTargets(
            pi=default_collate([item.pi for item in batch]),
            trajectory_probabilities=default_collate(
                [item.trajectory_probabilities for item in batch]
            ),
            visited_edges=default_collate([item.visited_edges for item in batch]),
            trajectories=default_collate([item.trajectories for item in batch]),
            traversal_coordinates=default_collate(
                [item.traversal_coordinates for item in batch]
            ),
        )
