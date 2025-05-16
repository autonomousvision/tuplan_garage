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
class DTPPTargets(AbstractModelFeature):
    """
    # TODO: describe components and their shapes
    """

    neighbors_trajectories: FeatureDataType
    scores: FeatureDataType
    ego: FeatureDataType
    weights: FeatureDataType

    def to_feature_tensor(self) -> DTPPTargets:
        """
        :return object which will be collated into a batch
        """
        return DTPPTargets(
            neighbors_trajectories=to_tensor(self.neighbors_trajectories),
            scores=to_tensor(self.scores),
            ego=to_tensor(self.ego),
            weights=to_tensor(self.weights),
        )

    def to_device(self, device: torch.device) -> DTPPTargets:
        """Implemented. See interface."""
        validate_type(self.neighbors_trajectories, torch.Tensor)
        validate_type(self.scores, torch.Tensor)
        validate_type(self.ego, torch.Tensor)
        validate_type(self.weights, torch.Tensor)

        return DTPPTargets(
            neighbors_trajectories=self.neighbors_trajectories.to(device),
            scores=self.scores.to(device),
            ego=self.ego.to(device),
            weights=self.weights.to(device),
        )

    @classmethod
    def deserialize(cls, data: Dict[str, Any]) -> DTPPTargets:
        """
        :return: Return dictionary of data that can be serialized
        """
        return DTPPTargets(
            neighbors_trajectories=data["neighbors_trajectories"],
            scores=data["scores"],
            ego=data["ego"],
            weights=data["weights"],
        )

    def unpack(self) -> List[DTPPTargets]:
        """
        :return: Unpack a batched feature to a list of features.
        """
        return [
            DTPPTargets(
                neighbors_trajectories=neighbors_trajectories[None],
                scores=scores[None],
                ego=ego[None],
                weights=weights[None],
            )
            for (neighbors_trajectories, scores, ego, weights) in zip(
                self.neighbors_trajectories,
                self.scores,
                self.ego,
                self.weights,
            )
        ]
