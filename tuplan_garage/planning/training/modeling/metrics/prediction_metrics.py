from typing import List

import torch
from nuplan.planning.training.modeling.metrics.abstract_training_metric import (
    AbstractTrainingMetric,
)
from nuplan.planning.training.modeling.types import TargetsType

from tuplan_garage.planning.training.preprocessing.features.trajectories_multimodal import (
    MultiModalTrajectories,
)


class MinAverageDisplacementError(AbstractTrainingMetric):
    """
    Metric representing the minimal displacement L2 error of all predicted trajectories averaged from all poses of a trajectory.
    """

    def __init__(self, name: str = "min_avg_displacement_error") -> None:
        """
        Initializes the class.

        :param name: the name of the metric (used in logger)
        """
        self._name = name

    def name(self) -> str:
        """
        Name of the metric
        """
        return self._name

    def get_list_of_required_target_types(self) -> List[str]:
        """Implemented. See interface."""
        return ["multimodal_trajectories"]

    def compute(self, predictions: TargetsType, targets: TargetsType) -> torch.Tensor:
        """
        Computes the metric given the ground truth targets and the model's predictions.

        :param predictions: model's predictions
        :param targets: ground truth targets from the dataset
        :return: metric scalar tensor
        """
        predicted_trajectories: MultiModalTrajectories = predictions[
            "multimodal_trajectories"
        ]
        targets_trajectory: MultiModalTrajectories = targets["multimodal_trajectories"]

        ade = torch.norm(
            predicted_trajectories.trajectories[..., :2]
            - targets_trajectory.trajectories[..., :2],
            dim=-1,
        ).mean(dim=-1)
        min_ade = torch.min(ade, dim=-1)[0]

        return torch.mean(min_ade)


class MinFinalDisplacementError(AbstractTrainingMetric):
    """
    Metric representing the displacement L2 error from the final pose of a trajectory.
    """

    def __init__(self, name: str = "min_final_displacement_error") -> None:
        """
        Initializes the class.

        :param name: the name of the metric (used in logger)
        """
        self._name = name

    def name(self) -> str:
        """
        Name of the metric
        """
        return self._name

    def get_list_of_required_target_types(self) -> List[str]:
        """Implemented. See interface."""
        return ["multimodal_trajectories"]

    def compute(self, predictions: TargetsType, targets: TargetsType) -> torch.Tensor:
        """
        Computes the metric given the ground truth targets and the model's predictions.

        :param predictions: model's predictions
        :param targets: ground truth targets from the dataset
        :return: metric scalar tensor
        """
        predicted_trajectories: MultiModalTrajectories = predictions[
            "multimodal_trajectories"
        ]
        targets_trajectory: MultiModalTrajectories = targets["multimodal_trajectories"]

        fde = torch.norm(
            predicted_trajectories.trajectories[..., -1, :2]
            - targets_trajectory.trajectories[..., -1, :2],
            dim=-1,
        )
        min_fde = torch.min(fde, dim=-1)[0]

        return torch.mean(min_fde)
