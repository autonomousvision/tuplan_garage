from typing import Dict, List, cast

import torch
from nuplan.planning.training.modeling.objectives.abstract_objective import (
    AbstractObjective,
)
from nuplan.planning.training.modeling.objectives.scenario_weight_utils import (
    extract_scenario_type_weight,
)
from nuplan.planning.training.modeling.types import (
    FeaturesType,
    ScenarioListType,
    TargetsType,
)
from nuplan.planning.training.preprocessing.features.trajectory import Trajectory


class L1Objective(AbstractObjective):
    """
    Objective for imitating the expert behavior via an L1-Loss function.
    """

    def __init__(
        self, scenario_type_loss_weighting: Dict[str, float], weight: float = 1.0
    ):
        """
        Initializes the class

        :param name: name of the objective
        :param weight: weight contribution to the overall loss
        """
        self._name = "l1_objective"
        self._weight = weight
        self._loss_function = torch.nn.modules.loss.L1Loss(reduction="none")
        self._scenario_type_loss_weighting = scenario_type_loss_weighting

    def name(self) -> str:
        """
        Name of the objective
        """
        return self._name

    def get_list_of_required_target_types(self) -> List[str]:
        """Implemented. See interface."""
        return ["trajectory"]

    def compute(
        self,
        predictions: FeaturesType,
        targets: TargetsType,
        scenarios: ScenarioListType,
    ) -> torch.Tensor:
        """
        Computes the objective's loss given the ground truth targets and the model's predictions
        and weights it based on a fixed weight factor.

        :param predictions: model's predictions
        :param targets: ground truth targets from the dataset
        :return: loss scalar tensor
        """
        predicted_trajectory = cast(Trajectory, predictions["trajectory"])
        targets_trajectory = cast(Trajectory, targets["trajectory"])

        scenario_weights = extract_scenario_type_weight(
            scenarios,
            self._scenario_type_loss_weighting,
            device=predicted_trajectory.data.device,
        )

        batch_size = predicted_trajectory.data.shape[0]
        assert predicted_trajectory.data.shape == targets_trajectory.data.shape

        loss = self._loss_function(
            predicted_trajectory.data.view(batch_size, -1),
            targets_trajectory.data.view(batch_size, -1),
        )

        return self._weight * torch.mean(loss * scenario_weights[..., None])
