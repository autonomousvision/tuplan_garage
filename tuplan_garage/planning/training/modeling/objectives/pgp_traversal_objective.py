from typing import Dict, List

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

from tuplan_garage.planning.training.preprocessing.features.pgp.pgp_targets import (
    PGPTargets,
)


class PGPTraversalObjective(AbstractObjective):
    """
    Abstract learning objective class.
    """

    def __init__(
        self, scenario_type_loss_weighting: Dict[str, float], weight: float = 1.0
    ) -> None:
        self._name = "pgp_traversal_objective"
        self._weight = weight
        self._scenario_type_loss_weighting = scenario_type_loss_weighting

    def name(self) -> str:
        """
        Name of the objective
        """
        return self._name

    def compute(
        self,
        predictions: FeaturesType,
        targets: TargetsType,
        scenarios: ScenarioListType,
    ) -> torch.Tensor:
        """
        Computes the objective's loss given the ground truth targets and the model's predictions.

        :param predictions: model's predictions
        :param targets: ground truth targets from the dataset
        :return: loss scalar tensor
        """
        predictions: PGPTargets = predictions["pgp_targets"]
        pi = predictions.pi
        evf_gt = predictions.visited_edges
        loss_weights = extract_scenario_type_weight(
            scenarios,
            self._scenario_type_loss_weighting,
            device=predictions.trajectories.device,
        )
        loss = -torch.sum((pi * evf_gt).flatten(start_dim=1), dim=1)
        return self._weight * torch.mean(loss * loss_weights)

    def get_list_of_required_target_types(self) -> List[str]:
        """
        :return list of required targets for the computations
        """
        return []
