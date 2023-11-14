import torch
from typing import List, Dict
from torch.nn import functional as F

from nuplan.planning.training.modeling.objectives.abstract_objective import AbstractObjective
from nuplan.planning.training.modeling.types import (
    FeaturesType,
    ScenarioListType,
    TargetsType,
)


class PlanLoss(AbstractObjective):
    """
    PlanLoss learning objective class.
    """

    def __init__(
        self,
        scenario_type_loss_weighting: Dict[str, float],
        name: str = 'agent_imitation_objective',
        weight: float = 1.0,
    ):
        """
        Initializes the class

        :param name: name of the objective
        :param weight: weight contribution to the overall loss
        """
        self._name = name
        self._weight = weight
        self._scenario_type_loss_weighting = scenario_type_loss_weighting

    def name(self) -> str:
        """
        Name of the objective
        """
        return "plan_loss"

    def compute(self, predictions: FeaturesType, targets: TargetsType, scenarios: ScenarioListType) -> torch.Tensor:
        """
        Computes the objective's loss given the ground truth targets and the model's predictions.

        :param predictions: model's predictions
        :param targets: ground truth targets from the dataset
        :return: loss scalar tensor
        """
        plan = predictions["trajectory"].data
        ego_future = predictions["gameformer_targets"].ego_agent_future

        loss = F.smooth_l1_loss(plan, ego_future)
        loss += F.smooth_l1_loss(plan[:, -1], ego_future[:, -1])
        return loss

    def get_list_of_required_target_types(self) -> List[str]:
        """
        :return list of required targets for the computations
        """
        return []




def planning_loss(plan, ego_future):
    loss = F.smooth_l1_loss(plan, ego_future)
    loss += F.smooth_l1_loss(plan[:, -1], ego_future[:, -1])

    return loss