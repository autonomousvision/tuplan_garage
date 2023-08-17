from typing import Dict, List, Tuple, cast

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

from tuplan_garage.planning.training.preprocessing.features.pgp.pgp_targets import (
    PGPTargets,
)


def min_ade(
    traj: torch.Tensor, traj_gt: torch.Tensor, masks: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Computes average displacement error for the best trajectory is a set, with respect to ground truth
    :param traj: predictions, shape [batch_size, num_modes, sequence_length, 2]
    :param traj_gt: ground truth trajectory, shape [batch_size, sequence_length, 2]
    :param masks: masks for varying length ground truth, shape [batch_size, sequence_length]
    :return errs, inds: errors and indices for modes with min error, shape [batch_size]
    """
    num_modes = traj.shape[1]

    traj_gt_rpt = traj_gt.unsqueeze(1).repeat(1, num_modes, 1, 1)
    masks_rpt = masks.unsqueeze(1).repeat(1, num_modes, 1)
    err = traj_gt_rpt - traj[:, :, :, 0:2]
    err = torch.pow(err, exponent=2)
    err = torch.sum(err, dim=3)
    err = torch.pow(err, exponent=0.5)
    err = torch.sum(err * (1 - masks_rpt), dim=2) / torch.sum((1 - masks_rpt), dim=2)
    err, inds = torch.min(err, dim=1)

    return err, inds


class minADEObjective(AbstractObjective):
    def __init__(
        self,
        scenario_type_loss_weighting: Dict[str, float],
        k: int,
        weight: float = 1.0,
    ) -> None:
        self._name = "minADE_objective"
        self._weight = weight
        self._k = k
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

        # Unpack arguments
        predictions: PGPTargets = predictions["pgp_targets"]
        traj_gt = cast(Trajectory, targets["trajectory"]).data[..., :2]
        traj = predictions.trajectories
        probs = predictions.trajectory_probabilities
        loss_weights = extract_scenario_type_weight(
            scenarios,
            self._scenario_type_loss_weighting,
            device=predictions.trajectories.device,
        )

        # Useful params
        batch_size = probs.shape[0]
        num_pred_modes = traj.shape[1]
        sequence_length = traj.shape[2]

        # Masks for variable length ground truth trajectories
        masks = (
            targets["masks"]
            if type(targets) == dict and "masks" in targets.keys()
            else torch.zeros(batch_size, sequence_length).to(traj.device)
        )

        min_k = min(self._k, num_pred_modes)

        _, inds_topk = torch.topk(probs, min_k, dim=1)
        batch_inds = torch.arange(batch_size).unsqueeze(1).repeat(1, min_k)
        traj_topk = traj[batch_inds, inds_topk]

        errs, _ = min_ade(traj_topk, traj_gt, masks)

        return self._weight * torch.mean(loss_weights * errs)

    def get_list_of_required_target_types(self) -> List[str]:
        """
        :return list of required targets for the computations
        """
        return ["trajectory"]
