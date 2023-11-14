import torch
from typing import List, Dict
from torch.nn import functional as F

from nuplan.planning.training.modeling.objectives.abstract_objective import AbstractObjective
from nuplan.planning.training.modeling.types import (
    FeaturesType,
    ScenarioListType,
    TargetsType,
)
from tuplan_garage.planning.training.preprocessing.features.gameformer.gameformer_targets import GameFormerTargets

class KLoss(AbstractObjective):
    """
    KLoss learning objective class.
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
        return self._name

    def compute(self, predictions: FeaturesType, targets: TargetsType, scenarios: ScenarioListType) -> torch.Tensor:
        """
        Computes the objective's loss given the ground truth targets and the model's predictions.

        :param predictions: model's predictions
        :param targets: ground truth targets from the dataset
        :return: loss scalar tensor
        """
        # outputs = predictions["decoder_outputs"]
        gameformer_targets: GameFormerTargets = predictions["gameformer_targets"]
        outputs = gameformer_targets.decoder_outputs
        ego_future = gameformer_targets.ego_agent_future
        neighbors_future = gameformer_targets.neighbor_agents_future
        neighbors_future_valid = torch.ne(neighbors_future[..., :2], 0)

        loss=torch.tensor(0.0, device=ego_future.device)
        levels = len(outputs.keys()) // 2 
        gt_future = torch.cat([ego_future[:, None], neighbors_future], dim=1)

        for k in range(levels):
            trajectories = outputs[f'level_{k}_interactions']
            scores = outputs[f'level_{k}_scores']
            predictions = trajectories[:, 1:] * neighbors_future_valid[:, :, None, :, 0, None]
            plan = trajectories[:, :1]
            trajectories = torch.cat([plan, predictions], dim=1)
            il_loss, future, best_mode = imitation_loss(trajectories, scores, gt_future)
            loss += il_loss 
        return loss

    def get_list_of_required_target_types(self) -> List[str]:
        """
        :return list of required targets for the computations
        """
        return []


def imitation_loss(gmm, scores, ground_truth):
    B, N = gmm.shape[0], gmm.shape[1]
    distance = torch.norm(gmm[:, :, :, :, :2] - ground_truth[:, :, None, :, :2], dim=-1)
    best_mode = torch.argmin(distance.mean(-1), dim=-1)

    mu = gmm[..., :2]
    best_mode_mu = mu[torch.arange(B)[:, None, None], torch.arange(N)[None, :, None], best_mode[:, :, None]]
    best_mode_mu = best_mode_mu.squeeze(2)
    dx = ground_truth[..., 0] - best_mode_mu[..., 0]
    dy = ground_truth[..., 1] - best_mode_mu[..., 1]

    cov = gmm[..., 2:]
    best_mode_cov = cov[torch.arange(B)[:, None, None], torch.arange(N)[None, :, None], best_mode[:, :, None]]
    best_mode_cov = best_mode_cov.squeeze(2)
    log_std_x = torch.clamp(best_mode_cov[..., 0], -2, 2)
    log_std_y = torch.clamp(best_mode_cov[..., 1], -2, 2)
    std_x = torch.exp(log_std_x)
    std_y = torch.exp(log_std_y)

    gmm_loss = log_std_x + log_std_y + 0.5 * (torch.square(dx/std_x) + torch.square(dy/std_y))
    gmm_loss = torch.mean(gmm_loss)

    score_loss = F.cross_entropy(scores.permute(0, 2, 1), best_mode, label_smoothing=0.2, reduction='none')
    score_loss = score_loss * torch.ne(ground_truth[:, :, 0, 0], 0)
    score_loss = torch.mean(score_loss)
    
    loss = gmm_loss + score_loss

    return loss, best_mode_mu, best_mode



