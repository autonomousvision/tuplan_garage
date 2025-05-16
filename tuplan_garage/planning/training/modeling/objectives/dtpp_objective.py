import torch
from typing import List, Dict
from torch.nn import functional as F

from nuplan.planning.training.modeling.objectives.abstract_objective import AbstractObjective
from nuplan.planning.training.modeling.types import (
    FeaturesType,
    ScenarioListType,
    TargetsType,
)

from tuplan_garage.planning.training.preprocessing.features.dtpp.dtpp_features import DTPPFeatures
from tuplan_garage.planning.training.preprocessing.features.dtpp.dtpp_targets import DTPPTargets


class DTPPObjective(AbstractObjective):
    """
    KLoss learning objective class.
    """

    def __init__(
        self,
        scenario_type_loss_weighting: Dict[str, float],
        name: str = 'dtpp_objective',
        weight: float = 1.0,
        second_stage_weight: float = 0.2,
    ):
        """
        Initializes the class

        :param name: name of the objective
        :param weight: weight contribution to the overall loss
        """
        self._name = name
        self._weight = weight
        self._scenario_type_loss_weighting = scenario_type_loss_weighting
        self._second_stage_weight = second_stage_weight

    def name(self) -> str:
        """
        Name of the objective
        """
        return self._name

    def compute(self, predictions: FeaturesType, targets: TargetsType, scenarios: ScenarioListType) -> torch.Tensor:
        
        dtpp_features: DTPPFeatures = predictions['dtpp_features']
        stage_1_results: DTPPTargets = predictions['stage_1_results']
        stage_2_results: DTPPTargets = predictions['stage_2_results']

        neighbors_gt_future=dtpp_features.neighbor_agents_future
        neighbors_future_valid = torch.ne(neighbors_gt_future[..., :3], 0)

        stage_1_loss = calc_loss(
            neighbors=stage_1_results.neighbors_trajectories,
            ego=dtpp_features.first_stage_ego_trajectory,
            ego_regularization=stage_1_results.ego,
            scores=stage_1_results.scores, 
            weights=stage_1_results.weights,
            ego_gt=dtpp_features.ego_agent_future,
            neighbors_gt=neighbors_gt_future,
            neighbors_valid=neighbors_future_valid,
        )

        stage_2_loss = calc_loss(
            neighbors=stage_2_results.neighbors_trajectories,
            ego=dtpp_features.second_stage_ego_trajectory,
            ego_regularization=stage_2_results.ego,
            scores=stage_2_results.scores, 
            weights=stage_2_results.weights,
            ego_gt=dtpp_features.ego_agent_future,
            neighbors_gt=neighbors_gt_future,
            neighbors_valid=neighbors_future_valid,
        )

        loss = stage_1_loss + self._second_stage_weight * stage_2_loss
        loss = torch.mean(dtpp_features.expert_error_in_bounds * loss)

        return loss
    
    def get_list_of_required_target_types(self) -> List[str]:
        """
        :return list of required targets for the computations
        """
        return []
    

def calc_loss(neighbors, ego, ego_regularization, scores, weights, ego_gt, neighbors_gt, neighbors_valid):
    mask = torch.ne(ego.sum(-1), 0)
    neighbors = neighbors[:, 0] * neighbors_valid 
    
    # Compute cmp_loss
    cmp_loss = F.smooth_l1_loss(neighbors, neighbors_gt, reduction='none')  # Element-wise loss
    cmp_loss = cmp_loss * mask[:, 0, None, :, None]  # Apply mask
    valid_mask_sum = mask[:, 0].sum(dim=-1)  # Sum of valid mask elements per batch
    cmp_loss = cmp_loss.sum(dim=(1, 2, 3))  # Sum over spatial/temporal dims

    # Avoid division by zero for cmp_loss
    epsilon = 1e-6
    cmp_loss = torch.where(
        valid_mask_sum > 0,
        cmp_loss / (valid_mask_sum + epsilon),
        torch.zeros_like(cmp_loss)
    )
    assert torch.isfinite(cmp_loss).all()

    # Compute regularization_loss
    regularization_loss = F.smooth_l1_loss(ego_regularization, ego_gt, reduction='none')  # Element-wise loss
    regularization_loss = regularization_loss * mask[:, 0, :, None]  # Apply mask
    regularization_mask_sum = mask[:, 0].sum(dim=-1)  # Sum of valid mask elements per batch
    regularization_loss = regularization_loss.sum(dim=(1, 2))  # Sum over spatial dims

    # Avoid division by zero for regularization_loss
    regularization_loss = torch.where(
        regularization_mask_sum > 0,
        regularization_loss / (regularization_mask_sum + epsilon),
        torch.zeros_like(regularization_loss)
    )
    assert torch.isfinite(regularization_loss).all()

    # Compute irl_loss
    label = torch.zeros(scores.shape[0], dtype=torch.long).to(scores.device)    
    irl_loss = F.cross_entropy(scores, label, reduction='none')  # Batch loss
    irl_loss = torch.nan_to_num(irl_loss, nan=0.0)  # Replace NaN with 0
    assert torch.isfinite(irl_loss).all()

    # Compute weights_regularization
    weights_regularization = torch.square(weights).mean()

    # Final loss
    loss = cmp_loss + irl_loss + 0.1 * regularization_loss + 0.01 * weights_regularization
    return loss