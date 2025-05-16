from typing import Dict
from dataclasses import asdict

from nuplan.planning.training.modeling.torch_module_wrapper import TorchModuleWrapper
from nuplan.planning.training.preprocessing.target_builders.ego_trajectory_target_builder import EgoTrajectoryTargetBuilder
from nuplan.planning.training.preprocessing.features.trajectory import Trajectory
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
from nuplan.planning.training.modeling.types import FeaturesType, TargetsType

from tuplan_garage.planning.simulation.planner.dtpp_planner.scenario_tree_prediction import Encoder, Decoder
from tuplan_garage.planning.training.preprocessing.feature_builders.dtpp.dtpp_feature_builder import DTPPFeatureBuilder
from tuplan_garage.planning.training.preprocessing.features.dtpp.dtpp_features import DTPPFeatures
from tuplan_garage.planning.training.preprocessing.features.dtpp.dtpp_targets import DTPPTargets

class DTPPModel(TorchModuleWrapper):
    def __init__(
        self,
        encoder: Encoder,
        decoder: Decoder,
        future_trajectory_sampling: TrajectorySampling,
        past_trajectory_sampling: TrajectorySampling,
    ):
        
        feature_builders = [DTPPFeatureBuilder(past_trajectory_sampling)]
        target_builders = [EgoTrajectoryTargetBuilder(future_trajectory_sampling)]

        super(DTPPModel, self).__init__(
            future_trajectory_sampling=future_trajectory_sampling,
            feature_builders=feature_builders,
            target_builders=target_builders)
        
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, features: Dict[str, DTPPFeatures]) -> TargetsType:

        inputs = asdict(features['dtpp_features'])

        # encode
        encoder_outputs = self.encoder(inputs)
        

        # first stage prediction
        first_stage_trajectory = features['dtpp_features'].first_stage_ego_trajectory
        # first_stage_trajectory = batch[7].to(args.device)

        neighbors_trajectories, scores, ego, weights = \
            self.decoder(encoder_outputs, first_stage_trajectory, inputs['neighbor_agents_past'], 30)
        stage_1_results = DTPPTargets(
            neighbors_trajectories=neighbors_trajectories,
            scores=scores,
            ego=ego,
            weights=weights,
        )

        # second stage prediction
        second_stage_trajectory = features['dtpp_features'].second_stage_ego_trajectory
        # second_stage_trajectory = batch[8].to(args.device)
        neighbors_trajectories, scores, ego, weights = \
            self.decoder(encoder_outputs, second_stage_trajectory, inputs['neighbor_agents_past'], 80)
        stage_2_results = DTPPTargets(
            neighbors_trajectories=neighbors_trajectories,
            scores=scores,
            ego=ego,
            weights=weights
        )

        out = {
            'stage_1_results': stage_1_results,
            'stage_2_results': stage_2_results,
        }
        out.update(features)
        return out