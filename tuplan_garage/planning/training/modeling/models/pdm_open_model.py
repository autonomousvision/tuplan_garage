import torch
import torch.nn as nn
from nuplan.planning.simulation.planner.abstract_planner import AbstractPlanner
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
from nuplan.planning.training.modeling.torch_module_wrapper import TorchModuleWrapper
from nuplan.planning.training.modeling.types import FeaturesType, TargetsType
from nuplan.planning.training.preprocessing.features.trajectory import Trajectory
from nuplan.planning.training.preprocessing.target_builders.ego_trajectory_target_builder import (
    EgoTrajectoryTargetBuilder,
)

from tuplan_garage.planning.simulation.planner.pdm_planner.utils.pdm_enums import (
    SE2Index,
)
from tuplan_garage.planning.training.preprocessing.feature_builders.pdm_feature_builder import (
    PDMFeatureBuilder,
)
from tuplan_garage.planning.training.preprocessing.features.pdm_feature import (
    PDMFeature,
)


class PDMOpenModel(TorchModuleWrapper):
    """
    Wrapper around PDM-Open MLP that consumes the ego history (position, velocity, acceleration)
    and the centerline to regresses ego's future trajectory.
    """

    def __init__(
        self,
        trajectory_sampling: TrajectorySampling,
        history_sampling: TrajectorySampling,
        planner: AbstractPlanner,
        centerline_samples: int = 120,
        centerline_interval: float = 1.0,
        hidden_dim: int = 512,
    ):
        """
        Constructor for PDMOpenModel
        :param trajectory_sampling: Sampling parameters of future trajectory
        :param history_sampling: Sampling parameters of past ego states
        :param planner: Planner for centerline extraction
        :param centerline_samples: Number of poses on the centerline, defaults to 120
        :param centerline_interval: Distance between centerline poses [m], defaults to 1.0
        :param hidden_dim: Size of the hidden dimensionality of the MLP, defaults to 512
        """

        feature_builders = [
            PDMFeatureBuilder(
                trajectory_sampling,
                history_sampling,
                planner,
                centerline_samples,
                centerline_interval,
            )
        ]

        target_builders = [
            EgoTrajectoryTargetBuilder(trajectory_sampling),
        ]

        self.trajectory_sampling = trajectory_sampling
        self.history_sampling = history_sampling

        self.centerline_samples = centerline_samples
        self.centerline_interval = centerline_interval

        self.hidden_dim = hidden_dim

        super().__init__(
            feature_builders=feature_builders,
            target_builders=target_builders,
            future_trajectory_sampling=trajectory_sampling,
        )

        self.state_encoding = nn.Sequential(
            nn.Linear(
                (history_sampling.num_poses + 1) * 3 * len(SE2Index), self.hidden_dim
            ),
            nn.ReLU(),
        )

        self.centerline_encoding = nn.Sequential(
            nn.Linear(self.centerline_samples * len(SE2Index), self.hidden_dim),
            nn.ReLU(),
        )

        self.planner_head = nn.Sequential(
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
            nn.Dropout(0.1),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, trajectory_sampling.num_poses * len(SE2Index)),
        )

    def forward(self, features: FeaturesType) -> TargetsType:
        """
        Predict
        :param features: input features containing
                        {
                            "pdm_features": PDFeature,
                        }
        :return: targets: predictions from network
                        {
                            "trajectory": Trajectory,
                        }
        """
        input: PDMFeature = features["pdm_features"]

        batch_size = input.ego_position.shape[0]

        ego_position = input.ego_position.reshape(batch_size, -1).float()
        ego_velocity = input.ego_velocity.reshape(batch_size, -1).float()
        ego_acceleration = input.ego_acceleration.reshape(batch_size, -1).float()

        # encode ego history states
        state_features = torch.cat(
            [ego_position, ego_velocity, ego_acceleration], dim=-1
        )
        state_encodings = self.state_encoding(state_features)

        # encode planner centerline
        planner_centerline = input.planner_centerline.reshape(batch_size, -1).float()
        centerline_encodings = self.centerline_encoding(planner_centerline)

        # decode future trajectory
        planner_features = torch.cat([state_encodings, centerline_encodings], dim=-1)
        output_trajectory = self.planner_head(planner_features)
        output_trajectory = output_trajectory.reshape(batch_size, -1, len(SE2Index))

        return {"trajectory": Trajectory(data=output_trajectory)}
