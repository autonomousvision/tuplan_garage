from __future__ import annotations

from typing import Type

from nuplan.common.actor_state.state_representation import Point2D
from nuplan.planning.scenario_builder.abstract_scenario import AbstractScenario
from nuplan.planning.simulation.planner.abstract_planner import (
    PlannerInitialization,
    PlannerInput,
)
from nuplan.planning.training.preprocessing.feature_builders.abstract_feature_builder import (
    AbstractFeatureBuilder,
)
from nuplan.planning.training.preprocessing.features.abstract_model_feature import (
    AbstractModelFeature,
)
from nuplan.planning.training.preprocessing.features.trajectory_utils import (
    convert_absolute_to_relative_poses,
)
from nuplan.planning.training.preprocessing.utils.agents_preprocessing import (
    sampled_past_ego_states_to_tensor,
    sampled_past_timestamps_to_tensor,
)

from tuplan_garage.planning.training.preprocessing.feature_builders.gameformer.data_utils import (
    agent_future_process,
    agent_past_process,
    get_neighbor_vector_set_map,
    map_process,
    sampled_tracked_objects_to_tensor_list,
)
from tuplan_garage.planning.training.preprocessing.features.gameformer.gameformer_features import (
    GameFormerFeatures,
)


class GameFormerFeatureBuilder(AbstractFeatureBuilder):
    """
    Abstract class that creates model input features from database samples.
    """

    def __init__(
        self,
        num_agents: int = 20,
    ) -> None:

        self.past_time_horizon = 2  # [seconds]
        self.num_past_poses = 10 * self.past_time_horizon
        self.future_time_horizon = 8  # [seconds]
        self.num_future_poses = 10 * self.future_time_horizon
        self.num_agents = num_agents

        self._map_features = [
            "LANE",
            "ROUTE_LANES",
            "CROSSWALK",
        ]  # name of map features to be extracted.
        self._max_elements = {
            "LANE": 40,
            "ROUTE_LANES": 10,
            "CROSSWALK": 5,
        }  # maximum number of elements to extract per feature layer.
        self._max_points = {
            "LANE": 50,
            "ROUTE_LANES": 50,
            "CROSSWALK": 30,
        }  # maximum number of points per feature to extract per feature layer.
        self._radius = 60  # [m] query radius scope relative to the current pose.
        self._interpolation_method = "linear"  # Interpolation method to apply when interpolating to maintain fixed size map elements.

    @classmethod
    def get_feature_type(cls) -> Type[AbstractModelFeature]:
        """Type of the built feature."""
        return GameFormerFeatures

    @classmethod
    def get_feature_unique_name(cls) -> str:
        """Unique string identifier of the built feature."""
        return "gameformer_features"

    def get_features_from_simulation(
        self, current_input: PlannerInput, initialization: PlannerInitialization
    ) -> AbstractModelFeature:
        """
        Constructs model input features from simulation history.
        :param current_input: Iteration specific inputs for building the feature.
        :param initialization: Additional data require for building the feature.
        :return: Constructed features.
        """
        raise NotImplementedError

    def get_features_from_scenario(
        self, scenario: AbstractScenario
    ) -> AbstractModelFeature:
        """
        Constructs model input features from a database samples.
        :param scenario: Generic scenario
        :return: Constructed features
        """

        scenario._map_name
        scenario.token
        self.scenario = scenario

        # get agent past tracks
        ego_agent_past, time_stamps_past = self.get_ego_agent()
        neighbor_agents_past, neighbor_agents_types = self.get_neighbor_agents()
        ego_agent_past, neighbor_agents_past, neighbor_indices = agent_past_process(
            ego_agent_past,
            time_stamps_past,
            neighbor_agents_past,
            neighbor_agents_types,
            self.num_agents,
        )

        # get vector set map
        vector_map = self.get_map(scenario.map_api)

        # get agent future tracks
        ego_agent_future = self.get_ego_agent_future()
        neighbor_agents_future = self.get_neighbor_agents_future(neighbor_indices)

        # gather data
        # data = {"map_name": map_name, "token": token, "ego_agent_past": ego_agent_past, "ego_agent_future": ego_agent_future,
        #         "neighbor_agents_past": neighbor_agents_past, "neighbor_agents_future": neighbor_agents_future}
        data = {
            "ego_agent_past": ego_agent_past,
            "ego_agent_future": ego_agent_future,
            "neighbor_agents_past": neighbor_agents_past,
            "neighbor_agents_future": neighbor_agents_future,
        }
        data.update(vector_map)

        return GameFormerFeatures(*data.values())

    def get_ego_agent(self):
        self.anchor_ego_state = self.scenario.initial_ego_state

        past_ego_states = self.scenario.get_ego_past_trajectory(
            iteration=0,
            num_samples=self.num_past_poses,
            time_horizon=self.past_time_horizon,
        )

        sampled_past_ego_states = list(past_ego_states) + [self.anchor_ego_state]
        past_ego_states_tensor = sampled_past_ego_states_to_tensor(
            sampled_past_ego_states
        )

        past_time_stamps = list(
            self.scenario.get_past_timestamps(
                iteration=0,
                num_samples=self.num_past_poses,
                time_horizon=self.past_time_horizon,
            )
        ) + [self.scenario.start_time]

        past_time_stamps_tensor = sampled_past_timestamps_to_tensor(past_time_stamps)

        return past_ego_states_tensor, past_time_stamps_tensor

    def get_neighbor_agents(self):
        present_tracked_objects = self.scenario.initial_tracked_objects.tracked_objects
        past_tracked_objects = [
            tracked_objects.tracked_objects
            for tracked_objects in self.scenario.get_past_tracked_objects(
                iteration=0,
                time_horizon=self.past_time_horizon,
                num_samples=self.num_past_poses,
            )
        ]

        sampled_past_observations = past_tracked_objects + [present_tracked_objects]
        (
            past_tracked_objects_tensor_list,
            past_tracked_objects_types,
        ) = sampled_tracked_objects_to_tensor_list(sampled_past_observations)

        return past_tracked_objects_tensor_list, past_tracked_objects_types

    def get_map(self, map_api):
        ego_state = self.scenario.initial_ego_state
        ego_coords = Point2D(ego_state.rear_axle.x, ego_state.rear_axle.y)
        route_roadblock_ids = self.scenario.get_route_roadblock_ids()
        traffic_light_data = self.scenario.get_traffic_light_status_at_iteration(0)

        coords, traffic_light_data = get_neighbor_vector_set_map(
            map_api,
            self._map_features,
            ego_coords,
            self._radius,
            route_roadblock_ids,
            traffic_light_data,
        )

        vector_map = map_process(
            ego_state.rear_axle,
            coords,
            traffic_light_data,
            self._map_features,
            self._max_elements,
            self._max_points,
            self._interpolation_method,
        )

        return vector_map

    def get_ego_agent_future(self):
        current_absolute_state = self.scenario.initial_ego_state

        trajectory_absolute_states = self.scenario.get_ego_future_trajectory(
            iteration=0,
            num_samples=self.num_future_poses,
            time_horizon=self.future_time_horizon,
        )

        # Get all future poses of the ego relative to the ego coordinate system
        trajectory_relative_poses = convert_absolute_to_relative_poses(
            current_absolute_state.rear_axle,
            [state.rear_axle for state in trajectory_absolute_states],
        )

        return trajectory_relative_poses

    def get_neighbor_agents_future(self, agent_index):
        current_ego_state = self.scenario.initial_ego_state
        present_tracked_objects = self.scenario.initial_tracked_objects.tracked_objects

        # Get all future poses of of other agents
        future_tracked_objects = [
            tracked_objects.tracked_objects
            for tracked_objects in self.scenario.get_future_tracked_objects(
                iteration=0,
                time_horizon=self.future_time_horizon,
                num_samples=self.num_future_poses,
            )
        ]

        sampled_future_observations = [present_tracked_objects] + future_tracked_objects
        future_tracked_objects_tensor_list, _ = sampled_tracked_objects_to_tensor_list(
            sampled_future_observations
        )
        agent_futures = agent_future_process(
            current_ego_state,
            future_tracked_objects_tensor_list,
            self.num_agents,
            agent_index,
        )

        return agent_futures
