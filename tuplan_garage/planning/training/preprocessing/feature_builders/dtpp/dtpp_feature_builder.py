import math
from typing import List, Type

import numpy as np
import torch
from nuplan.common.actor_state.state_representation import Point2D
from nuplan.common.actor_state.tracked_objects_types import TrackedObjectType
from nuplan.common.maps.abstract_map import MapObject, SemanticMapLayer
from nuplan.planning.scenario_builder.abstract_scenario import AbstractScenario
from nuplan.planning.simulation.planner.abstract_planner import (
    PlannerInitialization,
    PlannerInput,
)
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
from nuplan.planning.training.preprocessing.feature_builders.abstract_feature_builder import (
    AbstractFeatureBuilder,
    AbstractModelFeature,
)
from nuplan.planning.training.preprocessing.features.trajectory_utils import (
    convert_absolute_to_relative_poses,
)
from nuplan.planning.training.preprocessing.utils.agents_preprocessing import (
    sampled_past_ego_states_to_tensor,
    sampled_past_timestamps_to_tensor,
)
from shapely import Point

from tuplan_garage.planning.simulation.planner.dtpp_planner.spline_planner import (
    SplinePlanner,
)
from tuplan_garage.planning.simulation.planner.dtpp_planner.trajectory_tree_planner import (
    TrajTree,
)
from tuplan_garage.planning.training.preprocessing.feature_builders.dtpp.utils import (
    agent_future_process,
    agent_past_process,
    generate_paths,
    get_candidate_edges,
    get_candidate_paths,
    get_neighbor_vector_set_map,
    map_process,
    sampled_tracked_objects_to_tensor_list,
)
from tuplan_garage.planning.training.preprocessing.features.dtpp.dtpp_features import (
    DTPPFeatures,
)


class DTPPFeatureBuilder(AbstractFeatureBuilder):
    """
    Builder for constructing GPT features during training and simulation.
    Features include
        - surrounding agents (position, orientation and speed)
        - ego state (speed)
        - relevant on-route centerlines
    """

    def __init__(
        self,
        trajectory_sampling: TrajectorySampling,
    ) -> None:
        self.trajectory_sampling = trajectory_sampling

        self.past_time_horizon = 2  # [seconds]
        self.num_past_poses = 10 * self.past_time_horizon
        self.future_time_horizon = 8  # [seconds]
        self.max_target_speed = 15  # [m/s]
        self.first_stage_horizon = 3  # [seconds]
        self.num_future_poses = 10 * self.future_time_horizon
        self.num_agents = 20

        self._n_neighbors = 10
        self._n_candidates = 30
        self._time_length = 80

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
        self._radius = 80  # [m] query radius scope relative to the current pose.
        self._interpolation_method = "linear"  # Interpolation method to apply when interpolating to maintain fixed size map elements.

    @classmethod
    def get_feature_type(cls) -> Type[AbstractModelFeature]:
        """Type of the built feature."""
        return DTPPFeatures

    @classmethod
    def get_feature_unique_name(cls) -> str:
        """Unique string identifier of the built feature."""
        return "dtpp_features"

    def get_features_from_simulation(
        self, current_input: PlannerInput, initialization: PlannerInitialization
    ) -> DTPPFeatures:
        """Inherited, see superclass."""
        raise NotImplementedError(
            "feature builder only supports feature computation from scenario. Use observation adapter in simulation"
        )

    def get_features_from_scenario(self, scenario: AbstractScenario) -> DTPPFeatures:
        """Inherited, see superclass."""
        # map_name = scenario.map_api.map_name
        # token = scenario.token
        # self.scenario = scenario
        # self.map_api = scenario.map_api

        # get agent past tracks
        ego_agent_past, time_stamps_past = self.get_ego_agent(scenario=scenario)
        neighbor_agents_past, neighbor_agents_types = self.get_neighbor_agents(
            scenario=scenario
        )
        ego_agent_past, neighbor_agents_past, neighbor_indices = agent_past_process(
            ego_agent_past,
            time_stamps_past,
            neighbor_agents_past,
            neighbor_agents_types,
            self.num_agents,
        )

        # get vector set map
        vector_map = self.get_map(scenario=scenario)

        # get agent future tracks
        ego_agent_future = self.get_ego_agent_future(scenario=scenario)
        neighbor_agents_future = self.get_neighbor_agents_future(
            agent_index=neighbor_indices, scenario=scenario
        )

        # get candidate trajectories
        try:
            first_stage_trajs, second_stage_trajs = self.get_ego_candidate_trajectories(
                scenario=scenario
            )

            # check if the candidate trajectories are valid
            expert_error_1 = np.linalg.norm(
                ego_agent_future[None, self.first_stage_horizon * 10 - 1, :2]
                - first_stage_trajs[:, -1, :2],
                axis=-1,
            )

            expert_error_2 = np.linalg.norm(
                ego_agent_future[None, self.future_time_horizon * 10 - 1, :2]
                - second_stage_trajs[:, -1, :2],
                axis=-1,
            )

            # if stage 1 error > 1.5 or stage 2 error > 4, this scenario will be skipped (i.e. loss will be set to 0)
            expert_error_in_bounds = (
                np.min(expert_error_1) <= 1.5 or np.min(expert_error_2) <= 4
            )
            expert_error_in_bounds = np.array(
                [expert_error_in_bounds], dtype=np.float32
            )

            # sort the candidate trajectories
            first_stage_trajs = first_stage_trajs[np.argsort(expert_error_1)]
            second_stage_trajs = second_stage_trajs[np.argsort(expert_error_2)]
        except Exception as e:
            print(f"Error in getting candidate trajectories: {e}")
            # return a dummy trajectory if no feasible trajectory is found
            # this is different from the original implementation which only uses scenarios where trajectories can be found.
            # To make sure training is possible on all splits, we use a dummy trajectory here.
            # This training sample will be ignored by the loss function as the expert_error_in_bounds is set to False.
            first_stage_trajs = np.zeros(
                (1, self.first_stage_horizon * 10, 7), dtype=np.float32
            )
            second_stage_trajs = np.zeros(
                (1, self.future_time_horizon * 10, 7), dtype=np.float32
            )
            expert_error_in_bounds = np.array([False], dtype=np.float32)

        # gather data
        data = {
            "expert_error_in_bounds": expert_error_in_bounds,
            "ego_agent_past": ego_agent_past,
            "ego_agent_future": ego_agent_future,
            "first_stage_ego_trajectory": self.process_ego_trajectory(
                first_stage_trajs[..., :6]
            ),
            "second_stage_ego_trajectory": self.process_ego_trajectory(
                second_stage_trajs[..., :6]
            ),
            "neighbor_agents_past": neighbor_agents_past,
            "neighbor_agents_future": neighbor_agents_future[: self._n_neighbors],
        }
        data.update(vector_map)

        return DTPPFeatures(**data)

    def get_ego_agent(self, scenario: AbstractScenario):
        self.anchor_ego_state = scenario.initial_ego_state

        past_ego_states = scenario.get_ego_past_trajectory(
            iteration=0,
            num_samples=self.num_past_poses,
            time_horizon=self.past_time_horizon,
        )

        sampled_past_ego_states = list(past_ego_states) + [self.anchor_ego_state]
        past_ego_states_tensor = sampled_past_ego_states_to_tensor(
            sampled_past_ego_states
        )

        past_time_stamps = list(
            scenario.get_past_timestamps(
                iteration=0,
                num_samples=self.num_past_poses,
                time_horizon=self.past_time_horizon,
            )
        ) + [scenario.start_time]

        past_time_stamps_tensor = sampled_past_timestamps_to_tensor(past_time_stamps)

        return past_ego_states_tensor, past_time_stamps_tensor

    def get_neighbor_agents(self, scenario: AbstractScenario):
        present_tracked_objects = scenario.initial_tracked_objects.tracked_objects
        past_tracked_objects = [
            tracked_objects.tracked_objects
            for tracked_objects in scenario.get_past_tracked_objects(
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

    def get_map(self, scenario: AbstractScenario):
        ego_state = scenario.initial_ego_state
        ego_coords = Point2D(ego_state.rear_axle.x, ego_state.rear_axle.y)
        route_roadblock_ids = scenario.get_route_roadblock_ids()
        traffic_light_data = scenario.get_traffic_light_status_at_iteration(0)

        coords, traffic_light_data = get_neighbor_vector_set_map(
            scenario.map_api,
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

    def get_ego_agent_future(self, scenario: AbstractScenario):
        current_absolute_state = scenario.initial_ego_state

        trajectory_absolute_states = scenario.get_ego_future_trajectory(
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

    def get_neighbor_agents_future(self, agent_index, scenario: AbstractScenario):
        current_ego_state = scenario.initial_ego_state
        present_tracked_objects = scenario.initial_tracked_objects.tracked_objects

        # Get all future poses of of other agents
        future_tracked_objects = [
            tracked_objects.tracked_objects
            for tracked_objects in scenario.get_future_tracked_objects(
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

    def get_ego_candidate_trajectories(self, scenario: AbstractScenario):
        planner = SplinePlanner(self.first_stage_horizon, self.future_time_horizon)

        # Gather information about the environment
        route_roadblock_ids = scenario.get_route_roadblock_ids()
        observation = scenario.get_tracked_objects_at_iteration(0)
        ego_state = scenario.initial_ego_state
        route_roadblocks: List[MapObject] = []

        for id_ in route_roadblock_ids:
            block = scenario.map_api.get_map_object(id_, SemanticMapLayer.ROADBLOCK)
            block = block or scenario.map_api.get_map_object(
                id_, SemanticMapLayer.ROADBLOCK_CONNECTOR
            )
            route_roadblocks.append(block)

        candidate_lane_edge_ids = [
            edge.id
            for block in route_roadblocks
            if block
            for edge in block.interior_edges
        ]

        # Get obstacles
        object_types = [
            TrackedObjectType.VEHICLE,
            TrackedObjectType.BARRIER,
            TrackedObjectType.CZONE_SIGN,
            TrackedObjectType.TRAFFIC_CONE,
            TrackedObjectType.GENERIC_OBJECT,
        ]
        objects = observation.tracked_objects.get_tracked_objects_of_types(object_types)
        obstacles = []
        for obj in objects:
            if obj.box.geometry.distance(ego_state.car_footprint.geometry) > 30:
                continue

            if obj.tracked_object_type == TrackedObjectType.VEHICLE:
                if obj.velocity.magnitude() < 0.01:
                    obstacles.append(obj.box)
            else:
                obstacles.append(obj.box)

        # Get starting block
        starting_block = None
        cur_point = (
            scenario.initial_ego_state.rear_axle.x,
            scenario.initial_ego_state.rear_axle.y,
        )
        closest_distance = math.inf

        for block in route_roadblocks:
            for edge in block.interior_edges:
                distance = edge.polygon.distance(Point(cur_point))
                if distance < closest_distance:
                    starting_block = block
                    closest_distance = distance

            if np.isclose(closest_distance, 0):
                break

        # Get starting edges
        edges = get_candidate_edges(ego_state, starting_block)
        candidate_paths = get_candidate_paths(edges, ego_state, candidate_lane_edge_ids)
        paths = generate_paths(candidate_paths, obstacles, ego_state)
        speed_limit = edges[0].speed_limit_mps or self.max_target_speed

        # Initial tree (root node)
        # traj: x, y, heading, velocity, acceleration, curvature, time
        state = torch.tensor(
            [
                [
                    0,
                    0,
                    0,
                    ego_state.dynamic_car_state.rear_axle_velocity_2d.x,
                    ego_state.dynamic_car_state.rear_axle_acceleration_2d.x,
                    0,
                    0,
                ]
            ]
        )
        tree = TrajTree(state, None, 0)

        # 1st stage expand
        tree.expand_children(paths, self.first_stage_horizon, speed_limit, planner)
        leaves = TrajTree.get_children(tree)
        first_trajs = np.stack([leaf.total_traj[1:].numpy() for leaf in leaves]).astype(
            np.float32
        )

        # 2nd stage expand
        for leaf in leaves:
            leaf.expand_children(
                paths,
                self.future_time_horizon - self.first_stage_horizon,
                speed_limit,
                planner,
            )

        # Get all leaves
        leaves = TrajTree.get_children(leaves)
        second_trajs = np.stack(
            [leaf.total_traj[1:].numpy() for leaf in leaves]
        ).astype(np.float32)

        return first_trajs, second_trajs

    def process_ego_trajectory(self, ego_trajectory):
        trajectory = np.zeros(
            (self._n_candidates, self._time_length, 6), dtype=np.float32
        )
        if ego_trajectory.shape[0] > self._n_candidates:
            ego_trajectory = ego_trajectory[: self._n_candidates]

        if ego_trajectory.shape[1] < self._time_length:
            trajectory[
                : ego_trajectory.shape[0], : ego_trajectory.shape[1]
            ] = ego_trajectory
        else:
            trajectory[: ego_trajectory.shape[0]] = ego_trajectory

        return trajectory
