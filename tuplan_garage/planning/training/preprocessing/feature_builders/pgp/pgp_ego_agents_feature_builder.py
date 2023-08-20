from __future__ import annotations

from typing import List, Tuple, Type, cast

import numpy as np
import numpy.typing as npt
from nuplan.common.actor_state.ego_state import EgoState
from nuplan.common.actor_state.state_representation import StateSE2, TimePoint
from nuplan.common.actor_state.tracked_objects import TrackedObjects
from nuplan.common.actor_state.tracked_objects_types import TrackedObjectType
from nuplan.planning.metrics.utils.state_extractors import approximate_derivatives
from nuplan.planning.scenario_builder.abstract_scenario import AbstractScenario
from nuplan.planning.scenario_builder.scenario_utils import (
    sample_indices_with_time_horizon,
)
from nuplan.planning.simulation.observation.observation_type import DetectionsTracks
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
    build_ego_features,
    compute_yaw_rate_from_states,
    extract_and_pad_agent_poses,
    extract_and_pad_agent_velocities,
    filter_agents,
)

from tuplan_garage.planning.training.preprocessing.features.pgp.pgp_ego_agents import (
    PGPEgoAgents,
)


def filter_tracked_objects_by_type(
    tracked_objects: List[DetectionsTracks], object_type: TrackedObjectType
) -> List[DetectionsTracks]:
    return [
        DetectionsTracks(
            TrackedObjects(p.tracked_objects.get_tracked_objects_of_type(object_type))
        )
        for p in tracked_objects
    ]


def compute_acceleration_from_velocities(
    agent_states_horizon: List[List[StateSE2]], time_stamps: List[TimePoint]
) -> npt.NDArray[np.float32]:

    """
    Computes the acceleration from a sequence of velocities
    Can also be used to compute speed from a sequence of positions
    :param agent_states_horizon: agent trajectories [num_frames, num_agents, 1]
    :param time_stamps: the time stamps of each frame
    :return: <np.ndarray: num_frames, num_agents, 1> where last dimension is the acceleration
    """
    speed: npt.NDArray[np.float32] = np.array(agent_states_horizon, dtype=np.float32)
    acceleration_horizon = approximate_derivatives(
        speed.transpose(),
        np.array([stamp.time_s for stamp in time_stamps]),
        window_length=3,
    )

    return cast(npt.NDArray[np.float32], acceleration_horizon)


class PGPEgoAgentsFeatureBuilder(AbstractFeatureBuilder):
    """
    Abstract class that creates model input features from database samples.
    """

    def __init__(self, history_sampling: TrajectorySampling):
        self.num_past_poses = history_sampling.num_poses
        self.past_time_horizon = history_sampling.time_horizon

    @classmethod
    def get_feature_type(cls) -> Type[AbstractModelFeature]:
        """Type of the built feature."""
        return PGPEgoAgents

    @classmethod
    def get_feature_unique_name(cls) -> str:
        """Unique string identifier of the built feature."""
        return "pgp_ego_agent_features"

    def get_features_from_scenario(self, scenario: AbstractScenario) -> PGPEgoAgents:
        """Inherited, see superclass."""
        ego_states = [
            e
            for e in scenario.get_ego_past_trajectory(
                iteration=0,
                time_horizon=self.past_time_horizon,
                num_samples=self.num_past_poses,
            )
        ]
        ego_states.append(scenario.initial_ego_state)
        detections = [
            d
            for d in scenario.get_past_tracked_objects(
                iteration=0,
                time_horizon=self.past_time_horizon,
                num_samples=self.num_past_poses,
            )
        ]
        detections.append(scenario.initial_tracked_objects)
        time_stamps = [
            t
            for t in scenario.get_past_timestamps(
                iteration=0,
                num_samples=self.num_past_poses,
                time_horizon=self.past_time_horizon,
            )
        ] + [scenario.start_time]

        return self._compute_feature(ego_states, detections, time_stamps)

    def get_features_from_simulation(
        self, current_input: PlannerInput, initialization: PlannerInitialization
    ) -> PGPEgoAgents:
        """Inherited, see superclass."""
        history = current_input.history
        present_ego_state, present_observation = history.current_state
        past_observations = history.observations[:-1]
        past_ego_states = history.ego_states[:-1]

        assert (
            history.sample_interval
        ), "SimulationHistoryBuffer sample interval is None"

        indices = sample_indices_with_time_horizon(
            self.num_past_poses, self.past_time_horizon, history.sample_interval
        )

        try:
            sampled_past_observations = [
                cast(DetectionsTracks, past_observations[-idx]).tracked_objects
                for idx in reversed(indices)
            ]
            sampled_past_ego_states = [
                past_ego_states[-idx] for idx in reversed(indices)
            ]
        except IndexError:
            raise RuntimeError(
                f"SimulationHistoryBuffer duration: {history.duration} is "
                f"too short for requested past_time_horizon: {self.past_time_horizon}. "
                f"Please increase the simulation_buffer_duration in default_simulation.yaml"
            )

        sampled_past_observations = sampled_past_observations + [
            cast(DetectionsTracks, present_observation).tracked_objects
        ]
        sampled_past_observations = [
            DetectionsTracks(obs) for obs in sampled_past_observations
        ]
        sampled_past_ego_states = sampled_past_ego_states + [present_ego_state]
        time_stamps = [state.time_point for state in sampled_past_ego_states]

        return self._compute_feature(
            sampled_past_ego_states, sampled_past_observations, time_stamps
        )

    def get_features_from_planner(
        self,
        current_input: PlannerInput,
        initialization: PlannerInitialization,
        route_roadblock_ids: List[str],  # ignored
    ) -> PGPEgoAgents:
        """Inherited, see superclass."""
        return self.get_features_from_simulation(current_input, initialization)

    def _compute_feature(
        self,
        ego_states: List[EgoState],
        past_and_current_detections: List[DetectionsTracks],
        time_stamps: List[TimePoint],
    ) -> PGPEgoAgents:

        ego_feats = self._get_ego_representation(ego_states)

        vehicle_tracks = filter_tracked_objects_by_type(
            past_and_current_detections, TrackedObjectType.VEHICLE
        )
        (
            vehicle_agent_feats,
            vehicle_agent_masks,
        ) = self._get_surrounding_agents_representation(
            vehicle_tracks, ego_states, time_stamps
        )

        pedestrian_tracks = filter_tracked_objects_by_type(
            past_and_current_detections, TrackedObjectType.PEDESTRIAN
        )
        (
            pedestrians_agent_feats,
            pedestrians_agent_masks,
        ) = self._get_surrounding_agents_representation(
            pedestrian_tracks, ego_states, time_stamps
        )

        # change semantic of mask (1:value is available, 0: value is missing) according to pgp (1: value is missing, 0: else)
        vehicle_agent_masks = (~vehicle_agent_masks).astype(np.float64)
        pedestrians_agent_masks = (~pedestrians_agent_masks).astype(np.float64)

        return PGPEgoAgents(
            vehicle_agent_feats=vehicle_agent_feats,
            vehicle_agent_masks=vehicle_agent_masks,
            pedestrians_agent_feats=pedestrians_agent_feats,
            pedestrians_agent_masks=pedestrians_agent_masks,
            ego_feats=ego_feats,
        )

    def _get_ego_representation(self, ego_states: List[EgoState]) -> np.ndarray:
        """
        :return: Ego features of shape [num_poses, 5=(x,y,v,a,omega)]
        """
        ego_poses = build_ego_features(ego_states, reverse=True)
        ego_feats = np.zeros([len(ego_states), 5])
        ego_feats[:, :2] = ego_poses[:, :2]

        for t, ego_state in enumerate(ego_states):
            ego_feats[-t, 2] = ego_state.dynamic_car_state.speed
            ego_feats[-t, 3] = ego_state.dynamic_car_state.acceleration
            ego_feats[-t, 4] = ego_state.dynamic_car_state.angular_velocity

        return ego_feats[None]

    def _get_surrounding_agents_representation(
        self,
        past_and_current_detections: List[DetectionsTracks],
        ego_states: List[EgoState],
        time_stamps: List[TimePoint],
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:

        tracked_objects: List[TrackedObjects] = [
            t.tracked_objects for t in past_and_current_detections
        ]

        anchor_ego_state = ego_states[-1]
        agent_history = filter_agents(tracked_objects, reverse=True)

        if len(agent_history[-1].tracked_objects) == 0:
            # Return empty array when there are no agents in the scene
            agent_features: npt.NDArray[np.float32] = np.empty(
                shape=(len(agent_history), 0, 5), dtype=np.float32
            )
            agent_features_mask: npt.NDArray[np.bool8] = np.full(
                shape=(len(agent_history), 0, 5), fill_value=False, dtype=np.bool8
            )
        else:
            (
                agent_states_horizon,
                agent_states_horizon_masks,
            ) = extract_and_pad_agent_poses(agent_history, reverse=True)
            (
                agent_velocities_horizon,
                agent_velocities_horizon_masks,
            ) = extract_and_pad_agent_velocities(agent_history, reverse=True)

            agent_features_mask: npt.NDArray[np.bool8] = np.stack(
                [
                    agent_states_horizon_masks,
                    agent_states_horizon_masks,
                    agent_velocities_horizon_masks,
                    agent_velocities_horizon_masks,
                    agent_states_horizon_masks,
                ],
                axis=-1,
            )

            # Get all poses relative to the ego coordinate system
            agent_relative_poses = [
                convert_absolute_to_relative_poses(anchor_ego_state.rear_axle, states)
                for states in agent_states_horizon
            ]

            agent_speeds_horizon = [
                [
                    np.linalg.norm([agent_state.x, agent_state.y], ord=2)
                    for agent_state in agent_states
                ]
                for agent_states in agent_velocities_horizon
            ]

            yaw_rate_horizon = compute_yaw_rate_from_states(
                agent_states_horizon, time_stamps
            )
            acceleration_horizon = compute_acceleration_from_velocities(
                agent_speeds_horizon, time_stamps
            )

            # Append the agent box pose, velocities  together
            agent_features_list = [
                np.hstack([poses[:, :2], np.expand_dims(speed, axis=1), np.expand_dims(acceleration, axis=1), np.expand_dims(yaw_rate, axis=1)])  # type: ignore
                for poses, speed, acceleration, yaw_rate in zip(
                    agent_relative_poses,
                    agent_speeds_horizon,
                    acceleration_horizon.transpose(),
                    yaw_rate_horizon.transpose(),
                )
            ]

            agent_features = np.stack(agent_features_list)

        agent_features = np.swapaxes(agent_features, 0, 1)
        agent_features_mask = np.swapaxes(agent_features_mask, 0, 1)

        agent_features[~agent_features_mask] = 0

        return agent_features, agent_features_mask
