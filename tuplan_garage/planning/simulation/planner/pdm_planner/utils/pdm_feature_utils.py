from typing import List, Optional

import numpy as np
from nuplan.common.actor_state.state_representation import TimeDuration, TimePoint
from nuplan.planning.scenario_builder.scenario_utils import (
    sample_indices_with_time_horizon,
)
from nuplan.planning.simulation.planner.abstract_planner import PlannerInput
from nuplan.planning.simulation.trajectory.interpolated_trajectory import (
    InterpolatedTrajectory,
)
from nuplan.planning.training.modeling.torch_module_wrapper import TorchModuleWrapper
from shapely.geometry import Point

from tuplan_garage.planning.simulation.planner.pdm_planner.utils.pdm_array_representation import (
    ego_states_to_state_array,
)
from tuplan_garage.planning.simulation.planner.pdm_planner.utils.pdm_enums import (
    StateIndex,
)
from tuplan_garage.planning.simulation.planner.pdm_planner.utils.pdm_geometry_utils import (
    convert_absolute_to_relative_se2_array,
)
from tuplan_garage.planning.simulation.planner.pdm_planner.utils.pdm_path import PDMPath
from tuplan_garage.planning.training.preprocessing.feature_builders.pdm_feature_builder import (
    get_ego_acceleration,
    get_ego_position,
    get_ego_velocity,
)
from tuplan_garage.planning.training.preprocessing.features.pdm_feature import (
    PDMFeature,
)


def create_pdm_feature(
    model: TorchModuleWrapper,
    planner_input: PlannerInput,
    centerline: PDMPath,
    closed_loop_trajectory: Optional[InterpolatedTrajectory] = None,
    device: str = "cpu",
) -> PDMFeature:
    """
    Creates a PDMFeature (for PDM-Open and PDM-Offset) during simulation
    :param model: torch model (used to retrieve parameters)
    :param planner_input: nuPlan's planner input during simulation
    :param centerline: centerline path of PDM-* methods
    :param closed_loop_trajectory: trajectory of PDM-Closed (ignored if None)
    :return: PDMFeature dataclass
    """

    # feature building
    num_past_poses = model.history_sampling.num_poses
    past_time_horizon = model.history_sampling.time_horizon

    history = planner_input.history
    current_ego_state, _ = history.current_state
    past_ego_states = history.ego_states[:-1]

    indices = sample_indices_with_time_horizon(
        num_past_poses, past_time_horizon, history.sample_interval
    )
    sampled_past_ego_states = [past_ego_states[-idx] for idx in reversed(indices)]
    sampled_past_ego_states = sampled_past_ego_states + [current_ego_state]

    ego_position = get_ego_position(sampled_past_ego_states)
    ego_velocity = get_ego_velocity(sampled_past_ego_states)
    ego_acceleration = get_ego_acceleration(sampled_past_ego_states)

    # extract planner centerline
    current_progress: float = centerline.project(
        Point(*current_ego_state.rear_axle.array)
    )
    centerline_progress_values = (
        np.arange(model.centerline_samples, dtype=np.float64)
        * model.centerline_interval
        + current_progress
    )  # distance values to interpolate
    planner_centerline = convert_absolute_to_relative_se2_array(
        current_ego_state.rear_axle,
        centerline.interpolate(centerline_progress_values, as_array=True),
    )  # convert to relative coords

    if closed_loop_trajectory is not None:
        current_time: TimePoint = current_ego_state.time_point
        future_step_time: TimeDuration = TimeDuration.from_s(
            model.trajectory_sampling.step_time
        )
        future_time_points: List[TimePoint] = [
            current_time + future_step_time * (i + 1)
            for i in range(model.trajectory_sampling.num_poses)
        ]
        trajectory_ego_states = closed_loop_trajectory.get_state_at_times(
            future_time_points
        )  # sample to model trajectory

        planner_trajectory = ego_states_to_state_array(
            trajectory_ego_states
        )  # convert to array
        planner_trajectory = planner_trajectory[
            ..., StateIndex.STATE_SE2
        ]  # drop values
        planner_trajectory = convert_absolute_to_relative_se2_array(
            current_ego_state.rear_axle, planner_trajectory
        )  # convert to relative coords

    else:
        # use centerline as dummy value
        planner_trajectory = planner_centerline

    pdm_feature = PDMFeature(
        ego_position=ego_position,
        ego_velocity=ego_velocity,
        ego_acceleration=ego_acceleration,
        planner_centerline=planner_centerline,
        planner_trajectory=planner_trajectory,
    )

    pdm_feature = pdm_feature.to_feature_tensor()
    pdm_feature = pdm_feature.to_device(device)
    pdm_feature = pdm_feature.collate([pdm_feature])

    return pdm_feature
