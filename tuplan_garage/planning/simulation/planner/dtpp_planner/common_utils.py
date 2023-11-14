import pathlib

import pandas as pd
from nuplan.common.actor_state.vehicle_parameters import get_pacifica_parameters
from nuplan.planning.metrics.aggregator.weighted_average_metric_aggregator import (
    WeightedAverageMetricAggregator,
)
from nuplan.planning.metrics.evaluation_metrics.common.drivable_area_compliance import (
    DrivableAreaComplianceStatistics,
)
from nuplan.planning.metrics.evaluation_metrics.common.driving_direction_compliance import (
    DrivingDirectionComplianceStatistics,
)
from nuplan.planning.metrics.evaluation_metrics.common.ego_acceleration import (
    EgoAccelerationStatistics,
)
from nuplan.planning.metrics.evaluation_metrics.common.ego_expert_l2_error import (
    EgoExpertL2ErrorStatistics,
)
from nuplan.planning.metrics.evaluation_metrics.common.ego_expert_l2_error_with_yaw import (
    EgoExpertL2ErrorWithYawStatistics,
)
from nuplan.planning.metrics.evaluation_metrics.common.ego_is_comfortable import (
    EgoIsComfortableStatistics,
)
from nuplan.planning.metrics.evaluation_metrics.common.ego_is_making_progress import (
    EgoIsMakingProgressStatistics,
)
from nuplan.planning.metrics.evaluation_metrics.common.ego_jerk import EgoJerkStatistics
from nuplan.planning.metrics.evaluation_metrics.common.ego_lane_change import (
    EgoLaneChangeStatistics,
)
from nuplan.planning.metrics.evaluation_metrics.common.ego_lat_acceleration import (
    EgoLatAccelerationStatistics,
)
from nuplan.planning.metrics.evaluation_metrics.common.ego_lat_jerk import (
    EgoLatJerkStatistics,
)
from nuplan.planning.metrics.evaluation_metrics.common.ego_lon_acceleration import (
    EgoLonAccelerationStatistics,
)
from nuplan.planning.metrics.evaluation_metrics.common.ego_lon_jerk import (
    EgoLonJerkStatistics,
)
from nuplan.planning.metrics.evaluation_metrics.common.ego_mean_speed import (
    EgoMeanSpeedStatistics,
)
from nuplan.planning.metrics.evaluation_metrics.common.ego_progress_along_expert_route import (
    EgoProgressAlongExpertRouteStatistics,
)
from nuplan.planning.metrics.evaluation_metrics.common.ego_yaw_acceleration import (
    EgoYawAccelerationStatistics,
)
from nuplan.planning.metrics.evaluation_metrics.common.ego_yaw_rate import (
    EgoYawRateStatistics,
)
from nuplan.planning.metrics.evaluation_metrics.common.no_ego_at_fault_collisions import (
    EgoAtFaultCollisionStatistics,
)
from nuplan.planning.metrics.evaluation_metrics.common.planner_expert_average_heading_error_within_bound import (
    PlannerExpertAverageHeadingErrorStatistics,
)
from nuplan.planning.metrics.evaluation_metrics.common.planner_expert_average_l2_error_within_bound import (
    PlannerExpertAverageL2ErrorStatistics,
)
from nuplan.planning.metrics.evaluation_metrics.common.planner_expert_final_heading_error_within_bound import (
    PlannerExpertFinalHeadingErrorStatistics,
)
from nuplan.planning.metrics.evaluation_metrics.common.planner_expert_final_l2_error_within_bound import (
    PlannerExpertFinalL2ErrorStatistics,
)
from nuplan.planning.metrics.evaluation_metrics.common.planner_miss_rate_within_bound import (
    PlannerMissRateStatistics,
)
from nuplan.planning.metrics.evaluation_metrics.common.speed_limit_compliance import (
    SpeedLimitComplianceStatistics,
)
from nuplan.planning.metrics.evaluation_metrics.common.time_to_collision_within_bound import (
    TimeToCollisionStatistics,
)
from nuplan.planning.metrics.metric_engine import MetricsEngine

# Parameters
T = 8  # [s] planning horizon
DT = 0.1  # [s] time interval
LENGTH = get_pacifica_parameters().front_length  # [m] vehicle front length
WHEEL_BASE = get_pacifica_parameters().wheel_base  # [m] vehicle wheel base
WIDTH = get_pacifica_parameters().width  # [m] vehicle width
MAX_LEN = 120  # [m] max length of the path


# Simulation setting
def save_runner_reports(reports, output_dir, report_name):
    """
    Save runner reports to a parquet file in the output directory.
    :param reports: Runner reports returned from each simulation.
    :param output_dir: Output directory to save the report.
    :param report_name: Report name.
    """
    report_dicts = []

    for report in map(lambda x: x.__dict__, reports):  # type: ignore
        if (planner_report := report["planner_report"]) is not None:
            planner_report_statistics = planner_report.compute_summary_statistics()
            del report["planner_report"]
            report.update(planner_report_statistics)
        report_dicts.append(report)

    df = pd.DataFrame(report_dicts)
    df["duration"] = df["end_time"] - df["start_time"]

    save_path = pathlib.Path(output_dir) / report_name
    df.to_parquet(save_path)
    print(f"Saved runner reports to {save_path}")


def build_metrics_aggregators(experiment, output_dir, aggregator_metric_dir):
    """
    Build a list of metric aggregators.
    :param cfg: Config
    :return A list of metric aggregators, and the path in which they will  save the results
    """

    aggregator_save_path = f"{output_dir}/{aggregator_metric_dir}"
    aggregator_save_path = pathlib.Path(aggregator_save_path)

    metric_aggregators = []
    metric_aggregator_config = get_aggregator_config(experiment)

    if not aggregator_save_path.exists():
        aggregator_save_path.mkdir(exist_ok=True, parents=True)

    name = metric_aggregator_config[0]
    metric_weights = metric_aggregator_config[1]
    file_name = metric_aggregator_config[2]
    multiple_metrics = metric_aggregator_config[3]
    metric_aggregators.append(
        WeightedAverageMetricAggregator(
            name, metric_weights, file_name, aggregator_save_path, multiple_metrics
        )
    )

    return metric_aggregators


def get_aggregator_config(experiment):
    if experiment == "open_loop_boxes":
        name = "open_loop_boxes_weighted_average"
        metric_weights = {
            "planner_expert_average_l2_error_within_bound": 1,
            "planner_expert_average_heading_error_within_bound": 2,
            "planner_expert_final_l2_error_within_bound": 1,
            "planner_expert_final_heading_error_within_bound": 2,
            "default": 1.0,
        }
        file_name = "open_loop_boxes_weighted_average_metrics"
        multiple_metrics = ["planner_miss_rate_within_bound"]
        challenge_name = "open_loop_boxes"

    elif experiment == "closed_loop_nonreactive_agents":
        name = "closed_loop_nonreactive_agents_weighted_average"
        metric_weights = {
            "ego_progress_along_expert_route": 5.0,
            "time_to_collision_within_bound": 5.0,
            "speed_limit_compliance": 4.0,
            "ego_is_comfortable": 2.0,
            "default": 1.0,
        }
        file_name = "closed_loop_agents_weighted_average_metrics"
        multiple_metrics = [
            "no_ego_at_fault_collisions",
            "drivable_area_compliance",
            "ego_is_making_progress",
            "driving_direction_compliance",
        ]
        challenge_name = "closed_loop_nonreactive_agents"

    elif experiment == "closed_loop_reactive_agents":
        name = "closed_loop_reactive_agents_weighted_average"
        metric_weights = {
            "ego_progress_along_expert_route": 5.0,
            "time_to_collision_within_bound": 5.0,
            "speed_limit_compliance": 4.0,
            "ego_is_comfortable": 2.0,
            "default": 1.0,
        }
        file_name = "closed_loop_agents_weighted_average_metrics"
        multiple_metrics = [
            "no_ego_at_fault_collisions",
            "drivable_area_compliance",
            "ego_is_making_progress",
            "driving_direction_compliance",
        ]
        challenge_name = "closed_loop_reactive_agents"

    else:
        raise TypeError("Experiment type not supported!")

    return name, metric_weights, file_name, multiple_metrics, challenge_name


def get_scenario_map():
    scenario_map = {
        "accelerating_at_crosswalk": [15.0, -3.0],
        "accelerating_at_stop_sign": [15.0, -3.0],
        "accelerating_at_stop_sign_no_crosswalk": [15.0, -3.0],
        "accelerating_at_traffic_light": [15.0, -3.0],
        "accelerating_at_traffic_light_with_lead": [15.0, -3.0],
        "accelerating_at_traffic_light_without_lead": [15.0, -3.0],
        "behind_bike": [15.0, -3.0],
        "behind_long_vehicle": [15.0, -3.0],
        "behind_pedestrian_on_driveable": [15.0, -3.0],
        "behind_pedestrian_on_pickup_dropoff": [15.0, -3.0],
        "changing_lane": [15.0, -3.0],
        "changing_lane_to_left": [15.0, -3.0],
        "changing_lane_to_right": [15.0, -3.0],
        "changing_lane_with_lead": [15.0, -3.0],
        "changing_lane_with_trail": [15.0, -3.0],
        "crossed_by_bike": [15.0, -3.0],
        "crossed_by_vehicle": [15.0, -3.0],
        "following_lane_with_lead": [15.0, -3.0],
        "following_lane_with_slow_lead": [15.0, -3.0],
        "following_lane_without_lead": [15.0, -3.0],
        "high_lateral_acceleration": [15.0, -3.0],
        "high_magnitude_jerk": [15.0, -3.0],
        "high_magnitude_speed": [15.0, -3.0],
        "low_magnitude_speed": [15.0, -3.0],
        "medium_magnitude_speed": [15.0, -3.0],
        "near_barrier_on_driveable": [15.0, -3.0],
        "near_construction_zone_sign": [15.0, -3.0],
        "near_high_speed_vehicle": [15.0, -3.0],
        "near_long_vehicle": [15.0, -3.0],
        "near_multiple_bikes": [15.0, -3.0],
        "near_multiple_pedestrians": [15.0, -3.0],
        "near_multiple_vehicles": [15.0, -3.0],
        "near_pedestrian_at_pickup_dropoff": [15.0, -3.0],
        "near_pedestrian_on_crosswalk": [15.0, -3.0],
        "near_pedestrian_on_crosswalk_with_ego": [15.0, -3.0],
        "near_trafficcone_on_driveable": [15.0, -3.0],
        "on_all_way_stop_intersection": [15.0, -3.0],
        "on_carpark": [15.0, -3.0],
        "on_intersection": [15.0, -3.0],
        "on_pickup_dropoff": [15.0, -3.0],
        "on_stopline_crosswalk": [15.0, -3.0],
        "on_stopline_stop_sign": [15.0, -3.0],
        "on_stopline_traffic_light": [15.0, -3.0],
        "on_traffic_light_intersection": [15.0, -3.0],
        "starting_high_speed_turn": [15.0, -3.0],
        "starting_left_turn": [15.0, -3.0],
        "starting_low_speed_turn": [15.0, -3.0],
        "starting_protected_cross_turn": [15.0, -3.0],
        "starting_protected_noncross_turn": [15.0, -3.0],
        "starting_right_turn": [15.0, -3.0],
        "starting_straight_stop_sign_intersection_traversal": [15.0, -3.0],
        "starting_straight_traffic_light_intersection_traversal": [15.0, -3.0],
        "starting_u_turn": [15.0, -3.0],
        "starting_unprotected_cross_turn": [15.0, -3.0],
        "starting_unprotected_noncross_turn": [15.0, -3.0],
        "stationary": [15.0, -3.0],
        "stationary_at_crosswalk": [15.0, -3.0],
        "stationary_at_traffic_light_with_lead": [15.0, -3.0],
        "stationary_at_traffic_light_without_lead": [15.0, -3.0],
        "stationary_in_traffic": [15.0, -3.0],
        "stopping_at_crosswalk": [15.0, -3.0],
        "stopping_at_stop_sign_no_crosswalk": [15.0, -3.0],
        "stopping_at_stop_sign_with_lead": [15.0, -3.0],
        "stopping_at_stop_sign_without_lead": [15.0, -3.0],
        "stopping_at_traffic_light_with_lead": [15.0, -3.0],
        "stopping_at_traffic_light_without_lead": [15.0, -3.0],
        "stopping_with_lead": [15.0, -3.0],
        "traversing_crosswalk": [15.0, -3.0],
        "traversing_intersection": [15.0, -3.0],
        "traversing_narrow_lane": [15.0, -3.0],
        "traversing_pickup_dropoff": [15.0, -3.0],
        "traversing_traffic_light_intersection": [15.0, -3.0],
        "waiting_for_pedestrian_to_cross": [15.0, -3.0],
    }

    return scenario_map


def get_filter_parameters(num_scenarios_per_type=None, limit_total_scenarios=None):
    scenario_types = [
        "starting_left_turn",
        "starting_right_turn",
        "starting_straight_traffic_light_intersection_traversal",
        # 'stopping_with_lead',
        "high_lateral_acceleration",
        "high_magnitude_speed",
        "low_magnitude_speed",
        "traversing_pickup_dropoff",
        # 'waiting_for_pedestrian_to_cross',
        # 'behind_long_vehicle',
        # 'stationary_in_traffic',
        "near_multiple_vehicles",
        "changing_lane",
        "following_lane_with_lead",
    ]

    scenario_tokens = None  # List of scenario tokens to include
    log_names = None  # Filter scenarios by log names
    map_names = None  # Filter scenarios by map names

    num_scenarios_per_type = num_scenarios_per_type  # Number of scenarios per type
    limit_total_scenarios = limit_total_scenarios  # Limit total scenarios (float = fraction, int = num) - this filter can be applied on top of num_scenarios_per_type
    timestamp_threshold_s = None  # Filter scenarios to ensure scenarios have more than `timestamp_threshold_s` seconds between their initial lidar timestamps
    ego_displacement_minimum_m = None  # Whether to remove scenarios where the ego moves less than a certain amount

    expand_scenarios = False  # Whether to expand multi-sample scenarios to multiple single-sample scenarios
    remove_invalid_goals = (
        True  # Whether to remove scenarios where the mission goal is invalid
    )
    shuffle = False  # Whether to shuffle the scenarios

    ego_start_speed_threshold = (
        None  # Limit to scenarios where the ego reaches a certain speed from below
    )
    ego_stop_speed_threshold = (
        None  # Limit to scenarios where the ego reaches a certain speed from above
    )
    speed_noise_tolerance = None  # Value at or below which a speed change between two timepoints should be ignored as noise.

    return (
        scenario_types,
        scenario_tokens,
        log_names,
        map_names,
        num_scenarios_per_type,
        limit_total_scenarios,
        timestamp_threshold_s,
        ego_displacement_minimum_m,
        expand_scenarios,
        remove_invalid_goals,
        shuffle,
        ego_start_speed_threshold,
        ego_stop_speed_threshold,
        speed_noise_tolerance,
    )


def get_low_level_metrics():
    low_level_metrics = {
        "ego_acceleration": EgoAccelerationStatistics(
            name="ego_acceleration", category="Dynamics"
        ),
        "ego_expert_L2_error": EgoExpertL2ErrorStatistics(
            name="ego_expert_L2_error", category="Planning", discount_factor=1
        ),
        "ego_expert_l2_error_with_yaw": EgoExpertL2ErrorWithYawStatistics(
            name="ego_expert_l2_error_with_yaw", category="Planning", discount_factor=1
        ),
        "ego_jerk": EgoJerkStatistics(
            name="ego_jerk", category="Dynamics", max_abs_mag_jerk=8.37
        ),
        "ego_lane_change": EgoLaneChangeStatistics(
            name="ego_lane_change", category="Planning", max_fail_rate=0.3
        ),
        "ego_lat_acceleration": EgoLatAccelerationStatistics(
            name="ego_lat_acceleration", category="Dynamics", max_abs_lat_accel=4.89
        ),
        "ego_lat_jerk": EgoLatJerkStatistics(name="ego_lat_jerk", category="Dynamics"),
        "ego_lon_acceleration": EgoLonAccelerationStatistics(
            name="ego_lon_acceleration",
            category="Dynamics",
            min_lon_accel=-4.05,
            max_lon_accel=2.40,
        ),
        "ego_lon_jerk": EgoLonJerkStatistics(
            name="ego_lon_jerk", category="Dynamics", max_abs_lon_jerk=4.13
        ),
        "ego_mean_speed": EgoMeanSpeedStatistics(
            name="ego_mean_speed", category="Dynamics"
        ),
        "ego_progress_along_expert_route": EgoProgressAlongExpertRouteStatistics(
            name="ego_progress_along_expert_route",
            category="Planning",
            score_progress_threshold=2,
        ),
        "ego_yaw_acceleration": EgoYawAccelerationStatistics(
            name="ego_yaw_acceleration", category="Dynamics", max_abs_yaw_accel=1.93
        ),
        "ego_yaw_rate": EgoYawRateStatistics(
            name="ego_yaw_rate", category="Dynamics", max_abs_yaw_rate=0.95
        ),
        "planner_expert_average_l2_error_within_bound": PlannerExpertAverageL2ErrorStatistics(
            name="planner_expert_average_l2_error_within_bound",
            category="Planning",
            metric_score_unit="float",
            comparison_horizon=[3, 5, 8],
            comparison_frequency=1,
            max_average_l2_error_threshold=8,
        ),
    }

    return low_level_metrics


def get_high_level_metrics(low_level_metrics):
    high_level_metrics = {
        "drivable_area_compliance": DrivableAreaComplianceStatistics(
            name="drivable_area_compliance",
            category="Planning",
            lane_change_metric=low_level_metrics["ego_lane_change"],
            max_violation_threshold=0.3,
            metric_score_unit="bool",
        ),
        "driving_direction_compliance": DrivingDirectionComplianceStatistics(
            name="driving_direction_compliance",
            category="Planning",
            lane_change_metric=low_level_metrics["ego_lane_change"],
            metric_score_unit="bool",
        ),
        "ego_is_comfortable": EgoIsComfortableStatistics(
            name="ego_is_comfortable",
            category="Violations",
            metric_score_unit="bool",
            ego_jerk_metric=low_level_metrics["ego_jerk"],
            ego_lat_acceleration_metric=low_level_metrics["ego_lat_acceleration"],
            ego_lon_acceleration_metric=low_level_metrics["ego_lon_acceleration"],
            ego_lon_jerk_metric=low_level_metrics["ego_lon_jerk"],
            ego_yaw_acceleration_metric=low_level_metrics["ego_yaw_acceleration"],
            ego_yaw_rate_metric=low_level_metrics["ego_yaw_rate"],
        ),
        "ego_is_making_progress": EgoIsMakingProgressStatistics(
            name="ego_is_making_progress",
            category="Planning",
            ego_progress_along_expert_route_metric=low_level_metrics[
                "ego_progress_along_expert_route"
            ],
            metric_score_unit="bool",
            min_progress_threshold=0.2,
        ),
        "no_ego_at_fault_collisions": EgoAtFaultCollisionStatistics(
            name="no_ego_at_fault_collisions",
            category="Dynamics",
            metric_score_unit="float",
            ego_lane_change_metric=low_level_metrics["ego_lane_change"],
        ),
        "planner_expert_average_heading_error_within_bound": PlannerExpertAverageHeadingErrorStatistics(
            name="planner_expert_average_heading_error_within_bound",
            category="Planning",
            metric_score_unit="float",
            max_average_heading_error_threshold=0.8,
            planner_expert_average_l2_error_within_bound_metric=low_level_metrics[
                "planner_expert_average_l2_error_within_bound"
            ],
        ),
        "planner_expert_final_heading_error_within_bound": PlannerExpertFinalHeadingErrorStatistics(
            name="planner_expert_final_heading_error_within_bound",
            category="Planning",
            metric_score_unit="float",
            max_final_heading_error_threshold=0.8,
            planner_expert_average_l2_error_within_bound_metric=low_level_metrics[
                "planner_expert_average_l2_error_within_bound"
            ],
        ),
        "planner_expert_final_l2_error_within_bound": PlannerExpertFinalL2ErrorStatistics(
            name="planner_expert_final_l2_error_within_bound",
            category="Planning",
            metric_score_unit="float",
            max_final_l2_error_threshold=8,
            planner_expert_average_l2_error_within_bound_metric=low_level_metrics[
                "planner_expert_average_l2_error_within_bound"
            ],
        ),
        "planner_miss_rate_within_bound": PlannerMissRateStatistics(
            name="planner_miss_rate_within_bound",
            category="Planning",
            metric_score_unit="bool",
            max_displacement_threshold=[6.0, 8.0, 16.0],
            max_miss_rate_threshold=0.3,
            planner_expert_average_l2_error_within_bound_metric=low_level_metrics[
                "planner_expert_average_l2_error_within_bound"
            ],
        ),
        "speed_limit_compliance": SpeedLimitComplianceStatistics(
            name="speed_limit_compliance",
            category="Violations",
            metric_score_unit="float",
            max_violation_threshold=1.0,
            max_overspeed_value_threshold=2.23,
            lane_change_metric=low_level_metrics["ego_lane_change"],
        ),
    }

    high_level_metrics.update(
        {
            "time_to_collision_within_bound": TimeToCollisionStatistics(
                name="time_to_collision_within_bound",
                category="Planning",
                metric_score_unit="bool",
                time_step_size=0.1,
                time_horizon=3.0,
                least_min_ttc=0.95,
                ego_lane_change_metric=low_level_metrics["ego_lane_change"],
                no_ego_at_fault_collisions_metric=high_level_metrics[
                    "no_ego_at_fault_collisions"
                ],
            )
        }
    )

    return high_level_metrics


def get_metrics_config(experiment, low_level_metrics, high_level_metrics):
    if experiment == "open_loop_boxes":
        metrics = [
            low_level_metrics["planner_expert_average_l2_error_within_bound"],
            high_level_metrics["planner_expert_final_l2_error_within_bound"],
            high_level_metrics["planner_miss_rate_within_bound"],
            high_level_metrics["planner_expert_final_heading_error_within_bound"],
            high_level_metrics["planner_expert_average_heading_error_within_bound"],
        ]

    elif (
        experiment == "closed_loop_nonreactive_agents"
        or experiment == "closed_loop_reactive_agents"
    ):
        metrics = [
            low_level_metrics["ego_lane_change"],
            low_level_metrics["ego_jerk"],
            low_level_metrics["ego_lat_acceleration"],
            low_level_metrics["ego_lon_acceleration"],
            low_level_metrics["ego_lon_jerk"],
            low_level_metrics["ego_yaw_acceleration"],
            low_level_metrics["ego_yaw_rate"],
            low_level_metrics["ego_progress_along_expert_route"],
            high_level_metrics["drivable_area_compliance"],
            high_level_metrics["no_ego_at_fault_collisions"],
            high_level_metrics["time_to_collision_within_bound"],
            high_level_metrics["speed_limit_compliance"],
            high_level_metrics["ego_is_comfortable"],
            high_level_metrics["ego_is_making_progress"],
            high_level_metrics["driving_direction_compliance"],
        ]

    else:
        raise TypeError("Experiment type not supported!")

    return metrics


def build_metrics_engine(experiment, output_dir, metric_dir):
    main_save_path = pathlib.Path(output_dir) / metric_dir
    low_level_metrics = get_low_level_metrics()
    high_level_metrics = get_high_level_metrics(low_level_metrics)
    selected_metrics = get_metrics_config(
        experiment, low_level_metrics, high_level_metrics
    )

    metric_engine = MetricsEngine(main_save_path=main_save_path)
    for metric in selected_metrics:
        metric_engine.add_metric(metric)

    return metric_engine
