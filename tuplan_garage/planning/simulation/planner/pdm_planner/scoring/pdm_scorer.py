import copy
from typing import Dict, List, Optional

import numpy as np
import numpy.typing as npt
from nuplan.common.actor_state.ego_state import EgoState
from nuplan.common.actor_state.state_representation import StateSE2
from nuplan.common.actor_state.tracked_objects_types import AGENT_TYPES
from nuplan.common.maps.abstract_map import AbstractMap
from nuplan.common.maps.abstract_map_objects import LaneGraphEdgeMapObject
from nuplan.common.maps.maps_datatypes import SemanticMapLayer
from nuplan.planning.metrics.utils.collision_utils import CollisionType
from nuplan.planning.simulation.observation.idm.utils import (
    is_agent_ahead,
    is_agent_behind,
)
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
from shapely import Point, creation

from tuplan_garage.planning.simulation.planner.pdm_planner.observation.pdm_observation import (
    PDMObservation,
)
from tuplan_garage.planning.simulation.planner.pdm_planner.observation.pdm_occupancy_map import (
    PDMOccupancyMap,
)
from tuplan_garage.planning.simulation.planner.pdm_planner.scoring.pdm_comfort_metrics import (
    ego_is_comfortable,
)
from tuplan_garage.planning.simulation.planner.pdm_planner.scoring.pdm_scorer_utils import (
    get_collision_type,
)
from tuplan_garage.planning.simulation.planner.pdm_planner.utils.pdm_array_representation import (
    coords_array_to_polygon_array,
    state_array_to_coords_array,
)
from tuplan_garage.planning.simulation.planner.pdm_planner.utils.pdm_enums import (
    BBCoordsIndex,
    EgoAreaIndex,
    MultiMetricIndex,
    StateIndex,
    WeightedMetricIndex,
)
from tuplan_garage.planning.simulation.planner.pdm_planner.utils.pdm_path import PDMPath

# constants
# TODO: Add to config
WEIGHTED_METRICS_WEIGHTS = np.zeros(len(WeightedMetricIndex), dtype=np.float64)
WEIGHTED_METRICS_WEIGHTS[WeightedMetricIndex.PROGRESS] = 5.0
WEIGHTED_METRICS_WEIGHTS[WeightedMetricIndex.TTC] = 5.0
WEIGHTED_METRICS_WEIGHTS[WeightedMetricIndex.COMFORTABLE] = 2.0

# TODO: Add to config
DRIVING_DIRECTION_COMPLIANCE_THRESHOLD = 2.0  # [m] (driving direction)
DRIVING_DIRECTION_VIOLATION_THRESHOLD = 6.0  # [m] (driving direction)
STOPPED_SPEED_THRESHOLD = 5e-03  # [m/s] (ttc)
PROGRESS_DISTANCE_THRESHOLD = 0.1  # [m] (progress)


class PDMScorer:
    """Class to score proposals in PDM pipeline. Re-implements nuPlan's closed-loop metrics."""

    def __init__(self, proposal_sampling: TrajectorySampling):
        """
        Constructor of PDMScorer
        :param proposal_sampling: Sampling parameters for proposals
        """
        self._proposal_sampling = proposal_sampling

        # lazy loaded
        self._initial_ego_state: Optional[EgoState] = None
        self._observation: Optional[PDMObservation] = None
        self._centerline: Optional[PDMPath] = None
        self._route_lane_dict: Optional[Dict[str, LaneGraphEdgeMapObject]] = None
        self._drivable_area_map: Optional[PDMOccupancyMap] = None
        self._map_api: Optional[AbstractMap] = None

        self._num_proposals: Optional[int] = None
        self._states: Optional[npt.NDArray[np.float64]] = None
        self._ego_coords: Optional[npt.NDArray[np.float64]] = None
        self._ego_polygons: Optional[npt.NDArray[np.object_]] = None

        self._ego_areas: Optional[npt.NDArray[np.bool_]] = None

        self._multi_metrics: Optional[npt.NDArray[np.float64]] = None
        self._weighted_metrics: Optional[npt.NDArray[np.float64]] = None
        self._progress_raw: Optional[npt.NDArray[np.float64]] = None

        self._collision_time_idcs: Optional[npt.NDArray[np.float64]] = None
        self._ttc_time_idcs: Optional[npt.NDArray[np.float64]] = None

    def time_to_at_fault_collision(self, proposal_idx: int) -> float:
        """
        Returns time to at-fault collision for given proposal
        :param proposal_idx: index for proposal
        :return: time to infraction
        """
        return (
            self._collision_time_idcs[proposal_idx]
            * self._proposal_sampling.interval_length
        )

    def time_to_ttc_infraction(self, proposal_idx: int) -> float:
        """
        Returns time to ttc infraction for given proposal
        :param proposal_idx: index for proposal
        :return: time to infraction
        """
        return (
            self._ttc_time_idcs[proposal_idx] * self._proposal_sampling.interval_length
        )

    def score_proposals(
        self,
        states: npt.NDArray[np.float64],
        initial_ego_state: EgoState,
        observation: PDMObservation,
        centerline: PDMPath,
        route_lane_dict: Dict[str, LaneGraphEdgeMapObject],
        drivable_area_map: PDMOccupancyMap,
        map_api: AbstractMap,
    ) -> npt.NDArray[np.float64]:
        """
        Scores proposal similar to nuPlan's closed-loop metrics
        :param states: array representation of simulated proposals
        :param initial_ego_state: ego-vehicle state at current iteration
        :param observation: PDM's observation class
        :param centerline: path of the centerline
        :param route_lane_dict: dictionary containing on-route lanes
        :param drivable_area_map: Occupancy map of drivable are polygons
        :param map_api: map object
        :return: array containing score of each proposal
        """

        # initialize & lazy load class values
        self._reset(
            states,
            initial_ego_state,
            observation,
            centerline,
            route_lane_dict,
            drivable_area_map,
            map_api,
        )

        # fill value ego-area array (used across multiple metrics)
        self._calculate_ego_area()

        # 1. multiplicative metrics
        self._calculate_no_at_fault_collision()
        self._calculate_driving_direction_compliance()
        self._calculate_drivable_area_compliance()

        # 2. weighted metrics
        self._calculate_progress()
        self._calculate_ttc()
        self._calculate_is_comfortable()

        return self._aggregate_scores()

    def _aggregate_scores(self) -> npt.NDArray[np.float64]:
        """
        Aggregates metrics with multiplicative and weighted average.
        :return: array containing score of each proposal
        """

        # accumulate multiplicative metrics
        multiplicate_metric_scores = self._multi_metrics.prod(axis=0)

        # normalize and fill progress values
        raw_progress = self._progress_raw * multiplicate_metric_scores
        max_raw_progress = np.max(raw_progress)
        if max_raw_progress > PROGRESS_DISTANCE_THRESHOLD:
            normalized_progress = raw_progress / max_raw_progress
        else:
            normalized_progress = np.ones(len(raw_progress), dtype=np.float64)
            normalized_progress[multiplicate_metric_scores == 0.0] = 0.0
        self._weighted_metrics[WeightedMetricIndex.PROGRESS] = normalized_progress

        # accumulate weighted metrics
        weighted_metric_scores = (
            self._weighted_metrics * WEIGHTED_METRICS_WEIGHTS[..., None]
        ).sum(axis=0)
        weighted_metric_scores /= WEIGHTED_METRICS_WEIGHTS.sum()

        # calculate final scores
        final_scores = multiplicate_metric_scores * weighted_metric_scores

        return final_scores

    def _reset(
        self,
        states: npt.NDArray[np.float64],
        initial_ego_state: EgoState,
        observation: PDMObservation,
        centerline: PDMPath,
        route_lane_dict: Dict[str, LaneGraphEdgeMapObject],
        drivable_area_map: PDMOccupancyMap,
        map_api: AbstractMap,
    ) -> None:
        """
        Resets metric values and lazy loads input classes.
        :param states: array representation of simulated proposals
        :param initial_ego_state: ego-vehicle state at current iteration
        :param observation: PDM's observation class
        :param centerline: path of the centerline
        :param route_lane_dict: dictionary containing on-route lanes
        :param drivable_area_map: Occupancy map of drivable are polygons
        :param map_api: map object
        """
        assert states.ndim == 3
        assert states.shape[1] == self._proposal_sampling.num_poses + 1
        assert states.shape[2] == StateIndex.size()

        self._initial_ego_state = initial_ego_state
        self._observation = observation
        self._centerline = centerline
        self._route_lane_dict = route_lane_dict
        self._drivable_area_map = drivable_area_map
        self._map_api = map_api

        self._num_proposals = states.shape[0]

        # save ego state values
        self._states = states

        # calculate coordinates of ego corners and center
        self._ego_coords = state_array_to_coords_array(
            states, initial_ego_state.car_footprint.vehicle_parameters
        )

        # initialize all ego polygons from corners
        self._ego_polygons = coords_array_to_polygon_array(self._ego_coords)

        # zero initialize all remaining arrays.
        self._ego_areas = np.zeros(
            (
                self._num_proposals,
                self._proposal_sampling.num_poses + 1,
                len(EgoAreaIndex),
            ),
            dtype=np.bool_,
        )
        self._multi_metrics = np.zeros(
            (len(MultiMetricIndex), self._num_proposals), dtype=np.float64
        )
        self._weighted_metrics = np.zeros(
            (len(WeightedMetricIndex), self._num_proposals), dtype=np.float64
        )
        self._progress_raw = np.zeros(self._num_proposals, dtype=np.float64)

        # initialize infraction arrays with infinity (meaning no infraction occurs)
        self._collision_time_idcs = np.zeros(self._num_proposals, dtype=np.float64)
        self._ttc_time_idcs = np.zeros(self._num_proposals, dtype=np.float64)
        self._collision_time_idcs.fill(np.inf)
        self._ttc_time_idcs.fill(np.inf)

    def _calculate_ego_area(self) -> None:
        """
        Determines the area of proposals over time.
        Areas are (1) in multiple lanes, (2) non-drivable area, or (3) oncoming traffic
        """

        n_proposals, n_horizon, n_points, _ = self._ego_coords.shape
        coordinates = self._ego_coords.reshape(n_proposals * n_horizon * n_points, 2)

        in_polygons = self._drivable_area_map.points_in_polygons(coordinates)
        in_polygons = in_polygons.reshape(
            len(self._drivable_area_map), n_proposals, n_horizon, n_points
        ).transpose(
            1, 2, 0, 3
        )  # shape: n_proposals, n_horizon, n_polygons, n_points

        drivable_area_on_route_idcs: List[int] = [
            idx
            for idx, token in enumerate(self._drivable_area_map.tokens)
            if token in self._route_lane_dict.keys()
        ]  # index mask for on-route lanes

        corners_in_polygon = in_polygons[..., :-1]  # ignore center coordinate
        center_in_polygon = in_polygons[..., -1]  # only center

        # in_multiple_lanes: if
        # - more than one drivable polygon contains at least one corner
        # - no polygon contains all corners
        batch_multiple_lanes_mask = np.zeros((n_proposals, n_horizon), dtype=np.bool_)
        batch_multiple_lanes_mask = (corners_in_polygon.sum(axis=-1) > 0).sum(
            axis=-1
        ) > 1

        batch_not_single_lanes_mask = np.zeros((n_proposals, n_horizon), dtype=np.bool_)
        batch_not_single_lanes_mask = np.all(
            corners_in_polygon.sum(axis=-1) != 4, axis=-1
        )

        multiple_lanes_mask = np.logical_and(
            batch_multiple_lanes_mask, batch_not_single_lanes_mask
        )
        self._ego_areas[multiple_lanes_mask, EgoAreaIndex.MULTIPLE_LANES] = True

        # in_nondrivable_area: if at least one corner is not within any drivable polygon
        batch_nondrivable_area_mask = np.zeros((n_proposals, n_horizon), dtype=np.bool_)
        batch_nondrivable_area_mask = (corners_in_polygon.sum(axis=-2) > 0).sum(
            axis=-1
        ) < 4
        self._ego_areas[
            batch_nondrivable_area_mask, EgoAreaIndex.NON_DRIVABLE_AREA
        ] = True

        # in_oncoming_traffic: if center not in any drivable polygon that is on-route
        batch_oncoming_traffic_mask = np.zeros((n_proposals, n_horizon), dtype=np.bool_)
        batch_oncoming_traffic_mask = (
            center_in_polygon[..., drivable_area_on_route_idcs].sum(axis=-1) == 0
        )
        self._ego_areas[
            batch_oncoming_traffic_mask, EgoAreaIndex.ONCOMING_TRAFFIC
        ] = True

    def _calculate_no_at_fault_collision(self) -> None:
        """
        Re-implementation of nuPlan's at-fault collision metric.
        """
        no_collision_scores = np.ones(self._num_proposals, dtype=np.float64)

        proposal_collided_track_ids = {
            proposal_idx: copy.deepcopy(self._observation.collided_track_ids)
            for proposal_idx in range(self._num_proposals)
        }

        for time_idx in range(self._proposal_sampling.num_poses + 1):
            ego_polygons = self._ego_polygons[:, time_idx]
            intersecting = self._observation[time_idx].query(
                ego_polygons, predicate="intersects"
            )

            if len(intersecting) == 0:
                continue

            for proposal_idx, geometry_idx in zip(intersecting[0], intersecting[1]):
                token = self._observation[time_idx].tokens[geometry_idx]
                if (self._observation.red_light_token in token) or (
                    token in proposal_collided_track_ids[proposal_idx]
                ):
                    continue

                ego_in_multiple_lanes_or_nondrivable_area = (
                    self._ego_areas[proposal_idx, time_idx, EgoAreaIndex.MULTIPLE_LANES]
                    or self._ego_areas[
                        proposal_idx, time_idx, EgoAreaIndex.NON_DRIVABLE_AREA
                    ]
                )

                tracked_object = self._observation.unique_objects[token]

                # classify collision
                collision_type: CollisionType = get_collision_type(
                    self._states[proposal_idx, time_idx],
                    self._ego_polygons[proposal_idx, time_idx],
                    tracked_object,
                    self._observation[time_idx][token],
                )
                collisions_at_stopped_track_or_active_front: bool = collision_type in [
                    CollisionType.ACTIVE_FRONT_COLLISION,
                    CollisionType.STOPPED_TRACK_COLLISION,
                ]
                collision_at_lateral: bool = (
                    collision_type == CollisionType.ACTIVE_LATERAL_COLLISION
                )

                # 1. at fault collision
                if collisions_at_stopped_track_or_active_front or (
                    ego_in_multiple_lanes_or_nondrivable_area and collision_at_lateral
                ):
                    no_at_fault_collision_score = (
                        0.0
                        if tracked_object.tracked_object_type in AGENT_TYPES
                        else 0.5
                    )
                    no_collision_scores[proposal_idx] = np.minimum(
                        no_collision_scores[proposal_idx], no_at_fault_collision_score
                    )
                    self._collision_time_idcs[proposal_idx] = min(
                        time_idx, self._collision_time_idcs[proposal_idx]
                    )

                else:  # 2. no at fault collision
                    proposal_collided_track_ids[proposal_idx].append(token)

        self._multi_metrics[MultiMetricIndex.NO_COLLISION] = no_collision_scores

    def _calculate_ttc(self):
        """
        Re-implementation of nuPlan's time-to-collision metric.
        """

        ttc_scores = np.ones(self._num_proposals, dtype=np.float64)
        temp_collided_track_ids = {
            proposal_idx: copy.deepcopy(self._observation.collided_track_ids)
            for proposal_idx in range(self._num_proposals)
        }

        # calculate TTC for 1s in the future with less temporal resolution.
        future_time_idcs = np.arange(0, 10, 3)
        n_future_steps = len(future_time_idcs)

        # create polygons for each ego position and 1s future projection
        coords_exterior = self._ego_coords.copy()
        coords_exterior[:, :, BBCoordsIndex.CENTER, :] = coords_exterior[
            :, :, BBCoordsIndex.FRONT_LEFT, :
        ]
        coords_exterior_time_steps = np.repeat(
            coords_exterior[:, :, None], n_future_steps, axis=2
        )

        speeds = np.hypot(
            self._states[..., StateIndex.VELOCITY_X],
            self._states[..., StateIndex.VELOCITY_Y],
        )

        dxy_per_s = np.stack(
            [
                np.cos(self._states[..., StateIndex.HEADING]) * speeds,
                np.sin(self._states[..., StateIndex.HEADING]) * speeds,
            ],
            axis=-1,
        )

        for idx, future_time_idx in enumerate(future_time_idcs):
            delta_t = float(future_time_idx) * self._proposal_sampling.interval_length
            coords_exterior_time_steps[:, :, idx] = (
                coords_exterior_time_steps[:, :, idx] + dxy_per_s[:, :, None] * delta_t
            )

        polygons = creation.polygons(coords_exterior_time_steps)

        # check collision for each proposal and projection
        for time_idx in range(self._proposal_sampling.num_poses + 1):
            for step_idx, future_time_idx in enumerate(future_time_idcs):
                current_time_idx = time_idx + future_time_idx
                polygons_at_time_step = polygons[:, time_idx, step_idx]
                intersecting = self._observation[current_time_idx].query(
                    polygons_at_time_step, predicate="intersects"
                )

                if len(intersecting) == 0:
                    continue

                for proposal_idx, geometry_idx in zip(intersecting[0], intersecting[1]):
                    token = self._observation[current_time_idx].tokens[geometry_idx]
                    if (
                        (self._observation.red_light_token in token)
                        or (token in temp_collided_track_ids[proposal_idx])
                        or (speeds[proposal_idx, time_idx] < STOPPED_SPEED_THRESHOLD)
                    ):
                        continue

                    ego_in_multiple_lanes_or_nondrivable_area = (
                        self._ego_areas[
                            proposal_idx, time_idx, EgoAreaIndex.MULTIPLE_LANES
                        ]
                        or self._ego_areas[
                            proposal_idx, time_idx, EgoAreaIndex.NON_DRIVABLE_AREA
                        ]
                    )
                    ego_rear_axle: StateSE2 = StateSE2(
                        *self._states[proposal_idx, time_idx, StateIndex.STATE_SE2]
                    )

                    centroid = self._observation[current_time_idx][token].centroid
                    track_heading = self._observation.unique_objects[
                        token
                    ].box.center.heading
                    track_state = StateSE2(centroid.x, centroid.y, track_heading)
                    if is_agent_ahead(ego_rear_axle, track_state) or (
                        (
                            ego_in_multiple_lanes_or_nondrivable_area
                            or self._map_api.is_in_layer(
                                ego_rear_axle, layer=SemanticMapLayer.INTERSECTION
                            )
                        )
                        and not is_agent_behind(ego_rear_axle, track_state)
                    ):
                        ttc_scores[proposal_idx] = np.minimum(
                            ttc_scores[proposal_idx], 0.0
                        )
                        self._ttc_time_idcs[proposal_idx] = min(
                            time_idx, self._ttc_time_idcs[proposal_idx]
                        )
                    else:
                        temp_collided_track_ids[proposal_idx].append(token)

        self._weighted_metrics[WeightedMetricIndex.TTC] = ttc_scores

    def _calculate_progress(self) -> None:
        """
        Re-implementation of nuPlan's progress metric (non-normalized).
        Calculates progress along the centerline.
        """

        # calculate raw progress in meter
        progress_in_meter = np.zeros(self._num_proposals, dtype=np.float64)
        for proposal_idx in range(self._num_proposals):
            start_point = Point(
                *self._ego_coords[proposal_idx, 0, BBCoordsIndex.CENTER]
            )
            end_point = Point(*self._ego_coords[proposal_idx, -1, BBCoordsIndex.CENTER])
            progress = self._centerline.project([start_point, end_point])
            progress_in_meter[proposal_idx] = progress[1] - progress[0]

        self._progress_raw = progress_in_meter

    def _calculate_is_comfortable(self) -> None:
        """
        Re-implementation of nuPlan's comfortability metric.
        """
        time_point_s: npt.NDArray[np.float64] = (
            np.arange(0, self._proposal_sampling.num_poses + 1).astype(np.float64)
            * self._proposal_sampling.interval_length
        )
        is_comfortable = ego_is_comfortable(self._states, time_point_s)
        self._weighted_metrics[WeightedMetricIndex.COMFORTABLE] = np.all(
            is_comfortable, axis=-1
        )

    def _calculate_drivable_area_compliance(self) -> None:
        """
        Re-implementation of nuPlan's drivable area compliance metric
        """
        drivable_area_compliance_scores = np.ones(self._num_proposals, dtype=np.float64)
        off_road_mask = self._ego_areas[:, :, EgoAreaIndex.NON_DRIVABLE_AREA].any(
            axis=-1
        )
        drivable_area_compliance_scores[off_road_mask] = 0.0
        self._multi_metrics[
            MultiMetricIndex.DRIVABLE_AREA
        ] = drivable_area_compliance_scores

    def _calculate_driving_direction_compliance(self) -> None:
        """
        Re-implementation of nuPlan's driving direction compliance metric
        """
        center_coordinates = self._ego_coords[:, :, BBCoordsIndex.CENTER]
        cum_progress = np.zeros(
            (self._num_proposals, self._proposal_sampling.num_poses + 1),
            dtype=np.float64,
        )
        cum_progress[:, 1:] = (
            (center_coordinates[:, 1:] - center_coordinates[:, :-1]) ** 2.0
        ).sum(axis=-1) ** 0.5

        # mask out progress along the driving direction
        oncoming_traffic_masks = self._ego_areas[:, :, EgoAreaIndex.ONCOMING_TRAFFIC]
        cum_progress[~oncoming_traffic_masks] = 0.0

        driving_direction_compliance_scores = np.ones(
            self._num_proposals, dtype=np.float64
        )

        for proposal_idx in range(self._num_proposals):
            oncoming_traffic_progress, oncoming_traffic_mask = (
                cum_progress[proposal_idx],
                oncoming_traffic_masks[proposal_idx],
            )

            # split progress whenever ego changes traffic direction
            oncoming_progress_splits = np.split(
                oncoming_traffic_progress,
                np.where(np.diff(oncoming_traffic_mask))[0] + 1,
            )

            # sum up progress of splitted intervals
            # Note: splits along the driving direction will have a sum of zero.
            max_oncoming_traffic_progress = max(
                oncoming_progress.sum()
                for oncoming_progress in oncoming_progress_splits
            )

            if max_oncoming_traffic_progress < DRIVING_DIRECTION_COMPLIANCE_THRESHOLD:
                driving_direction_compliance_scores[proposal_idx] = 1.0
            elif max_oncoming_traffic_progress < DRIVING_DIRECTION_VIOLATION_THRESHOLD:
                driving_direction_compliance_scores[proposal_idx] = 0.5
            else:
                driving_direction_compliance_scores[proposal_idx] = 0.0

        self._multi_metrics[
            MultiMetricIndex.DRIVING_DIRECTION
        ] = driving_direction_compliance_scores
