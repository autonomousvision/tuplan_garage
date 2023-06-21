from typing import Dict, List, Optional, Tuple

from shapely.geometry import Polygon
import shapely.creation

import numpy as np
from nuplan.common.actor_state.ego_state import EgoState
from nuplan.common.actor_state.tracked_objects import TrackedObject
from nuplan.common.actor_state.tracked_objects_types import TrackedObjectType
from nuplan.common.maps.abstract_map import AbstractMap
from nuplan.common.maps.abstract_map_objects import LaneGraphEdgeMapObject
from nuplan.common.maps.maps_datatypes import (
    TrafficLightStatusData,
    TrafficLightStatusType,
)
from nuplan.planning.simulation.observation.observation_type import Observation
from nuplan.planning.simulation.planner.pdm_planner.observation.pdm_object_manager import (
    PDMObjectManager,
)
from nuplan.planning.simulation.planner.pdm_planner.observation.pdm_observation_utils import (
    get_drivable_area_map,
)
from nuplan.planning.simulation.planner.pdm_planner.observation.pdm_occupancy_map import (
    PDMOccupancyMap,
)
from nuplan.planning.simulation.planner.pdm_planner.utils.pdm_enums import BBCoordsIndex


class PDMObservation:
    """PDM's observation class for forecasted occupancy maps."""

    def __init__(
        self,
        trajectory_samples: int,
        proposal_samples: int,
        sample_interval: float,
        map_radius: float = 50,
        observation_sample_res: int = 2,
    ):
        """
        Constructor of PDMObservation
        :param trajectory_samples: number of trajectory samples
        :param proposal_samples: number of proposal samples
        :param sample_interval: interval between trajectory/proposal samples [s]
        :param map_radius: radius around ego to consider, defaults to 50
        :param observation_sample_res: sample resolution of forecast, defaults to 2
        """

        self._map_radius: float = map_radius

        # observation at least include trajectory horizon
        # or proposal horizon +1s (for TTC metric)
        self._observation_samples: int = (
            proposal_samples + int(1 / sample_interval)
            if proposal_samples + int(1 / sample_interval) > trajectory_samples
            else trajectory_samples
        )
        self._observation_sample_res: int = observation_sample_res
        self._sample_interval: float = sample_interval  # [s]

        # useful things
        self._global_to_local_idcs = [
            idx // observation_sample_res
            for idx in range(self._observation_samples + observation_sample_res)
        ]
        self._collided_track_ids: List[str] = []
        self._red_light_token = "red_light"

        # lazy loaded (during update)
        self._occupancy_maps: Optional[List[PDMOccupancyMap]] = None
        self._drivable_area_map: Optional[PDMOccupancyMap] = None
        self._object_manager: Optional[PDMObjectManager] = None

        self._initialized: bool = False

    def __getitem__(self, time_idx) -> PDMOccupancyMap:
        """
        Retrieves occupancy map for time_idx and adapt temporal resolution.
        :param time_idx: index for future simulation iterations [10Hz]
        :return: occupancy map
        """
        assert self._initialized, "PDMObservation: Has not been updated yet!"
        assert (
            0 <= time_idx < len(self._global_to_local_idcs)
        ), f"PDMObservation: index {time_idx} out of range!"

        local_idx = self._global_to_local_idcs[time_idx]
        return self._occupancy_maps[local_idx]

    @property
    def drivable_area_map(self) -> PDMOccupancyMap:
        """
        Getter for drivable area map.
        :return: occupancy map of lanes and carpark around ego
        """
        assert self._initialized, "PDMObservation: Has not been updated yet!"
        return self._drivable_area_map

    @property
    def collided_track_ids(self) -> List[str]:
        """
        Getter for past collided track tokens.
        :return: list of tokens
        """
        assert self._initialized, "PDMObservation: Has not been updated yet!"
        return self._collided_track_ids

    @property
    def red_light_token(self) -> str:
        """
        Getter for red light token indicator
        :return: _description_
        """
        return self._red_light_token

    @property
    def unique_objects(self) -> Dict[str, TrackedObject]:
        """
        Getter for unique tracked objects
        :return: dictionary of tokens, tracked objects
        """
        assert self._initialized, "PDMObservation: Has not been updated yet!"
        return self._object_manager.unique_objects

    def update(
        self,
        ego_state: EgoState,
        observation: Observation,
        traffic_light_data: List[TrafficLightStatusData],
        route_lane_dict: Dict[str, LaneGraphEdgeMapObject],
        map_api: AbstractMap,
    ) -> None:
        """
        Update & lazy loads information  of PDMObservation.
        :param ego_state: state of ego vehicle
        :param observation: input observation of nuPlan
        :param traffic_light_data: list of traffic light states
        :param route_lane_dict: dictionary of on-route lanes
        :param map_api: map object of nuPlan
        """

        self._occupancy_maps: List[PDMOccupancyMap] = []
        self._drivable_area_map = get_drivable_area_map(map_api, ego_state, self._map_radius)
        self._object_manager = self._get_object_manager(ego_state, observation)

        traffic_light_tokens, traffic_light_polygons = self._get_traffic_light_geometries(
            traffic_light_data, route_lane_dict
        )

        (
            static_object_tokens,
            static_object_coords,
            dynamic_object_tokens,
            dynamic_object_coords,
            dynamic_object_dxy,
        ) = self._object_manager.get_nearest_objects(ego_state.center.point)

        has_static_object, has_dynamic_object = (
            len(static_object_tokens) > 0,
            len(dynamic_object_tokens) > 0,
        )

        if has_static_object and static_object_coords.ndim == 1:
            static_object_coords = static_object_coords[None, ...]

        if has_dynamic_object and dynamic_object_coords.ndim == 1:
            dynamic_object_coords = dynamic_object_coords[None, ...]
            dynamic_object_dxy = dynamic_object_dxy[None, ...]

        if has_static_object:
            static_object_coords[..., BBCoordsIndex.CENTER, :] = static_object_coords[
                ..., BBCoordsIndex.FRONT_LEFT, :
            ]
            static_object_polygons = shapely.creation.polygons(static_object_coords)

        else:
            static_object_polygons = np.array([], dtype=np.object_)

        if has_dynamic_object:
            dynamic_object_coords[..., BBCoordsIndex.CENTER, :] = dynamic_object_coords[
                ..., BBCoordsIndex.FRONT_LEFT, :
            ]
        else:
            dynamic_object_polygons = np.array([], dtype=np.object_)
            dynamic_object_tokens = []

        traffic_light_polygons = np.array(traffic_light_polygons, dtype=np.object_)

        for sample in np.arange(
            0,
            self._observation_samples + self._observation_sample_res,
            self._observation_sample_res,
        ):
            if has_dynamic_object:
                delta_t = float(sample) * self._sample_interval
                dynamic_object_coords_t = (
                    dynamic_object_coords + delta_t * dynamic_object_dxy[:, None]
                )
                dynamic_object_polygons = shapely.creation.polygons(dynamic_object_coords_t)

            all_polygons = np.concatenate(
                [static_object_polygons, dynamic_object_polygons, traffic_light_polygons], axis=0
            )

            occupancy_map = PDMOccupancyMap(
                static_object_tokens + dynamic_object_tokens + traffic_light_tokens, all_polygons
            )
            self._occupancy_maps.append(occupancy_map)

        # save collided objects to ignore in the future
        ego_polygon: Polygon = ego_state.car_footprint.geometry
        intersecting_obstacles = self._occupancy_maps[0].intersects(ego_polygon)
        new_collided_track_ids = []

        for intersecting_obstacle in intersecting_obstacles:
            if self._red_light_token in intersecting_obstacle:
                within = ego_polygon.within(self._occupancy_maps[0][intersecting_obstacle])
                if not within:
                    continue
            new_collided_track_ids.append(intersecting_obstacle)

        self._collided_track_ids = self._collided_track_ids + new_collided_track_ids
        self._initialized = True

    def _get_object_manager(
        self, ego_state: EgoState, observation: Observation
    ) -> PDMObjectManager:
        """
        Creates object manager class, but adding valid tracked objects.
        :param ego_state: state of ego-vehicle
        :param observation: input observation of nuPlan
        :return: PDMObjectManager class
        """
        object_manager = PDMObjectManager()

        for object in observation.tracked_objects:
            if (
                (object.tracked_object_type == TrackedObjectType.EGO)
                or (
                    self._map_radius
                    and ego_state.center.distance_to(object.center) > self._map_radius
                )
                or (object.track_token in self._collided_track_ids)
            ):
                continue

            object_manager.add_object(object)

        return object_manager

    def _get_traffic_light_geometries(
        self,
        traffic_light_data: List[TrafficLightStatusData],
        route_lane_dict: Dict[str, LaneGraphEdgeMapObject],
    ) -> Tuple[List[str], List[Polygon]]:
        """
        Collects red traffic lights along ego's route.
        :param traffic_light_data: list of traffic light states
        :param route_lane_dict: dictonary of on-route lanes
        :return: tuple of tokens and polygons of red traffic lights
        """
        traffic_light_tokens, traffic_light_polygons = [], []

        for data in traffic_light_data:
            lane_connector_id = str(data.lane_connector_id)

            if (data.status == TrafficLightStatusType.RED) and (
                lane_connector_id in route_lane_dict.keys()
            ):
                lane_connector = route_lane_dict[lane_connector_id]
                traffic_light_tokens.append(f"{self._red_light_token}_{lane_connector_id}")
                traffic_light_polygons.append(lane_connector.polygon)

        return traffic_light_tokens, traffic_light_polygons
