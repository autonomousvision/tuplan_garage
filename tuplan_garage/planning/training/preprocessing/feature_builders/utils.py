from typing import List, Dict, Tuple, Union
import numpy as np
import numpy.typing as npt
from shapely.geometry import Point
import warnings
import logging

from nuplan.common.actor_state.ego_state import EgoState
from nuplan.common.actor_state.state_representation import StateSE2
from nuplan.common.maps.abstract_map_objects import LaneGraphEdgeMapObject, RoadBlockGraphEdgeMapObject
from nuplan.planning.training.preprocessing.features.trajectory_utils import _convert_absolute_to_relative_states
from tuplan_garage.planning.simulation.planner.pdm_planner.utils.graph_search.dijkstra import Dijkstra
from tuplan_garage.planning.simulation.planner.pdm_planner.utils.pdm_path import PDMPath
from tuplan_garage.planning.simulation.planner.pdm_planner.utils.pdm_geometry_utils import normalize_angle
from tuplan_garage.planning.simulation.planner.pdm_planner.observation.pdm_occupancy_map import PDMOccupancyMap
from tuplan_garage.planning.training.preprocessing.features.llm_features import AvailableCenterlines, Centerline

logger = logging.getLogger(__name__)

class CenterlineExtractor():
    def __init__(
        self,
        resolution: float,
        max_length: float,
        distance_behind: float=1.0
    ):
        self.resolution = resolution
        self.max_length = max_length
        self.distance_behind = distance_behind

    def get_centerlines(
        self,
        ego_state: EgoState,
        route_lane_dict: Dict[str, LaneGraphEdgeMapObject],
        route_roadblock_dict: Dict[str, RoadBlockGraphEdgeMapObject],
        drivable_area_map: PDMOccupancyMap,
        only_on_route: bool,
    ) -> Dict[str, List[StateSE2]]:
        available_centerlines = self.get_available_centerlines(
            ego_state=ego_state,
            route_lane_dict=route_lane_dict,
            route_roadblock_dict=route_roadblock_dict,
            drivable_area_map=drivable_area_map,
        )
        lanes: Dict[str, Centerline] = {
            k:getattr(available_centerlines,k) 
            for k in ["current","left","right"]
            if getattr(available_centerlines,k) is not None
        }
        if only_on_route:
            min_lane_changes = min([l.required_lane_changes_to_route for l in lanes.values()])
            on_route_lanes = [
                k for k in lanes.keys()
                if (
                    lanes[k].required_lane_changes_to_route == 0
                    or
                    lanes[k].required_lane_changes_to_route == min_lane_changes
                )
            ]
            lanes = {k:v for k,v in lanes.items() if k in on_route_lanes}
        
        return {
            k: [
                StateSE2(x=pose[0], y=pose[1], heading=pose[2])
                for pose in v.poses
            ]
            for k,v in lanes.items()
        }
    
    def get_available_centerlines(
        self,
        ego_state: EgoState,
        route_lane_dict: Dict[str, LaneGraphEdgeMapObject],
        route_roadblock_dict: Dict[str, RoadBlockGraphEdgeMapObject],
        drivable_area_map: PDMOccupancyMap,
    ) -> AvailableCenterlines:
        '''
        :returns Dict containing a the current, left and right centerline
            each centerline is given by a list of poses, i.e. [x,y,heading]
        '''

        # assign ego to a lane
        current_lane = get_starting_lane(
            ego_state=ego_state,
            route_lane_dict=route_lane_dict,
            drivable_area_map=drivable_area_map,
        )

        adjacent_lanes = get_adjacent_lanes(current_lane)

        min_lane_changes_to_route = get_lane_changes_to_route(
            current_lane=current_lane,
            route_roadblock_dict=route_roadblock_dict,
            route_lane_dict=route_lane_dict
        )

        # dijkstra search to identify sequence of centerlines
        lanes = dict()
        for key, start_lane in zip(
            ["current","left","right"],
            [current_lane, *adjacent_lanes]
        ):
            if start_lane is None:
                lanes[key] = None
            else:
                states_along_path = get_discrete_centerline(
                    current_lane=start_lane,
                    route_roadblock_dict=route_roadblock_dict,
                    route_lane_dict=route_lane_dict,
                )

                required_lane_changes = min_lane_changes_to_route.get(start_lane.id, np.inf)

                # transform poses to local coordinate frame
                local_states_along_path = _convert_absolute_to_relative_states(
                    origin_absolute_state=ego_state.center,
                    absolute_states=states_along_path,
                )

                # sample lanes to desired length and resolution
                lane_states = sample_discrete_path(
                    states_along_path=local_states_along_path,
                    resolution=self.resolution,
                    max_length=self.max_length,
                    distance_behind=self.distance_behind,
                )
                lane_poses = [
                    np.array([state.x, state.y, state.heading])
                    for state in lane_states
                ]
                if start_lane.speed_limit_mps is not None:
                    speed_limit = start_lane.speed_limit_mps
                else:
                    # fallback speed limit
                    speed_limit = 15.0
                if len(start_lane.outgoing_edges) == 1:
                    successor_id = start_lane.outgoing_edges[0].id
                else:
                    successor_id = None
                lanes[key] = Centerline(
                    id=start_lane.id,
                    successor_id=successor_id,
                    poses=lane_poses,
                    required_lane_changes_to_route=required_lane_changes,
                    speed_limit=speed_limit,
                )

        return AvailableCenterlines(
            left=lanes["left"],
            right=lanes["right"],
            current=lanes["current"],
        )

def get_lane_changes_to_route(
    current_lane: LaneGraphEdgeMapObject,
    route_roadblock_dict: Dict[str, RoadBlockGraphEdgeMapObject],
    route_lane_dict: Dict[str, LaneGraphEdgeMapObject],
) -> Dict[str, int]:
    def _lane_is_on_route(lane: LaneGraphEdgeMapObject):
        return get_route_plan(
            current_lane=lane,
            route_roadblock_dict=route_roadblock_dict,
            route_lane_dict=route_lane_dict,
        )[1]

    
    lane_changes_for_lanes_in_roadbloack = {
        l.id:0
        for l in current_lane.parent.interior_edges 
        if _lane_is_on_route(l)
    }

    queue = [(l,0) for l in current_lane.parent.interior_edges if l.id in lane_changes_for_lanes_in_roadbloack.keys()]
    while queue:
        lane, lane_changes = queue.pop(0)
        if lane.id not in lane_changes_for_lanes_in_roadbloack.keys():
            lane_changes_for_lanes_in_roadbloack.update({lane.id: lane_changes})
        # get adjacent lanes, append if not in visited and not in queue
        for adj_lane in get_adjacent_lanes(lane):
            if (
                adj_lane is not None and
                adj_lane.id not in lane_changes_for_lanes_in_roadbloack.keys() and
                adj_lane.id not in [item[0] for item in queue]
            ):
                queue.append((adj_lane, lane_changes+1))

    return lane_changes_for_lanes_in_roadbloack


def get_adjacent_lanes(
    current_lane: LaneGraphEdgeMapObject
) -> Tuple[LaneGraphEdgeMapObject]:

    def _filter_candidates(
        candidate_lanes: List[LaneGraphEdgeMapObject],
        side: str
    ) -> Union[LaneGraphEdgeMapObject, None]:
        if len(candidate_lanes) == 0:
            return None        
        # We know that the candidates start adjacent to each other as the incoming lanes are adjacent
        # Decide which one stays beside the current lane by comparing the distance of the boundaries at the end of the roadblock
        if side == "right":
            fde = [ 
                l.left_boundary.discrete_path[-1].distance_to(current_lane.right_boundary.discrete_path[-1])
                for l in candidate_lanes
            ]
        else: # side == "left"
            fde = [ 
                l.right_boundary.discrete_path[-1].distance_to(current_lane.left_boundary.discrete_path[-1])
                for l in candidate_lanes
            ]
        if min(fde) > 0.1:
            return None
        else:
            return candidate_lanes[fde.index(min(fde))]

    def _get_candidates(
        current_lane: LaneGraphEdgeMapObject,
        side:str
    ) -> List[LaneGraphEdgeMapObject]:
        idx = 0 if side=="left" else 1
        candidates: List[LaneGraphEdgeMapObject] = []
        if current_lane.adjacent_edges[idx] is not None:
            candidates.append(current_lane.adjacent_edges[idx])
        previous_lanes_of_adjacent = current_lane.incoming_edges + [
            lane.adjacent_edges[idx] 
            for lane in current_lane.incoming_edges
            if lane.adjacent_edges[idx] is not None
        ]
        candidates.extend(
            [l for p in previous_lanes_of_adjacent for l in p.outgoing_edges]
        )
        return list(set([c for c in candidates if c.id != current_lane.id]))

    # Search for adjacent lanes. An adjacent lane can be obtained
    # 1) by current_lane.adjacent_edges
    # 2) successor of adjacent lane of predecessor lane
    # 3) sucessor of predecessor which is not the current lane

    # left neighbor
    candidates_left = _get_candidates(current_lane, "left")
    left = _filter_candidates(candidate_lanes=candidates_left, side="left")

    # right neighbor
    candidates_right = _get_candidates(current_lane, "right")
    right = _filter_candidates(candidate_lanes=candidates_right, side="right")

    return (left, right)

def sample_discrete_path(
    states_along_path: List[StateSE2],
    resolution: float,
    max_length: float,
    distance_behind: float,
) -> List[StateSE2]:

    path = PDMPath(discrete_path=states_along_path)
    warnings.filterwarnings("ignore", message="invalid value encountered in line_locate_point", category=RuntimeWarning)
    ego_progress_along_path = path.project([Point(0.0,0.0)])[0]

    end_progress = min(max_length+ego_progress_along_path, path.length)
    start_progress = max(ego_progress_along_path-distance_behind, 0)
    sampling_distances = np.arange(start_progress, end_progress, resolution)

    return path.interpolate(
        distances=sampling_distances,
        as_array=False,
    ).tolist()

def get_route_plan(
    current_lane: LaneGraphEdgeMapObject,
    route_roadblock_dict: Dict[str, RoadBlockGraphEdgeMapObject],
    route_lane_dict: Dict[str, LaneGraphEdgeMapObject],
    search_depth: int = 30,
) -> Tuple[List[LaneGraphEdgeMapObject], bool]:
    """
    Applies a Dijkstra search on the lane-graph to retrieve discrete centerline.
    :param current_lane: _description_
    :param search_depth: depth of search (for runtime), defaults to 30
    :return:
            - A route starting from the given start edge
            - A bool indicating if the route is successfully found. Successful means that there exists a path
              from the start edge to an edge contained in the end roadblock.
              If unsuccessful the shortest deepest path is returned.
    """

    roadblocks = list(route_roadblock_dict.values())
    roadblock_ids = list(route_roadblock_dict.keys())

    # find current roadblock index
    start_idx = np.argmax(np.array(roadblock_ids) == current_lane.get_roadblock_id())
    roadblock_window = roadblocks[start_idx : start_idx + search_depth]

    graph_search = Dijkstra(current_lane, list(route_lane_dict.keys()))
    return graph_search.search(roadblock_window[-1])

def get_discrete_centerline(
    current_lane: LaneGraphEdgeMapObject,
    route_roadblock_dict: Dict[str, RoadBlockGraphEdgeMapObject],
    route_lane_dict: Dict[str, LaneGraphEdgeMapObject],
    search_depth: int = 30,
) -> List[StateSE2]:
    """
    Applies a Dijkstra search on the lane-graph to retrieve discrete centerline.
    :param current_lane: _description_
    :param search_depth: depth of search (for runtime), defaults to 30
    :return: list of discrete states on centerline (x,y,θ)
    """
    route_plan, _ = get_route_plan(
        current_lane=current_lane,
        route_roadblock_dict=route_roadblock_dict,
        route_lane_dict=route_lane_dict,
        search_depth=search_depth,
    )
    if len(route_plan) == 1 and len(route_plan[0].outgoing_edges) > 0:
        # no baseline-path beyond the current roadblock was found
        # as a fallback, we add the successor lane to make sure the
        # route_centerline is long enough
        route_plan = route_plan + [route_plan[0].outgoing_edges[0]]

    centerline_discrete_path: List[StateSE2] = []
    for lane in route_plan:
        centerline_discrete_path.extend(lane.baseline_path.discrete_path)

    return centerline_discrete_path

def get_starting_lane(
    ego_state: EgoState,
    route_lane_dict: Dict[str, LaneGraphEdgeMapObject],
    drivable_area_map: PDMOccupancyMap,
) -> LaneGraphEdgeMapObject:
    """
    Returns the most suitable starting lane, in ego's vicinity.
    :param ego_state: state of ego-vehicle
    :return: lane object (on-route)
    """
    starting_lane: LaneGraphEdgeMapObject = None
    on_route_lanes, heading_error = get_intersecting_lanes(
        ego_state=ego_state,
        route_lane_dict=route_lane_dict,
        drivable_area_map=drivable_area_map,
    )

    if on_route_lanes:
        # 1. Option: find lanes from lane occupancy-map
        # select lane with lowest heading error
        starting_lane = on_route_lanes[np.argmin(np.abs(heading_error))]
        return starting_lane

    else:
        # 2. Option: find any intersecting or close lane on-route
        closest_distance = np.inf
        for edge in route_lane_dict.values():
            if edge.contains_point(ego_state.center):
                starting_lane = edge
                break

            distance = edge.polygon.distance(ego_state.car_footprint.geometry)
            if distance < closest_distance:
                starting_lane = edge
                closest_distance = distance

    return starting_lane

def get_intersecting_lanes(
    ego_state: EgoState,
    route_lane_dict: Dict[str, LaneGraphEdgeMapObject],
    drivable_area_map: PDMOccupancyMap,
) -> Tuple[List[LaneGraphEdgeMapObject], List[float]]:
    """
    Returns on-route lanes and heading errors where ego-vehicle intersects.
    :param ego_state: state of ego-vehicle
    :return: tuple of lists with lane objects and heading errors [rad].
    """

    ego_position_array: npt.NDArray[np.float64] = ego_state.rear_axle.array
    ego_rear_axle_point: Point = Point(*ego_position_array)
    ego_heading: float = ego_state.rear_axle.heading

    intersecting_lanes = drivable_area_map.intersects(ego_rear_axle_point)

    on_route_lanes, on_route_heading_errors = [], []
    for lane_id in intersecting_lanes:
        if lane_id in route_lane_dict.keys():
            # collect baseline path as array
            lane_object = route_lane_dict[lane_id]
            lane_discrete_path: List[StateSE2] = lane_object.baseline_path.discrete_path
            lane_state_se2_array = np.array(
                [state.array for state in lane_discrete_path], dtype=np.float64
            )
            # calculate nearest state on baseline
            lane_distances = (ego_position_array[None, ...] - lane_state_se2_array) ** 2
            lane_distances = lane_distances.sum(axis=-1) ** 0.5

            # calculate heading error
            heading_error = lane_discrete_path[np.argmin(lane_distances)].heading - ego_heading
            heading_error = np.abs(normalize_angle(heading_error))

            # add lane to candidates
            on_route_lanes.append(lane_object)
            on_route_heading_errors.append(heading_error)

    return on_route_lanes, on_route_heading_errors