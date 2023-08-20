from typing import Dict, List

import numpy as np
from nuplan.common.actor_state.ego_state import EgoState
from nuplan.common.actor_state.state_representation import StateSE2
from nuplan.common.maps.abstract_map import AbstractMap
from nuplan.common.maps.abstract_map_objects import (
    LaneGraphEdgeMapObject,
    RoadBlockGraphEdgeMapObject,
)
from nuplan.common.maps.maps_datatypes import SemanticMapLayer

from tuplan_garage.planning.simulation.planner.pdm_planner.utils.graph_search.bfs_roadblock import (
    BreadthFirstSearchRoadBlock,
)
from tuplan_garage.planning.simulation.planner.pdm_planner.utils.graph_search.dijkstra import (
    Dijkstra,
)


def normalize_angle(angle):
    return np.arctan2(np.sin(angle), np.cos(angle))


def get_nearest_roadblock_id(
    ego_pose: StateSE2, map_api: AbstractMap, route_roadblock_ids: List[str]
) -> int:

    heading_error_thresh = np.pi / 4
    displacement_error_thresh = 3

    layers = [SemanticMapLayer.ROADBLOCK, SemanticMapLayer.ROADBLOCK_CONNECTOR]
    # get close roadblock candidates
    roadblock_candidates = []

    roadblock_dict = map_api.get_proximal_map_objects(
        point=ego_pose.point, radius=4.0, layers=layers
    )
    roadblock_candidates = (
        roadblock_dict[SemanticMapLayer.ROADBLOCK]
        + roadblock_dict[SemanticMapLayer.ROADBLOCK_CONNECTOR]
    )

    if not roadblock_candidates:
        for layer in layers:
            roadblock_id_, distance = map_api.get_distance_to_nearest_map_object(
                point=ego_pose.point, layer=layer
            )
            roadblock = map_api.get_map_object(roadblock_id_, layer)

            if roadblock:
                roadblock_candidates.append(roadblock)

    on_route_candidates, on_route_candidate_displacement_errors = [], []
    candidates, candidate_displacement_errors = [], []

    roadblock_displacement_errors = []
    roadblock_heading_errors = []

    for idx, roadblock in enumerate(roadblock_candidates):

        lane_displacement_error, lane_heading_error = np.inf, np.inf

        for lane in roadblock.interior_edges:

            lane_discrete_path: List[StateSE2] = lane.baseline_path.discrete_path
            lane_discrete_points = np.array(
                [state.point.array for state in lane_discrete_path], dtype=np.float64
            )
            lane_state_distances = (
                (lane_discrete_points - ego_pose.point.array[None, ...]) ** 2.0
            ).sum(-1) ** 0.5
            argmin_state = np.argmin(lane_state_distances)

            heading_error = np.abs(
                normalize_angle(
                    normalize_angle(
                        lane_discrete_path[argmin_state].heading - ego_pose.heading
                    )
                )
            )
            displacement_error = lane_state_distances[argmin_state]

            if displacement_error < lane_displacement_error:
                lane_heading_error, lane_displacement_error = (
                    heading_error,
                    displacement_error,
                )

            if (
                heading_error < heading_error_thresh
                and displacement_error < displacement_error_thresh
            ):
                if roadblock.id in route_roadblock_ids:
                    on_route_candidates.append(roadblock)
                    on_route_candidate_displacement_errors.append(displacement_error)
                else:
                    candidates.append(roadblock)
                    candidate_displacement_errors.append(displacement_error)

        roadblock_displacement_errors.append(lane_displacement_error)
        roadblock_heading_errors.append(lane_heading_error)

    if on_route_candidates:
        return on_route_candidates[np.argmin(on_route_candidate_displacement_errors)].id
    elif candidates:
        return candidates[np.argmin(candidate_displacement_errors)].id

    return roadblock_candidates[np.argmin(roadblock_displacement_errors)].id


def get_correct_route_roadblock_ids(
    route_roadblock_ids: List[str],
    ego_pose: StateSE2,
    map_api: AbstractMap,
    max_search_depth_backward: int = 15,
    max_search_depth_forward: int = 30,
) -> List[str]:

    # remove repeating roadblock ids
    route_roadblock_ids = list(dict.fromkeys(route_roadblock_ids))

    nearest_block_id = get_nearest_roadblock_id(
        ego_pose=ego_pose, map_api=map_api, route_roadblock_ids=route_roadblock_ids
    )
    correct_route_roadblock_ids: List[str] = list(route_roadblock_ids).copy()

    # fix starting off-route
    if nearest_block_id not in route_roadblock_ids:
        # Backward search
        graph_search = BreadthFirstSearchRoadBlock(
            route_roadblock_ids[0], map_api, forward_search=False
        )
        (path, path_id), path_found = graph_search.search(
            nearest_block_id, max_depth=max_search_depth_backward
        )

        if path_found:
            correct_route_roadblock_ids = path_id + correct_route_roadblock_ids
        else:
            graph_search = BreadthFirstSearchRoadBlock(
                nearest_block_id, map_api, forward_search=True
            )
            (path, path_id), path_found = graph_search.search(
                correct_route_roadblock_ids, max_depth=max_search_depth_forward
            )

            if path_found:
                end_roadblock_idx = np.argmax(
                    np.array(correct_route_roadblock_ids) == path_id[-1]
                )
                correct_route_roadblock_ids = (
                    path_id + correct_route_roadblock_ids[end_roadblock_idx + 1 :]
                )

    # fix interrupted route
    for start_roadblock_id, end_roadblock_id in zip(
        correct_route_roadblock_ids[:-1], correct_route_roadblock_ids[1:]
    ):
        start_roadblock = map_api.get_map_object(
            object_id=start_roadblock_id, layer=SemanticMapLayer.ROADBLOCK
        ) or map_api.get_map_object(
            object_id=start_roadblock_id, layer=SemanticMapLayer.ROADBLOCK_CONNECTOR
        )
        next_incoming_edge_ids = [
            incoming_edge.id for incoming_edge in start_roadblock.incoming_edges
        ]
        is_incoming = start_roadblock in next_incoming_edge_ids

        if is_incoming:
            continue

        graph_search = BreadthFirstSearchRoadBlock(
            start_roadblock_id, map_api, forward_search=True
        )
        (path, path_id), path_found = graph_search.search(
            end_roadblock_id, max_depth=max_search_depth_forward
        )

        if path_found and path and len(path) >= 3:
            # append missing intermediate roadblocks
            idx = correct_route_roadblock_ids.index(start_roadblock_id)
            correct_route_roadblock_ids = (
                correct_route_roadblock_ids[:idx]
                + path_id
                + correct_route_roadblock_ids[idx + 1 :]
            )
    return correct_route_roadblock_ids


def get_closed_loop_route(
    ego_state: EgoState,
    map_api: AbstractMap,
    route_roadblock_ids: List[str],
) -> Dict[str, LaneGraphEdgeMapObject]:

    # initialize all lanes and roadblocks on route
    roadblock_dict: Dict[str, RoadBlockGraphEdgeMapObject] = {}
    lane_dict: Dict[str, LaneGraphEdgeMapObject] = {}

    for roadblock_id in route_roadblock_ids:

        roadblock = map_api.get_map_object(roadblock_id, SemanticMapLayer.ROADBLOCK)
        roadblock = roadblock or map_api.get_map_object(
            roadblock_id, SemanticMapLayer.ROADBLOCK_CONNECTOR
        )
        roadblock_dict[roadblock.id] = roadblock

        for lane in roadblock.interior_edges:
            lane_dict[lane.id] = lane

    # find the nearest lane
    current_lane = get_current_lane(ego_state, lane_dict)

    # path planning
    roadblocks, roadblock_ids = list(roadblock_dict.values()), list(
        roadblock_dict.keys()
    )

    offset = np.argmax(np.array(roadblock_ids) == current_lane.get_roadblock_id())
    roadblock_window = roadblocks[offset : offset + 30]

    graph_search = Dijkstra(current_lane, list(lane_dict.keys()))
    (route_plan, lane_change), path_found = graph_search.search(
        roadblock_window[-1], len(roadblock_window)
    )

    route_lane_dict = {}
    for route_lane in route_plan:
        route_lane_dict[route_lane.id] = route_lane

    return route_lane_dict


def get_current_lane(
    ego_state: EgoState, lane_dict: Dict[str, LaneGraphEdgeMapObject]
) -> LaneGraphEdgeMapObject:
    """
    Get the starting edge based on ego state. If a lane graph object does not contain the ego state then
    the closest one is taken instead.
    :param ego_state: Current ego state.
    :return: The starting LaneGraphEdgeMapObject.
    """

    starting_edge = None
    closest_distance = np.inf

    for edge in lane_dict.values():

        if edge.contains_point(ego_state.center):
            starting_edge = edge
            break

        distance = edge.polygon.distance(ego_state.car_footprint.geometry)

        # In case the ego does not start on a road block
        if distance < closest_distance:
            starting_edge = edge
            closest_distance = distance

    return starting_edge
