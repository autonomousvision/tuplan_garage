from typing import Dict, List, Tuple

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import scipy
import torch
from nuplan.common.actor_state.state_representation import Point2D
from nuplan.common.actor_state.tracked_objects_types import TrackedObjectType
from nuplan.common.actor_state.vehicle_parameters import get_pacifica_parameters
from nuplan.common.geometry.torch_geometry import (
    global_state_se2_tensor_to_local,
    vector_set_coordinates_to_local_frame,
)
from nuplan.common.maps.abstract_map import AbstractMap
from nuplan.common.maps.maps_datatypes import TrafficLightStatusData
from nuplan.planning.simulation.observation.idm.utils import path_to_linestring
from nuplan.planning.training.preprocessing.feature_builders.vector_builder_utils import (
    LaneSegmentTrafficLightData,
    MapObjectPolylines,
    VectorFeatureLayer,
    VectorFeatureLayerMapping,
    get_lane_polylines,
    get_map_object_polygons,
    get_route_lane_polylines_from_roadblock_ids,
    get_traffic_light_encoding,
)
from nuplan.planning.training.preprocessing.features.agents import Agents
from nuplan.planning.training.preprocessing.utils.agents_preprocessing import (
    AgentInternalIndex,
    EgoInternalIndex,
    compute_yaw_rate_from_state_tensors,
    filter_agents_tensor,
    pack_agents_tensor,
    pad_agent_states,
)
from nuplan.planning.training.preprocessing.utils.vector_preprocessing import (
    interpolate_points,
)
from shapely import LineString, Point
from shapely.geometry.base import CAP_STYLE

from tuplan_garage.planning.simulation.planner.dtpp_planner.bezier_path import (
    calc_4points_bezier_path,
)
from tuplan_garage.planning.simulation.planner.dtpp_planner.common_utils import (
    MAX_LEN,
    WIDTH,
)
from tuplan_garage.planning.simulation.planner.dtpp_planner.path_planner import (
    calc_spline_course,
)
from tuplan_garage.planning.simulation.planner.dtpp_planner.planner_utils import (
    calculate_path_heading,
    check_path,
    wrap_to_pi,
)


# Network input features
def _extract_agent_tensor(tracked_objects, track_token_ids, object_types):
    """
    Extracts the relevant data from the agents present in a past detection into a tensor.
    Only objects of specified type will be transformed. Others will be ignored.
    The output is a tensor as described in AgentInternalIndex
    :param tracked_objects: The tracked objects to turn into a tensor.
    :track_token_ids: A dictionary used to assign track tokens to integer IDs.
    :object_type: TrackedObjectType to filter agents by.
    :return: The generated tensor and the updated track_token_ids dict.
    """
    agents = tracked_objects.get_tracked_objects_of_types(object_types)
    agent_types = []
    output = torch.zeros((len(agents), AgentInternalIndex.dim()), dtype=torch.float32)
    max_agent_id = len(track_token_ids)

    for idx, agent in enumerate(agents):
        if agent.track_token not in track_token_ids:
            track_token_ids[agent.track_token] = max_agent_id
            max_agent_id += 1
        track_token_int = track_token_ids[agent.track_token]

        output[idx, AgentInternalIndex.track_token()] = float(track_token_int)
        output[idx, AgentInternalIndex.vx()] = agent.velocity.x
        output[idx, AgentInternalIndex.vy()] = agent.velocity.y
        output[idx, AgentInternalIndex.heading()] = agent.center.heading
        output[idx, AgentInternalIndex.width()] = agent.box.width
        output[idx, AgentInternalIndex.length()] = agent.box.length
        output[idx, AgentInternalIndex.x()] = agent.center.x
        output[idx, AgentInternalIndex.y()] = agent.center.y
        agent_types.append(agent.tracked_object_type)

    return output, track_token_ids, agent_types


def sampled_tracked_objects_to_tensor_list(past_tracked_objects):
    """
    Tensorizes the agents features from the provided past detections.
    For N past detections, output is a list of length N, with each tensor as described in `_extract_agent_tensor()`.
    :param past_tracked_objects: The tracked objects to tensorize.
    :return: The tensorized objects.
    """
    object_types = [
        TrackedObjectType.VEHICLE,
        TrackedObjectType.PEDESTRIAN,
        TrackedObjectType.BICYCLE,
    ]
    output = []
    output_types = []
    track_token_ids = {}

    for i in range(len(past_tracked_objects)):
        tensorized, track_token_ids, agent_types = _extract_agent_tensor(
            past_tracked_objects[i], track_token_ids, object_types
        )
        output.append(tensorized)
        output_types.append(agent_types)

    return output, output_types


def global_velocity_to_local(velocity, anchor_heading):
    velocity_x = velocity[:, 0] * torch.cos(anchor_heading) + velocity[
        :, 1
    ] * torch.sin(anchor_heading)
    velocity_y = velocity[:, 1] * torch.cos(anchor_heading) - velocity[
        :, 0
    ] * torch.sin(anchor_heading)

    return torch.stack([velocity_x, velocity_y], dim=-1)


def convert_absolute_quantities_to_relative(agent_state, ego_state, agent_type="ego"):
    """
    Converts the agent' poses and relative velocities from absolute to ego-relative coordinates.
    :param agent_state: The agent states to convert, in the AgentInternalIndex schema.
    :param ego_state: The ego state to convert, in the EgoInternalIndex schema.
    :return: The converted states, in AgentInternalIndex schema.
    """
    ego_pose = torch.tensor(
        [
            float(ego_state[EgoInternalIndex.x()].item()),
            float(ego_state[EgoInternalIndex.y()].item()),
            float(ego_state[EgoInternalIndex.heading()].item()),
        ],
        dtype=torch.float64,
    )

    if agent_type == "ego":
        agent_global_poses = agent_state[
            :, [EgoInternalIndex.x(), EgoInternalIndex.y(), EgoInternalIndex.heading()]
        ]
        transformed_poses = global_state_se2_tensor_to_local(
            agent_global_poses, ego_pose, precision=torch.float64
        )
        agent_state[:, EgoInternalIndex.x()] = transformed_poses[:, 0].float()
        agent_state[:, EgoInternalIndex.y()] = transformed_poses[:, 1].float()
        agent_state[:, EgoInternalIndex.heading()] = transformed_poses[:, 2].float()
    else:
        agent_global_poses = agent_state[
            :,
            [
                AgentInternalIndex.x(),
                AgentInternalIndex.y(),
                AgentInternalIndex.heading(),
            ],
        ]
        agent_global_velocities = agent_state[
            :, [AgentInternalIndex.vx(), AgentInternalIndex.vy()]
        ]
        transformed_poses = global_state_se2_tensor_to_local(
            agent_global_poses, ego_pose, precision=torch.float64
        )
        transformed_velocities = global_velocity_to_local(
            agent_global_velocities, ego_pose[-1]
        )
        agent_state[:, AgentInternalIndex.x()] = transformed_poses[:, 0].float()
        agent_state[:, AgentInternalIndex.y()] = transformed_poses[:, 1].float()
        agent_state[:, AgentInternalIndex.heading()] = transformed_poses[:, 2].float()
        agent_state[:, AgentInternalIndex.vx()] = transformed_velocities[:, 0].float()
        agent_state[:, AgentInternalIndex.vy()] = transformed_velocities[:, 1].float()

    return agent_state


def agent_past_process(
    past_ego_states,
    past_time_stamps,
    past_tracked_objects,
    tracked_objects_types,
    num_agents,
):
    """
    This function process the data from the raw agent data.
    :param past_ego_states: The input tensor data of the ego past.
    :param past_time_stamps: The input tensor data of the past timestamps.
    :param past_time_stamps: The input tensor data of other agents in the past.
    :return: ego_agent_array, other_agents_array.
    """
    agents_states_dim = Agents.agents_states_dim()
    ego_history = past_ego_states
    time_stamps = past_time_stamps
    agents = past_tracked_objects

    anchor_ego_state = ego_history[-1, :].squeeze().clone()
    ego_tensor = convert_absolute_quantities_to_relative(ego_history, anchor_ego_state)
    agent_history = filter_agents_tensor(agents, reverse=True)
    agent_types = tracked_objects_types[-1]

    """
    Model input feature representing the present and past states of the ego and agents, including:
    ego: <np.ndarray: num_frames, 7>
        The num_frames includes both present and past frames.
        The last dimension is the ego pose (x, y, heading) velocities (vx, vy) acceleration (ax, ay) at time t.
    agents: <np.ndarray: num_frames, num_agents, 8>
        Agent features indexed by agent feature type.
        The num_frames includes both present and past frames.
        The num_agents is padded to fit the largest number of agents across all frames.
        The last dimension is the agent pose (x, y, heading) velocities (vx, vy, yaw rate) and size (length, width) at time t.
    """

    if agent_history[-1].shape[0] == 0:
        # Return zero tensor when there are no agents in the scene
        agents_tensor = torch.zeros((len(agent_history), 0, agents_states_dim)).float()
    else:
        local_coords_agent_states = []
        padded_agent_states = pad_agent_states(agent_history, reverse=True)

        for agent_state in padded_agent_states:
            local_coords_agent_states.append(
                convert_absolute_quantities_to_relative(
                    agent_state, anchor_ego_state, "agent"
                )
            )

        # Calculate yaw rate
        yaw_rate_horizon = compute_yaw_rate_from_state_tensors(
            padded_agent_states, time_stamps
        )

        agents_tensor = pack_agents_tensor(local_coords_agent_states, yaw_rate_horizon)

    """
    Post-process the agents tensor to select a fixed number of agents closest to the ego vehicle.
    agents: <np.ndarray: num_agents, num_frames, 11>]].
        Agent type is one-hot encoded: [1, 0, 0] vehicle, [0, 1, 0] pedestrain, [0, 0, 1] bicycle
            and added to the feature of the agent
        The num_agents is padded or trimmed to fit the predefined number of agents across.
        The num_frames includes both present and past frames.
    """
    agents = np.zeros(
        shape=(num_agents, agents_tensor.shape[0], agents_tensor.shape[-1] + 3),
        dtype=np.float32,
    )

    # sort agents according to distance to ego
    distance_to_ego = torch.norm(agents_tensor[-1, :, :2], dim=-1)
    indices = list(torch.argsort(distance_to_ego).numpy())[:num_agents]

    # fill agent features into the array
    for i, j in enumerate(indices):
        agents[i, :, : agents_tensor.shape[-1]] = agents_tensor[
            :, j, : agents_tensor.shape[-1]
        ].numpy()
        if agent_types[j] == TrackedObjectType.VEHICLE:
            agents[i, :, agents_tensor.shape[-1] :] = [1, 0, 0]
        elif agent_types[j] == TrackedObjectType.PEDESTRIAN:
            agents[i, :, agents_tensor.shape[-1] :] = [0, 1, 0]
        else:
            agents[i, :, agents_tensor.shape[-1] :] = [0, 0, 1]

    return ego_tensor.numpy().astype(np.float32), agents, indices


def agent_future_process(
    anchor_ego_state, future_tracked_objects, num_agents, agent_index
):
    anchor_ego_state = torch.tensor(
        [
            anchor_ego_state.rear_axle.x,
            anchor_ego_state.rear_axle.y,
            anchor_ego_state.rear_axle.heading,
            anchor_ego_state.dynamic_car_state.rear_axle_velocity_2d.x,
            anchor_ego_state.dynamic_car_state.rear_axle_velocity_2d.y,
            anchor_ego_state.dynamic_car_state.rear_axle_acceleration_2d.x,
            anchor_ego_state.dynamic_car_state.rear_axle_acceleration_2d.y,
        ]
    )

    agent_future = filter_agents_tensor(future_tracked_objects)
    local_coords_agent_states = []
    for agent_state in agent_future:
        local_coords_agent_states.append(
            convert_absolute_quantities_to_relative(
                agent_state, anchor_ego_state, "agent"
            )
        )
    padded_agent_states = pad_agent_states_with_zeros(local_coords_agent_states)

    # fill agent features into the array
    agent_futures = np.zeros(
        shape=(num_agents, padded_agent_states.shape[0] - 1, 3), dtype=np.float32
    )
    for i, j in enumerate(agent_index):
        agent_futures[i] = padded_agent_states[
            1:,
            j,
            [
                AgentInternalIndex.x(),
                AgentInternalIndex.y(),
                AgentInternalIndex.heading(),
            ],
        ].numpy()

    return agent_futures


def pad_agent_states_with_zeros(agent_trajectories):
    key_frame = agent_trajectories[0]
    track_id_idx = AgentInternalIndex.track_token()

    pad_agent_trajectories = torch.zeros(
        (len(agent_trajectories), key_frame.shape[0], key_frame.shape[1]),
        dtype=torch.float32,
    )
    for idx in range(len(agent_trajectories)):
        frame = agent_trajectories[idx]
        mapped_rows = frame[:, track_id_idx]

        for row_idx in range(key_frame.shape[0]):
            if row_idx in mapped_rows:
                pad_agent_trajectories[idx, row_idx] = frame[
                    frame[:, track_id_idx] == row_idx
                ]

    return pad_agent_trajectories


def convert_feature_layer_to_fixed_size(
    ego_pose,
    feature_coords,
    feature_tl_data,
    max_elements,
    max_points,
    traffic_light_encoding_dim,
    interpolation,
):
    """
    Converts variable sized map features to fixed size tensors. Map elements are padded/trimmed to max_elements size.
        Points per feature are interpolated to maintain max_points size.
    :param ego_pose: the current pose of the ego vehicle.
    :param feature_coords: Vector set of coordinates for collection of elements in map layer.
        [num_elements, num_points_in_element (variable size), 2]
    :param feature_tl_data: Optional traffic light status corresponding to map elements at given index in coords.
        [num_elements, traffic_light_encoding_dim (4)]
    :param max_elements: Number of elements to pad/trim to.
    :param max_points: Number of points to interpolate or pad/trim to.
    :param traffic_light_encoding_dim: Dimensionality of traffic light data.
    :param interpolation: Optional interpolation mode for maintaining fixed number of points per element.
        None indicates trimming and zero-padding to take place in lieu of interpolation. Interpolation options: 'linear' and 'area'.
    :return
        coords_tensor: The converted coords tensor.
        tl_data_tensor: The converted traffic light data tensor (if available).
        avails_tensor: Availabilities tensor identifying real vs zero-padded data in coords_tensor and tl_data_tensor.
    :raise ValueError: If coordinates and traffic light data size do not match.
    """
    if feature_tl_data is not None and len(feature_coords) != len(feature_tl_data):
        raise ValueError(
            f"Size between feature coords and traffic light data inconsistent: {len(feature_coords)}, {len(feature_tl_data)}"
        )

    # trim or zero-pad elements to maintain fixed size
    coords_tensor = torch.zeros((max_elements, max_points, 2), dtype=torch.float32)
    avails_tensor = torch.zeros((max_elements, max_points), dtype=torch.bool)
    tl_data_tensor = (
        torch.zeros(
            (max_elements, max_points, traffic_light_encoding_dim), dtype=torch.float32
        )
        if feature_tl_data is not None
        else None
    )

    # get elements according to the mean distance to the ego pose
    mapping = {}
    for i, e in enumerate(feature_coords):
        dist = torch.norm(e - ego_pose[None, :2], dim=-1).min()
        mapping[i] = dist

    mapping = sorted(mapping.items(), key=lambda item: item[1])
    sorted_elements = mapping[:max_elements]

    # pad or trim waypoints in a map element
    for idx, element_idx in enumerate(sorted_elements):
        element_coords = feature_coords[element_idx[0]]

        # interpolate to maintain fixed size if the number of points is not enough
        element_coords = interpolate_points(
            element_coords, max_points, interpolation=interpolation
        )
        coords_tensor[idx] = element_coords
        avails_tensor[idx] = True  # specify real vs zero-padded data

        if tl_data_tensor is not None and feature_tl_data is not None:
            tl_data_tensor[idx] = feature_tl_data[element_idx[0]]

    return coords_tensor, tl_data_tensor, avails_tensor


def get_neighbor_vector_set_map(
    map_api: AbstractMap,
    map_features: List[str],
    point: Point2D,
    radius: float,
    route_roadblock_ids: List[str],
    traffic_light_status_data: List[TrafficLightStatusData],
) -> Tuple[Dict[str, MapObjectPolylines], Dict[str, LaneSegmentTrafficLightData]]:
    """
    Extract neighbor vector set map information around ego vehicle.
    :param map_api: map to perform extraction on.
    :param map_features: Name of map features to extract.
    :param point: [m] x, y coordinates in global frame.
    :param radius: [m] floating number about vector map query range.
    :param route_roadblock_ids: List of ids of roadblocks/roadblock connectors (lane groups) within goal route.
    :param traffic_light_status_data: A list of all available data at the current time step.
    :return:
        coords: Dictionary mapping feature name to polyline vector sets.
        traffic_light_data: Dictionary mapping feature name to traffic light info corresponding to map elements
            in coords.
    :raise ValueError: if provided feature_name is not a valid VectorFeatureLayer.
    """
    coords: Dict[str, MapObjectPolylines] = {}
    traffic_light_data: Dict[str, LaneSegmentTrafficLightData] = {}
    feature_layers: List[VectorFeatureLayer] = []

    for feature_name in map_features:
        try:
            feature_layers.append(VectorFeatureLayer[feature_name])
        except KeyError:
            raise ValueError(
                f"Object representation for layer: {feature_name} is unavailable"
            )

    # extract lanes
    if VectorFeatureLayer.LANE in feature_layers:
        lanes_mid, lanes_left, lanes_right, lane_ids = get_lane_polylines(
            map_api, point, radius
        )

        # lane baseline paths
        coords[VectorFeatureLayer.LANE.name] = lanes_mid

        # lane traffic light data
        traffic_light_data[VectorFeatureLayer.LANE.name] = get_traffic_light_encoding(
            lane_ids, traffic_light_status_data
        )

        # lane boundaries
        if VectorFeatureLayer.LEFT_BOUNDARY in feature_layers:
            coords[VectorFeatureLayer.LEFT_BOUNDARY.name] = MapObjectPolylines(
                lanes_left.polylines
            )
        if VectorFeatureLayer.RIGHT_BOUNDARY in feature_layers:
            coords[VectorFeatureLayer.RIGHT_BOUNDARY.name] = MapObjectPolylines(
                lanes_right.polylines
            )

    # extract route
    if VectorFeatureLayer.ROUTE_LANES in feature_layers:
        route_polylines = get_route_lane_polylines_from_roadblock_ids(
            map_api, point, radius, route_roadblock_ids
        )
        coords[VectorFeatureLayer.ROUTE_LANES.name] = route_polylines

    # extract generic map objects
    for feature_layer in feature_layers:
        if feature_layer in VectorFeatureLayerMapping.available_polygon_layers():
            polygons = get_map_object_polygons(
                map_api,
                point,
                radius,
                VectorFeatureLayerMapping.semantic_map_layer(feature_layer),
            )
            coords[feature_layer.name] = polygons

    return coords, traffic_light_data


def map_process(
    anchor_state,
    coords,
    traffic_light_data,
    map_features,
    max_elements,
    max_points,
    interpolation_method,
):
    """
    This function process the data from the raw vector set map data.
    :param anchor_state: The current state of the ego vehicle.
    :param coords: The input data of the vectorized map coordinates.
    :param traffic_light_data: The input data of the traffic light data.
    :return: dict of the map elements.
    """

    # convert data to tensor list
    anchor_state_tensor = torch.tensor(
        [anchor_state.x, anchor_state.y, anchor_state.heading], dtype=torch.float32
    )
    list_tensor_data = {}

    for feature_name, feature_coords in coords.items():
        list_feature_coords = []

        # Pack coords into tensor list
        for element_coords in feature_coords.to_vector():
            list_feature_coords.append(
                torch.tensor(element_coords, dtype=torch.float32)
            )
        list_tensor_data[f"coords.{feature_name}"] = list_feature_coords

        # Pack traffic light data into tensor list if it exists
        if feature_name in traffic_light_data:
            list_feature_tl_data = []

            for element_tl_data in traffic_light_data[feature_name].to_vector():
                list_feature_tl_data.append(
                    torch.tensor(element_tl_data, dtype=torch.float32)
                )
            list_tensor_data[
                f"traffic_light_data.{feature_name}"
            ] = list_feature_tl_data

    """
    Vector set map data structure, including:
    coords: Dict[str, List[<np.ndarray: num_elements, num_points, 2>]].
            The (x, y) coordinates of each point in a map element across map elements per sample.
    traffic_light_data: Dict[str, List[<np.ndarray: num_elements, num_points, 4>]].
            One-hot encoding of traffic light status for each point in a map element across map elements per sample.
            Encoding: green [1, 0, 0, 0] yellow [0, 1, 0, 0], red [0, 0, 1, 0], unknown [0, 0, 0, 1]
    availabilities: Dict[str, List[<np.ndarray: num_elements, num_points>]].
            Boolean indicator of whether feature data is available for point at given index or if it is zero-padded.
    """

    tensor_output = {}
    traffic_light_encoding_dim = LaneSegmentTrafficLightData.encoding_dim()

    for feature_name in map_features:
        if f"coords.{feature_name}" in list_tensor_data:
            feature_coords = list_tensor_data[f"coords.{feature_name}"]

            feature_tl_data = (
                list_tensor_data[f"traffic_light_data.{feature_name}"]
                if f"traffic_light_data.{feature_name}" in list_tensor_data
                else None
            )

            coords, tl_data, avails = convert_feature_layer_to_fixed_size(
                anchor_state_tensor,
                feature_coords,
                feature_tl_data,
                max_elements[feature_name],
                max_points[feature_name],
                traffic_light_encoding_dim,
                interpolation=interpolation_method  # apply interpolation only for lane features
                if feature_name
                in [
                    VectorFeatureLayer.LANE.name,
                    VectorFeatureLayer.LEFT_BOUNDARY.name,
                    VectorFeatureLayer.RIGHT_BOUNDARY.name,
                    VectorFeatureLayer.ROUTE_LANES.name,
                    VectorFeatureLayer.CROSSWALK.name,
                ]
                else None,
            )

            coords = vector_set_coordinates_to_local_frame(
                coords, avails, anchor_state_tensor
            )

            tensor_output[f"vector_set_map.coords.{feature_name}"] = coords
            tensor_output[f"vector_set_map.availabilities.{feature_name}"] = avails

            if tl_data is not None:
                tensor_output[
                    f"vector_set_map.traffic_light_data.{feature_name}"
                ] = tl_data

    """
    Post-precoss the map elements to different map types. Each map type is a array with the following shape.
    N: number of map elements (fixed for a given map feature)
    P: number of points (fixed for a given map feature)
    F: number of features
    """

    for feature_name in map_features:
        if feature_name == "LANE":
            polylines = tensor_output[f"vector_set_map.coords.{feature_name}"].numpy()
            traffic_light_state = tensor_output[
                f"vector_set_map.traffic_light_data.{feature_name}"
            ].numpy()
            avails = tensor_output[
                f"vector_set_map.availabilities.{feature_name}"
            ].numpy()
            vector_map_lanes = polyline_process(polylines, avails, traffic_light_state)

        elif feature_name == "CROSSWALK":
            polylines = tensor_output[f"vector_set_map.coords.{feature_name}"].numpy()
            avails = tensor_output[
                f"vector_set_map.availabilities.{feature_name}"
            ].numpy()
            vector_map_crosswalks = polyline_process(polylines, avails)

        elif feature_name == "ROUTE_LANES":
            polylines = tensor_output[f"vector_set_map.coords.{feature_name}"].numpy()
            avails = tensor_output[
                f"vector_set_map.availabilities.{feature_name}"
            ].numpy()
            vector_map_route_lanes = polyline_process(polylines, avails)

        else:
            pass

    vector_map_output = {
        "map_lanes": vector_map_lanes,
        "map_crosswalks": vector_map_crosswalks,
        "route_lanes": vector_map_route_lanes,
    }

    return vector_map_output


def polyline_process(polylines, avails, traffic_light=None):
    dim = 3 if traffic_light is None else 7
    new_polylines = np.zeros(
        shape=(polylines.shape[0], polylines.shape[1], dim), dtype=np.float32
    )

    for i in range(polylines.shape[0]):
        if avails[i][0]:
            polyline = polylines[i]
            polyline_heading = wrap_to_pi(
                np.arctan2(
                    polyline[1:, 1] - polyline[:-1, 1],
                    polyline[1:, 0] - polyline[:-1, 0],
                )
            )
            polyline_heading = np.insert(polyline_heading, -1, polyline_heading[-1])[
                :, np.newaxis
            ]
            if traffic_light is None:
                new_polylines[i] = np.concatenate([polyline, polyline_heading], axis=-1)
            else:
                new_polylines[i] = np.concatenate(
                    [polyline, polyline_heading, traffic_light[i]], axis=-1
                )

    return new_polylines


# Path planning functions
def get_candidate_paths(edges, ego_state, candidate_lane_edge_ids):
    # get all paths
    paths = []
    for edge in edges:
        paths.extend(depth_first_search(edge, candidate_lane_edge_ids))

    # extract path polyline
    candidate_paths = []

    for i, path in enumerate(paths):
        path_polyline = []
        for edge in path:
            path_polyline.extend(edge.baseline_path.discrete_path)

        path_polyline = check_path(np.array(path_to_linestring(path_polyline).coords))
        dist_to_ego = scipy.spatial.distance.cdist(
            [(ego_state.rear_axle.x, ego_state.rear_axle.y)], path_polyline
        )
        path_polyline = path_polyline[dist_to_ego.argmin() :]
        if len(path_polyline) < 3:
            continue

        path_len = len(path_polyline) * 0.25
        polyline_heading = calculate_path_heading(path_polyline)
        path_polyline = np.stack(
            [path_polyline[:, 0], path_polyline[:, 1], polyline_heading], axis=1
        )
        candidate_paths.append((path_len, dist_to_ego.min(), path_polyline))

    # trim paths by length
    max_path_len = max([v[0] for v in candidate_paths])
    acceptable_path_len = MAX_LEN / 2 if max_path_len > MAX_LEN / 2 else max_path_len
    paths = [v for v in candidate_paths if v[0] >= acceptable_path_len]

    return paths


def generate_paths(paths, obstacles, ego_state):
    new_paths = []
    path_distance = []
    for (path_len, dist, path_polyline) in paths:
        if len(path_polyline) > 81:
            sampled_index = np.array([5, 10, 15, 20]) * 4
        elif len(path_polyline) > 61:
            sampled_index = np.array([5, 10, 15]) * 4
        elif len(path_polyline) > 41:
            sampled_index = np.array([5, 10]) * 4
        elif len(path_polyline) > 21:
            sampled_index = [20]
        else:
            sampled_index = [1]

        target_states = path_polyline[sampled_index].tolist()
        for j, state in enumerate(target_states):
            first_stage_path = calc_4points_bezier_path(
                ego_state.rear_axle.x,
                ego_state.rear_axle.y,
                ego_state.rear_axle.heading,
                state[0],
                state[1],
                state[2],
                3,
                sampled_index[j],
            )[0]
            second_stage_path = path_polyline[sampled_index[j] + 1 :, :2]
            path_polyline = np.concatenate(
                [first_stage_path, second_stage_path], axis=0
            )
            new_paths.append(path_polyline)
            path_distance.append(dist)

    # evaluate paths
    candiate_paths = {}
    for path, dist in zip(new_paths, path_distance):
        cost = calculate_cost(path, dist, obstacles)
        candiate_paths[cost] = path

    # sort paths by cost
    candidate_paths = []
    for cost in sorted(candiate_paths.keys())[:3]:
        path = candiate_paths[cost]
        path = post_process(path, ego_state)
        candidate_paths.append(path)

    return candidate_paths


def calculate_cost(path, dist, obstacles):
    # path curvature
    curvature = calculate_path_curvature(path[0:100])
    curvature = np.max(curvature)

    # lane change
    lane_change = dist

    # check obstacles
    obstacles = check_obstacles(path[0:100:10], obstacles)

    # final cost
    cost = 10 * obstacles + 1 * lane_change + 0.1 * curvature

    return cost


def post_process(path, ego_state):
    path = transform_to_ego_frame(path, ego_state)
    index = np.arange(0, len(path), 10)
    x = path[:, 0][index]
    y = path[:, 1][index]

    # spline interpolation
    rx, ry, ryaw, rk = calc_spline_course(x, y)
    spline_path = np.stack([rx, ry, ryaw, rk], axis=1)
    ref_path = spline_path[: MAX_LEN * 10]

    return ref_path


def calculate_path_curvature(path):
    dx = np.gradient(path[:, 0])
    dy = np.gradient(path[:, 1])
    d2x = np.gradient(dx)
    d2y = np.gradient(dy)
    curvature = np.abs(dx * d2y - d2x * dy) / (dx**2 + dy**2) ** (3 / 2)

    return curvature


def check_obstacles(path, obstacles):
    expanded_path = LineString(path).buffer((WIDTH / 2), cap_style=CAP_STYLE.square)

    for obstacle in obstacles:
        obstacle_polygon = obstacle.geometry
        if expanded_path.intersects(obstacle_polygon):
            return 1

    return 0


def get_candidate_edges(ego_state, starting_block):
    edges = []
    edges_distance = []
    ego_point = (ego_state.rear_axle.x, ego_state.rear_axle.y)

    for edge in starting_block.interior_edges:
        edges_distance.append(edge.polygon.distance(Point(ego_point)))
        if edge.polygon.distance(Point(ego_point)) < 4:
            edges.append(edge)

    # if no edge is close to ego, use the closest edge
    if len(edges) == 0:
        edges.append(starting_block.interior_edges[np.argmin(edges_distance)])

    return edges


def depth_first_search(
    starting_edge, candidate_lane_edge_ids, target_depth=MAX_LEN, depth=0
):
    if depth >= target_depth:
        return [[starting_edge]]
    else:
        traversed_edges = []
        child_edges = [
            edge
            for edge in starting_edge.outgoing_edges
            if edge.id in candidate_lane_edge_ids
        ]

        if child_edges:
            for child in child_edges:
                edge_len = len(child.baseline_path.discrete_path) * 0.25
                traversed_edges.extend(
                    depth_first_search(
                        child, candidate_lane_edge_ids, depth=depth + edge_len
                    )
                )

        if len(traversed_edges) == 0:
            return [[starting_edge]]

        edges_to_return = []

        for edge_seq in traversed_edges:
            edges_to_return.append([starting_edge] + edge_seq)

        return edges_to_return


def transform_to_ego_frame(path, ego_state):
    ego_x, ego_y, ego_h = (
        ego_state.rear_axle.x,
        ego_state.rear_axle.y,
        ego_state.rear_axle.heading,
    )
    path_x, path_y = path[:, 0], path[:, 1]
    ego_path_x = np.cos(ego_h) * (path_x - ego_x) + np.sin(ego_h) * (path_y - ego_y)
    ego_path_y = -np.sin(ego_h) * (path_x - ego_x) + np.cos(ego_h) * (path_y - ego_y)
    ego_path = np.stack([ego_path_x, ego_path_y], axis=-1)

    return ego_path


# Visulazation functions
def create_ego_raster(vehicle_state):
    # Extract ego vehicle dimensions
    vehicle_parameters = get_pacifica_parameters()
    ego_width = vehicle_parameters.width
    ego_front_length = vehicle_parameters.front_length
    ego_rear_length = vehicle_parameters.rear_length

    # Extract ego vehicle state
    x_center, y_center, heading = vehicle_state[0], vehicle_state[1], vehicle_state[2]
    ego_bottom_right = (x_center - ego_rear_length, y_center - ego_width / 2)

    # Paint the rectangle
    rect = plt.Rectangle(
        ego_bottom_right,
        ego_front_length + ego_rear_length,
        ego_width,
        linewidth=2,
        color="r",
        alpha=0.6,
        zorder=3,
        transform=mpl.transforms.Affine2D().rotate_around(
            *(x_center, y_center), heading
        )
        + plt.gca().transData,
    )
    plt.gca().add_patch(rect)


def create_agents_raster(agents):
    for i in range(agents.shape[0]):
        if agents[i, 0] != 0:
            x_center, y_center, heading = agents[i, 0], agents[i, 1], agents[i, 2]
            agent_length, agent_width = agents[i, 6], agents[i, 7]
            agent_bottom_right = (
                x_center - agent_length / 2,
                y_center - agent_width / 2,
            )

            rect = plt.Rectangle(
                agent_bottom_right,
                agent_length,
                agent_width,
                linewidth=2,
                color="m",
                alpha=0.6,
                zorder=3,
                transform=mpl.transforms.Affine2D().rotate_around(
                    *(x_center, y_center), heading
                )
                + plt.gca().transData,
            )
            plt.gca().add_patch(rect)


def create_map_raster(lanes, crosswalks, route_lanes):
    for i in range(lanes.shape[0]):
        lane = lanes[i]
        if lane[0][0] != 0:
            plt.plot(lane[:, 0], lane[:, 1], "k", linewidth=3)  # plot centerline

    for j in range(crosswalks.shape[0]):
        crosswalk = crosswalks[j]
        if crosswalk[0][0] != 0:
            plt.plot(
                crosswalk[:, 0], crosswalk[:, 1], "b", linewidth=4
            )  # plot crosswalk

    for k in range(route_lanes.shape[0]):
        route_lane = route_lanes[k]
        if route_lane[0][0] != 0:
            plt.plot(
                route_lane[:, 0], route_lane[:, 1], "g", linewidth=4
            )  # plot route_lanes


def draw_trajectory(ego_trajectory, agent_trajectories):
    # plot ego
    plt.plot(ego_trajectory[:, 0], ego_trajectory[:, 1], "r", linewidth=3, zorder=3)

    # plot others
    for i in range(agent_trajectories.shape[0]):
        if agent_trajectories[i, -1, 0] != 0:
            trajectory = agent_trajectories[i]
            plt.plot(trajectory[:, 0], trajectory[:, 1], "m", linewidth=3, zorder=3)


def draw_plans(trajectory_plans, stage=1):
    f = "y--" if stage == 1 else "c--"

    for i in range(trajectory_plans.shape[0]):
        trajectory = trajectory_plans[i]
        plt.plot(trajectory[:, 0], trajectory[:, 1], f, linewidth=2, zorder=5 - stage)
