from typing import Dict, List, Tuple

import torch
from nuplan.common.actor_state.state_representation import Point2D
from nuplan.common.actor_state.tracked_objects_types import TrackedObjectType
from nuplan.common.geometry.torch_geometry import (
    global_state_se2_tensor_to_local,
    vector_set_coordinates_to_local_frame,
)
from nuplan.common.maps.abstract_map import AbstractMap
from nuplan.common.maps.maps_datatypes import TrafficLightStatusData
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
    sampled_past_ego_states_to_tensor,
    sampled_past_timestamps_to_tensor,
)
from nuplan.planning.training.preprocessing.utils.vector_preprocessing import (
    interpolate_points,
)


def observation_adapter(
    history_buffer, traffic_light_data, map_api, route_roadblock_ids, device="cpu"
):
    num_agents = 20
    map_features = [
        "LANE",
        "ROUTE_LANES",
        "CROSSWALK",
    ]  # name of map features to be extracted.
    max_elements = {
        "LANE": 40,
        "ROUTE_LANES": 10,
        "CROSSWALK": 5,
    }  # maximum number of elements to extract per feature layer.
    max_points = {
        "LANE": 50,
        "ROUTE_LANES": 50,
        "CROSSWALK": 30,
    }  # maximum number of points per feature to extract per feature layer.
    radius = 80  # [m] query radius scope relative to the current pose.
    interpolation_method = "linear"

    ego_state_buffer = (
        history_buffer.ego_state_buffer
    )  # Past ego state including the current
    observation_buffer = (
        history_buffer.observation_buffer
    )  # Past observations including the current

    ego_agent_past = sampled_past_ego_states_to_tensor(ego_state_buffer)
    (
        past_tracked_objects_tensor_list,
        past_tracked_objects_types,
    ) = sampled_tracked_objects_to_tensor_list(observation_buffer)
    time_stamps_past = sampled_past_timestamps_to_tensor(
        [state.time_point for state in ego_state_buffer]
    )
    ego_state = history_buffer.current_state[0]
    ego_coords = Point2D(ego_state.rear_axle.x, ego_state.rear_axle.y)
    coords, traffic_light_data = get_neighbor_vector_set_map(
        map_api,
        map_features,
        ego_coords,
        radius,
        route_roadblock_ids,
        traffic_light_data,
    )

    ego_agent_past, neighbor_agents_past = agent_past_process(
        ego_agent_past,
        time_stamps_past,
        past_tracked_objects_tensor_list,
        past_tracked_objects_types,
        num_agents,
    )

    vector_map = map_process(
        ego_state.rear_axle,
        coords,
        traffic_light_data,
        map_features,
        max_elements,
        max_points,
        interpolation_method,
    )

    data = {
        "ego_agent_past": ego_agent_past[1:],
        "neighbor_agents_past": neighbor_agents_past[:, 1:],
    }
    data.update(vector_map)
    data = convert_to_model_inputs(data, device)

    return data


def convert_to_model_inputs(data, device):
    tensor_data = {}
    for k, v in data.items():
        tensor_data[k] = v.float().unsqueeze(0).to(device)

    return tensor_data


def extract_agent_tensor(tracked_objects, track_token_ids, object_types):
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
    object_types = [
        TrackedObjectType.VEHICLE,
        TrackedObjectType.PEDESTRIAN,
        TrackedObjectType.BICYCLE,
    ]
    output = []
    output_types = []
    track_token_ids = {}

    for i in range(len(past_tracked_objects)):
        tensorized, track_token_ids, agent_types = extract_agent_tensor(
            past_tracked_objects[i].tracked_objects, track_token_ids, object_types
        )
        output.append(tensorized)
        output_types.append(agent_types)

    return output, output_types


def convert_feature_layer_to_fixed_size(
    ego_pose,
    feature_coords,
    feature_tl_data,
    max_elements,
    max_points,
    traffic_light_encoding_dim,
    interpolation,
):
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
    agents_states_dim = Agents.agents_states_dim()
    ego_history = past_ego_states
    time_stamps = past_time_stamps
    agents = past_tracked_objects

    anchor_ego_state = ego_history[-1, :].squeeze().clone()
    ego_tensor = convert_absolute_quantities_to_relative(ego_history, anchor_ego_state)
    agent_history = filter_agents_tensor(agents, reverse=True)
    agent_types = tracked_objects_types[-1]

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

    agents = torch.zeros(
        (num_agents, agents_tensor.shape[0], agents_tensor.shape[-1] + 3),
        dtype=torch.float32,
    )

    # sort agents according to distance to ego
    distance_to_ego = torch.norm(agents_tensor[-1, :, :2], dim=-1)
    indices = list(torch.argsort(distance_to_ego).numpy())[:num_agents]

    # fill agent features into the array
    added_agents = 0
    for i in indices:
        if added_agents >= num_agents:
            break

        if agents_tensor[-1, i, 0] < -6.0:
            continue

        agents[added_agents, :, : agents_tensor.shape[-1]] = agents_tensor[
            :, i, : agents_tensor.shape[-1]
        ]

        if agent_types[i] == TrackedObjectType.VEHICLE:
            agents[added_agents, :, agents_tensor.shape[-1] :] = torch.tensor([1, 0, 0])
        elif agent_types[i] == TrackedObjectType.PEDESTRIAN:
            agents[added_agents, :, agents_tensor.shape[-1] :] = torch.tensor([0, 1, 0])
        else:
            agents[added_agents, :, agents_tensor.shape[-1] :] = torch.tensor([0, 0, 1])

        added_agents += 1

    return ego_tensor, agents


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

    for feature_name in map_features:
        if feature_name == "LANE":
            polylines = tensor_output[f"vector_set_map.coords.{feature_name}"]
            traffic_light_state = tensor_output[
                f"vector_set_map.traffic_light_data.{feature_name}"
            ]
            avails = tensor_output[f"vector_set_map.availabilities.{feature_name}"]
            vector_map_lanes = polyline_process(polylines, avails, traffic_light_state)

        elif feature_name == "CROSSWALK":
            polylines = tensor_output[f"vector_set_map.coords.{feature_name}"]
            avails = tensor_output[f"vector_set_map.availabilities.{feature_name}"]
            vector_map_crosswalks = polyline_process(polylines, avails)

        elif feature_name == "ROUTE_LANES":
            polylines = tensor_output[f"vector_set_map.coords.{feature_name}"]
            avails = tensor_output[f"vector_set_map.availabilities.{feature_name}"]
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
    new_polylines = torch.zeros(
        (polylines.shape[0], polylines.shape[1], dim), dtype=torch.float32
    )

    for i in range(polylines.shape[0]):
        if avails[i][0]:
            polyline = polylines[i]
            polyline_heading = torch.atan2(
                polyline[1:, 1] - polyline[:-1, 1], polyline[1:, 0] - polyline[:-1, 0]
            )
            polyline_heading = torch.fmod(polyline_heading, 2 * torch.pi)
            polyline_heading = torch.cat(
                [polyline_heading, polyline_heading[-1].unsqueeze(0)], dim=0
            ).unsqueeze(-1)
            if traffic_light is None:
                new_polylines[i] = torch.cat([polyline, polyline_heading], dim=-1)
            else:
                new_polylines[i] = torch.cat(
                    [polyline, polyline_heading, traffic_light[i]], dim=-1
                )

    return new_polylines
