from __future__ import annotations

from typing import Dict, List, Tuple, Type, Union

import numpy as np
import numpy.typing as npt
from nuplan.common.actor_state.ego_state import EgoState
from nuplan.common.actor_state.state_representation import StateSE2
from nuplan.common.maps.abstract_map import AbstractMap, MapObject
from nuplan.common.maps.abstract_map_objects import LaneGraphEdgeMapObject
from nuplan.common.maps.maps_datatypes import (
    SemanticMapLayer,
    TrafficLightStatusData,
    TrafficLightStatusType,
)
from nuplan.planning.scenario_builder.abstract_scenario import AbstractScenario
from nuplan.planning.simulation.planner.abstract_planner import (
    PlannerInitialization,
    PlannerInput,
)
from nuplan.planning.training.preprocessing.feature_builders.abstract_feature_builder import (
    AbstractFeatureBuilder,
    AbstractModelFeature,
)
from scipy.interpolate import interp1d
from scipy.spatial.distance import cdist
from shapely.geometry import Polygon

from tuplan_garage.planning.training.preprocessing.feature_builders.pgp.pgp_graph_map_feature_builder_utils import (
    calculate_lane_progress,
    convert_absolute_to_relative_array,
    points_in_polygons,
)
from tuplan_garage.planning.training.preprocessing.feature_builders.route_utils import (
    get_correct_route_roadblock_ids,
)
from tuplan_garage.planning.training.preprocessing.features.pgp.pgp_graph_map import (
    PGPGraphMap,
)


def state_se2_to_array(state_se2: StateSE2):
    return np.array([state_se2.x, state_se2.y, state_se2.heading], dtype=np.float64)


state_se2_to_array_vectorize = np.vectorize(state_se2_to_array, signature="()->(3)")


class PGPGraphMapFeatureBuilder(AbstractFeatureBuilder):
    """
    Abstract class that creates model input features from database samples.
    """

    def __init__(
        self,
        map_extent: Tuple[int, int, int, int],
        polyline_resolution: int,
        polyline_length: int,
        proximal_edges_dist_thresh: float = 4,
        proximal_edges_yaw_thresh: float = np.pi / 4,
    ):
        self.map_extent = map_extent
        self.polyline_resolution = polyline_resolution
        self.polyline_length = polyline_length
        self.proximal_edges_dist_thresh = proximal_edges_dist_thresh
        self.proximal_edges_yaw_thresh = proximal_edges_yaw_thresh
        self.SEQUENCE_ELEMENT_INDICATORS = [
            SemanticMapLayer.STOP_LINE,
            SemanticMapLayer.CROSSWALK,
            SemanticMapLayer.CARPARK_AREA,
        ]
        self.SEQUENCE_ELEMENT_FEATURE_DIM = (
            len(self.SEQUENCE_ELEMENT_INDICATORS) + 4
        )  # 4=(x,y,theta, boundary_flag)

    @classmethod
    def get_feature_type(cls) -> Type[AbstractModelFeature]:
        """Type of the built feature."""
        return PGPGraphMap

    @classmethod
    def get_feature_unique_name(cls) -> str:
        """Unique string identifier of the built feature."""
        return "pgp_graph_map"

    def get_features_from_simulation(
        self, current_input: PlannerInput, initialization: PlannerInitialization
    ) -> PGPGraphMap:
        """
        Constructs model input features from simulation history.
        :param current_input: Iteration specific inputs for building the feature.
        :param initialization: Additional data require for building the feature.
        :return: Constructed features.
        """
        ego_state = current_input.history.ego_states[-1]
        map_api = initialization.map_api
        route_roadblock_ids = initialization.route_roadblock_ids
        traffic_light_status = [t for t in current_input.traffic_light_data]

        return self._compute_feature(
            ego_state, map_api, route_roadblock_ids, traffic_light_status
        )

    def get_features_from_planner(
        self,
        current_input: PlannerInput,
        initialization: PlannerInitialization,
        route_roadblock_ids: List[str],
    ) -> PGPGraphMap:
        """
        Constructs model input features from input information giving from the planner.
        Allows avoiding recomputation of features in each iteration.
        :param current_input: Iteration specific inputs for building the feature.
        :param initialization: Additional data require for building the feature.
        :return: Constructed features.
        """
        ego_state = current_input.history.ego_states[-1]
        map_api = initialization.map_api
        traffic_light_status = [t for t in current_input.traffic_light_data]

        return self._compute_feature(
            ego_state,
            map_api,
            route_roadblock_ids,
            traffic_light_status,
            correct_route_roadblock_ids=False,
        )

    def get_features_from_scenario(self, scenario: AbstractScenario) -> PGPGraphMap:
        """
        Constructs model input features from a database samples.
        :param scenario: Generic scenario
        :return: Constructed features
        """
        ego_state = scenario.initial_ego_state
        map_api = scenario.map_api
        route_roadblock_ids = scenario.get_route_roadblock_ids()
        traffic_light_status = [
            t for t in scenario.get_traffic_light_status_at_iteration(0)
        ]

        return self._compute_feature(
            ego_state, map_api, route_roadblock_ids, traffic_light_status
        )

    def _compute_feature(
        self,
        ego_state: EgoState,
        map_api: AbstractMap,
        route_roadblock_ids: List[str],
        traffic_light_status: List[TrafficLightStatusData],
        correct_route_roadblock_ids: bool = True,
    ) -> PGPGraphMap:

        if correct_route_roadblock_ids:
            route_roadblock_ids = get_correct_route_roadblock_ids(
                route_roadblock_ids=route_roadblock_ids,
                ego_pose=ego_state.center,
                map_api=map_api,
            )

        # closed_loop_route: Dict[str, LaneGraphEdgeMapObject] = \
        #     get_closed_loop_route(ego_state, map_api, route_roadblock_ids)

        # closed_loop_route_lanes = list(closed_loop_route.values())
        # for lane in closed_loop_route_lanes:
        #     left_lane, right_lane = lane.adjacent_edges

        #     if left_lane:
        #         closed_loop_route[left_lane.id] = left_lane

        #     if right_lane:
        #         closed_loop_route[right_lane.id] = right_lane

        if self.map_extent is None:
            num_roadblocks_ahead = 30
            route_roadblock_ids = route_roadblock_ids[:num_roadblocks_ahead]
            lanes = self.get_lanes_on_route(map_api, route_roadblock_ids)
        else:
            # Get lanes around agent within map_extent
            lanes = self.get_lanes_around_agent(ego_state, map_api)

        outgoing_lane_ids_lookup = {
            lane.id: [edge.id for edge in lane.outgoing_edges] for lane in lanes
        }
        incoming_lane_ids_lookup = {
            lane.id: [edge.id for edge in lane.incoming_edges] for lane in lanes
        }

        # Get relevant polygon layers from the map_api
        polygons = self.get_polygons_around_agent(ego_state, map_api)

        # Get vectorized representation of lanes
        lane_node_feats, lane_node_ids = self.get_lane_node_feats(
            ego_state, lanes, polygons
        )

        if self.map_extent:
            # Discard lanes outside map extent
            lane_node_feats, lane_node_ids = self.discard_poses_outside_extent(
                lane_node_feats, lane_node_ids
            )
        # Get edges
        e_succ = self.get_successor_edges(lane_node_ids, outgoing_lane_ids_lookup)
        e_prox = self.get_proximal_edges(
            lane_node_feats,
            e_succ,
            self.proximal_edges_dist_thresh,
            self.proximal_edges_yaw_thresh,
            outgoing_lane_ids_lookup,
            incoming_lane_ids_lookup,
            lane_node_ids,
        )

        # Concatenate flag indicating whether a node has successors to lane node feats
        lane_node_feats = self.add_boundary_flag(e_succ, lane_node_feats)

        # Add dummy node (0, 0, 0, 0, 0) if no lane nodes are found
        if len(lane_node_feats) == 0:
            lane_node_feats = [np.zeros((1, self.SEQUENCE_ELEMENT_FEATURE_DIM))]
            lane_node_ids = [-1]

        # Get edge lookup tables
        s_next, edge_type = self.get_edge_lookup(e_succ, e_prox, len(lane_node_feats))

        edge_on_route_mask, nodes_on_route_flag = self.get_on_route_feature(
            lanes, lane_node_ids, route_roadblock_ids, s_next
        )

        # edge_on_route_mask, nodes_on_route_flag = self.get_on_closed_loop_route_feature(
        #     closed_loop_route, lanes, lane_node_ids, route_roadblock_ids, s_next
        # )

        red_light_mask, red_light_flag = self.get_traffic_light_feature(
            lanes, lane_node_ids, traffic_light_status, s_next
        )

        # Convert list of lane node feats to fixed size numpy array and masks
        lane_node_feats, lane_node_masks = self.list_to_tensor(
            lane_node_feats,
            len(lane_node_feats),
            self.polyline_length,
            self.SEQUENCE_ELEMENT_FEATURE_DIM,
        )

        lane_node_ids = np.array(lane_node_ids, dtype=int)[..., None]

        return PGPGraphMap(
            lane_node_feats=lane_node_feats,
            lane_node_masks=lane_node_masks,
            lane_ids=lane_node_ids,
            s_next=s_next,
            edge_type=edge_type,
            edge_on_route_mask=edge_on_route_mask,
            nodes_on_route_flag=nodes_on_route_flag,
            red_light_flag=red_light_flag,
            red_light_mask=red_light_mask,
        )

    def get_traffic_light_feature(
        self,
        lanes: List[MapObject],
        lane_node_ids: List[str],
        traffic_light_status: List[TrafficLightStatusData],
        s_next: np.ndarray,
    ):
        """
        calculates traffic light flag and mask. The mask is 0 if the edge leads to a node on a LaneConnector where a red light
        prohibits going. The feature is 1 if the node is on a LaneConnector where a red light prohibits going.
        """
        lanes_traffic_light_status_flag = self.get_traffic_light_status_data_flag(
            lanes, traffic_light_status
        )

        lanes_traffic_light_status_flag.update({-1: False})
        nodes_traffic_light_status_flag = [
            lanes_traffic_light_status_flag[n] for n in lane_node_ids
        ]

        num_nodes = s_next.shape[0]
        num_edges = s_next.shape[1]

        red_light_mask = np.ones_like(s_next)
        for node in range(num_nodes):
            for edge in range(num_edges):
                successor_id = s_next[node, edge].astype(int)
                if successor_id >= num_nodes:
                    successor_id = successor_id - num_nodes
                if nodes_traffic_light_status_flag[successor_id]:
                    red_light_mask[node, edge] = 0

        # last edge refers to goal state (i.e. not leaving the node).
        red_light_mask[:, -1] = 1

        traffic_light_status_feature = np.asarray(nodes_traffic_light_status_flag)
        traffic_light_status_flag = traffic_light_status_feature[:, None, None].repeat(
            self.polyline_length, axis=1
        )
        return red_light_mask, traffic_light_status_flag

    def get_traffic_light_status_data_flag(
        self,
        lanes: List[MapObject],
        traffic_light_status: List[TrafficLightStatusData],
    ):
        # extract traffic light status for each lane
        lanes_traffic_light_status: Dict[str, TrafficLightStatusType] = {}
        for lane in lanes:
            if lane.has_traffic_lights():
                relevant_status = [
                    t
                    for t in traffic_light_status
                    if t.lane_connector_id == int(lane.id)
                ]
                lane_status = (
                    relevant_status[0].status
                    if len(relevant_status) > 0
                    else TrafficLightStatusType(value=TrafficLightStatusType.UNKNOWN)
                )
            else:
                lane_status = TrafficLightStatusType(value=TrafficLightStatusType.GREEN)
            lanes_traffic_light_status.update({lane.id: lane_status})
        # encode traffic light status for each lane
        return {
            k: s.value == TrafficLightStatusType.RED.value
            for k, s in lanes_traffic_light_status.items()
        }

    def get_on_route_feature(
        self,
        lanes: List[MapObject],
        lane_node_ids: List[str],
        route_roadblock_ids: List[str],
        s_next: np.ndarray,
    ) -> None:
        """
        Returns:
            edge_on_route_mask: [num_nodes, num_edges] indicating if an edge leads to a node that is on route
            nodes_on_route_flag: [num_nodes, num_poses, 1] indicating if a pose on the polyline is on route
                This can be used to stack it ont the lane_node_feats to add route information to the nodes
        """

        lanes_on_route_flag = {
            lane.id: lane.get_roadblock_id() in route_roadblock_ids for lane in lanes
        }
        lanes_on_route_flag.update({-1: False})
        nodes_on_route_flag = [lanes_on_route_flag[n] for n in lane_node_ids]

        num_nodes = s_next.shape[0]
        num_edges = s_next.shape[1]

        edge_on_route_mask = np.zeros_like(s_next)
        for node in range(num_nodes):
            for edge in range(num_edges):
                successor_id = s_next[node, edge].astype(int)
                if successor_id >= num_nodes:
                    successor_id = successor_id - num_nodes
                if nodes_on_route_flag[successor_id]:
                    edge_on_route_mask[node, edge] = 1

        # last edge refers to goal state (i.e. not leaving the node). This is always considered to be on route
        edge_on_route_mask[:, -1] = 1

        nodes_on_route_feature = np.asarray(nodes_on_route_flag)
        nodes_on_route_flag = nodes_on_route_feature[:, None, None].repeat(
            self.polyline_length, axis=1
        )

        return edge_on_route_mask, nodes_on_route_flag

    def get_on_closed_loop_route_feature(
        self,
        route_dict: Dict[str, LaneGraphEdgeMapObject],
        lanes: List[MapObject],
        lane_node_ids: List[str],
        route_roadblock_ids: List[str],
        s_next: np.ndarray,
    ) -> None:
        """
        Returns:
            edge_on_route_mask: [num_nodes, num_edges] indicating if an edge leads to a node that is on route
            nodes_on_route_flag: [num_nodes, num_poses, 1] indicating if a pose on the polyline is on route
                This can be used to stack it ont the lane_node_feats to add route information to the nodes
        """

        route_lane_ids = list(route_dict.keys())
        # print(route_lane_ids)
        lanes_on_route_flag = {lane.id: (lane.id in route_lane_ids) for lane in lanes}
        lanes_on_route_flag.update({-1: False})
        nodes_on_route_flag = [lanes_on_route_flag[n] for n in lane_node_ids]
        # print(nodes_on_route_flag)

        num_nodes = s_next.shape[0]
        num_edges = s_next.shape[1]

        edge_on_route_mask = np.zeros_like(s_next)
        for node in range(num_nodes):
            for edge in range(num_edges):
                successor_id = s_next[node, edge].astype(int)

                if successor_id >= num_nodes:
                    successor_id = successor_id - num_nodes

                if nodes_on_route_flag[successor_id]:
                    edge_on_route_mask[node, edge] = 1

        # last edge refers to goal state (i.e. not leaving the node). This is always considered to be on route
        edge_on_route_mask[:, -1] = 1

        nodes_on_route_feature = np.asarray(nodes_on_route_flag)
        nodes_on_route_flag = nodes_on_route_feature[:, None, None].repeat(
            self.polyline_length, axis=1
        )

        return edge_on_route_mask, nodes_on_route_flag

    def discretize_polyline(self, poses: List[StateSE2]) -> npt.NDArray[np.float64]:

        state_se2_array = state_se2_to_array_vectorize(
            np.array(poses, dtype=np.object_)
        )

        state_se2_array[:, 2] = np.unwrap(state_se2_array[:, 2], axis=0)

        progress = calculate_lane_progress(state_se2_array)
        interpolator = interp1d(progress, state_se2_array, axis=0)
        min_progress, max_progress = progress.min(), progress.max()

        num_samples = int((max_progress / self.polyline_resolution) + 1)
        sample_progress = (
            np.arange(0, num_samples, dtype=np.float64) * self.polyline_resolution
        )

        clipped_sample_progress = np.clip(sample_progress, min_progress, max_progress)
        interpolated_state_array = interpolator(clipped_sample_progress)

        last_interpolated_point = interpolated_state_array[-1]
        last_lane_point = state_se2_array[-1]

        residual_distance = (
            (last_interpolated_point - last_lane_point) ** 2
        ).sum() ** 0.5

        if residual_distance > 0.5 * self.polyline_resolution:
            interpolated_state_array = np.concatenate(
                [interpolated_state_array, last_lane_point[None, ...]], axis=0
            )

        return interpolated_state_array

    def get_lanes_on_route(
        self, map_api: AbstractMap, route_roadblock_ids
    ) -> List[MapObject]:  # -> Dict[SemanticMapLayer, List[MapObject]]:

        route_roadblocks = []
        for id_ in route_roadblock_ids:
            block = map_api.get_map_object(id_, SemanticMapLayer.ROADBLOCK)
            block = block or map_api.get_map_object(
                id_, SemanticMapLayer.ROADBLOCK_CONNECTOR
            )

            route_roadblocks.append(block)

        lanes = [
            lane for block in route_roadblocks if block for lane in block.interior_edges
        ]

        return lanes

    def get_lanes_around_agent(
        self, global_pose: EgoState, map_api: AbstractMap
    ) -> List[MapObject]:  # -> Dict[SemanticMapLayer, List[MapObject]]:
        radius = max(self.map_extent)
        lanes = map_api.get_proximal_map_objects(
            global_pose.rear_axle.point,
            radius=radius,
            layers=[SemanticMapLayer.LANE, SemanticMapLayer.LANE_CONNECTOR],
        )
        lanes = lanes[SemanticMapLayer.LANE] + lanes[SemanticMapLayer.LANE_CONNECTOR]
        return lanes

    def get_polygons_around_agent(
        self, global_pose: EgoState, map_api: AbstractMap
    ) -> Dict[SemanticMapLayer, List[MapObject]]:
        radius = max(self.map_extent) if self.map_extent else 200
        polygons = map_api.get_proximal_map_objects(
            global_pose.rear_axle.point,
            radius=radius,
            layers=self.SEQUENCE_ELEMENT_INDICATORS,
        )
        return polygons

    def get_lane_node_feats(
        self, global_pose: EgoState, lanes: List[MapObject], polygons: List[MapObject]
    ) -> Tuple[List[np.ndarray], List[str]]:

        lane_ids = [k.id for k in lanes]

        lane_discrete_paths = [lane.baseline_path.discrete_path for lane in lanes]

        lanes = [
            self.discretize_polyline(discrete_path)
            for discrete_path in lane_discrete_paths
        ]

        # Get flags indicating whether a lane lies on stop lines or crosswalks
        lane_flags = self.get_lane_flags(lanes, polygons)
        lanes = [
            convert_absolute_to_relative_array(global_pose.rear_axle, lane)
            for lane in lanes
        ]

        # Concatenate lane poses and lane flags
        lane_node_feats = [
            np.concatenate((lanes[i], lane_flags[i]), axis=1) for i in range(len(lanes))
        ]

        # Split lane centerlines into smaller segments:
        lane_node_feats, lane_node_ids = self.split_lanes(
            lane_node_feats, self.polyline_length, lane_ids
        )

        return lane_node_feats, lane_node_ids

    def discard_poses_outside_extent(
        self, pose_set: List[np.ndarray], ids: List[str] = None
    ) -> Union[List[np.ndarray], Tuple[List[np.ndarray], List[str]]]:
        """
        Discards lane or agent poses outside predefined extent in target agent's frame of reference.
        :param pose_set: agent or lane polyline poses
        :param ids: annotation record tokens for pose_set. Only applies to lanes.
        :return: Updated pose set
        """
        updated_pose_set = []
        updated_ids = []

        for m, poses in enumerate(pose_set):
            flag = False
            for n, pose in enumerate(poses):
                if (
                    self.map_extent[0] <= pose[0] <= self.map_extent[1]
                    and self.map_extent[2] <= pose[1] <= self.map_extent[3]
                ):
                    flag = True

            if flag:
                updated_pose_set.append(poses)
                if ids is not None:
                    updated_ids.append(ids[m])

        if ids is not None:
            return updated_pose_set, updated_ids
        else:
            return updated_pose_set

    @staticmethod
    def split_lanes(
        lanes: List[np.ndarray], max_len: int, lane_ids: List[str]
    ) -> Tuple[List[np.ndarray], List[str]]:
        """
        Splits lanes into roughly equal sized smaller segments with defined maximum length
        :param lanes: list of lane poses
        :param max_len: maximum admissible length of polyline
        :param lane_ids: list of lane ID tokens
        :return lane_segments: list of smaller lane segments
                lane_segment_ids: list of lane ID tokens corresponding to original lane that the segment is part of
        """
        lane_segments = []
        lane_segment_ids = []
        for idx, lane in enumerate(lanes):
            n_segments = int(np.ceil(len(lane) / max_len))
            n_poses = int(np.ceil(len(lane) / n_segments))
            for n in range(n_segments):
                lane_segment = lane[n * n_poses : (n + 1) * n_poses]
                lane_segments.append(lane_segment)
                lane_segment_ids.append(lane_ids[idx])

        return lane_segments, lane_segment_ids

    @staticmethod
    def get_lane_flags(
        lanes: List[List[Tuple]], map_object_dict: Dict[str, List[Polygon]]
    ) -> List[np.ndarray]:
        """
        Returns flags indicating whether each pose on lane polylines lies on polygon map layers
        like stop-lines or cross-walks
        :param lanes: list of lane poses
        :param polygons: dictionary of polygon layers
        :return lane_flags: list of ndarrays with flags
        """

        lane_flags = []
        for lane_num, lane in enumerate(lanes):

            flags = np.zeros((len(lane), len(map_object_dict.keys())), dtype=np.bool_)
            lane_points = lane[..., :2]
            for n, k in enumerate(map_object_dict.keys()):

                polygon_list = [obj.polygon for obj in map_object_dict[k]]
                p_in_p = points_in_polygons(lane_points, polygon_list)
                flags[:, n] = np.any(p_in_p, axis=0)

            lane_flags.append(flags.astype(np.float64))

        return lane_flags

    @staticmethod
    def list_to_tensor(
        feat_list: List[np.ndarray], max_num: int, max_len: int, feat_size: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Converts a list of sequential features (e.g. lane polylines or agent history) to fixed size numpy arrays for
        forming mini-batches
        :param feat_list: List of sequential features
        :param max_num: Maximum number of sequences in List
        :param max_len: Maximum length of each sequence
        :param feat_size: Feature dimension
        :return: 1) ndarray of features of shape [max_num, max_len, feat_dim]. Has zeros where elements are missing,
            2) ndarray of binary masks of shape [max_num, max_len, feat_dim]. Has ones where elements are missing.
        """
        feat_array = np.zeros((max_num, max_len, feat_size))
        mask_array = np.ones((max_num, max_len, feat_size))
        for n, feats in enumerate(feat_list):
            feat_array[n, : len(feats), :] = feats
            mask_array[n, : len(feats), :] = 0

        return feat_array, mask_array

    @staticmethod
    def get_successor_edges(
        lane_ids: List[str], outgoing_lane_ids_lookup: Dict[str, List[str]]
    ) -> List[List[int]]:
        """
        Returns successor edge list for each node
        Note: lane_ids are the ids of the polyline sequences after splitting while lanes are the original lanes
        """
        e_succ = []
        for node_id, lane_id in enumerate(lane_ids):
            e_succ_node = []
            if node_id + 1 < len(lane_ids) and lane_id == lane_ids[node_id + 1]:
                e_succ_node.append(node_id + 1)
            else:
                outgoing_lane_ids = outgoing_lane_ids_lookup[lane_id]
                for outgoing_id in outgoing_lane_ids:
                    if outgoing_id in lane_ids:
                        e_succ_node.append(lane_ids.index(outgoing_id))

            e_succ.append(e_succ_node)

        return e_succ

    def get_proximal_edges(
        self,
        lane_node_feats: List[np.ndarray],
        e_succ: List[List[int]],
        proximal_edges_dist_thresh: float,
        proximal_edges_yaw_thresh: float,
        outgoing_lane_ids_lookup: Dict[str, List[str]],
        incoming_lane_ids_lookup: Dict[str, List[str]],
        lane_ids: List[str],
    ) -> List[List[int]]:
        """
        Returns proximal edge list for each node
        """

        e_prox = [[] for _ in lane_node_feats]

        lane_node_feats_array, lane_node_masks = self.list_to_tensor(
            lane_node_feats,
            len(lane_node_feats),
            self.polyline_length,
            self.SEQUENCE_ELEMENT_FEATURE_DIM - 1,
        )

        num_lanes, num_samples, num_features = lane_node_feats_array.shape
        lane_node_masks = (1 - lane_node_masks).astype(np.bool_).any(-1)

        mean_yaw_cos = np.mean(
            np.cos(lane_node_feats_array[..., 2]), axis=-1, where=lane_node_masks
        )
        mean_yaw_sin = np.mean(
            np.sin(lane_node_feats_array[..., 2]), axis=-1, where=lane_node_masks
        )

        normalized_yaw = np.arctan2(mean_yaw_sin, mean_yaw_cos)[..., None]
        pairwise_yaw_err = normalized_yaw - normalized_yaw.swapaxes(0, 1)

        normalized_pairwise_yaw_err = np.arctan2(
            np.sin(pairwise_yaw_err), np.cos(pairwise_yaw_err)
        )
        valid_yaw_err = np.abs(normalized_pairwise_yaw_err) <= proximal_edges_yaw_thresh

        lane_centroids = np.mean(
            lane_node_feats_array[..., :2], axis=-2, where=lane_node_masks[..., None]
        )
        lane_centroid_within_distance = cdist(lane_centroids, lane_centroids) <= 20.0

        for src_node_id, src_node_feats in enumerate(lane_node_feats):
            for dest_node_id in range(src_node_id + 1, len(lane_node_feats)):
                if (
                    # edge cannot be successor and proximal edge at the same time
                    (dest_node_id not in e_succ[src_node_id])
                    and (valid_yaw_err[src_node_id, dest_node_id])
                    and (lane_centroid_within_distance[src_node_id, dest_node_id])
                    and (src_node_id not in e_succ[dest_node_id])
                    # proximal nodes cannot belong to the same lane (prevents successor skip edges)
                    # or between nodes on lane and successor lane or lane connector
                    # and (lane_ids[src_node_id] != lane_ids[dest_node_id])
                    # and (lane_ids[dest_node_id] not in outgoing_lane_ids_lookup[lane_ids[src_node_id]])
                    # # no proximal edges for lane connectors to make expert assignment more stable
                    # and (len(list(set(outgoing_lane_ids_lookup[lane_ids[dest_node_id]]) & set(outgoing_lane_ids_lookup[lane_ids[src_node_id]]))) == 0)
                    # and (len(list(set(incoming_lane_ids_lookup[lane_ids[dest_node_id]]) & set(incoming_lane_ids_lookup[lane_ids[src_node_id]]))) == 0)
                ):
                    dest_node_feats = lane_node_feats[dest_node_id]
                    pairwise_dist = cdist(src_node_feats[:, :2], dest_node_feats[:, :2])
                    min_dist = np.min(pairwise_dist)

                    if min_dist <= proximal_edges_dist_thresh:

                        e_prox[src_node_id].append(dest_node_id)
                        e_prox[dest_node_id].append(src_node_id)

        return e_prox

    @staticmethod
    def add_boundary_flag(e_succ: List[List[int]], lane_node_feats: np.ndarray):
        """
        Adds a binary flag to lane node features indicating whether the lane node has any successors.
        Serves as an indicator for boundary nodes.
        """
        for n, lane_node_feat_array in enumerate(lane_node_feats):
            flag = 1 if len(e_succ[n]) == 0 else 0
            lane_node_feats[n] = np.concatenate(
                (lane_node_feat_array, flag * np.ones((len(lane_node_feat_array), 1))),
                axis=1,
            )

        return lane_node_feats

    def get_edge_lookup(
        self, e_succ: List[List[int]], e_prox: List[List[int]], num_nodes: int
    ):
        """
        Returns edge look up tables
        :param e_succ: Lists of successor edges for each node
        :param e_prox: Lists of proximal edges for each node
        :return:
        s_next: Look-up table mapping source node to destination node for each edge. Each row corresponds to
        a source node, with entries corresponding to destination nodes. Last entry is always a terminal edge to a goal
        state at that node. shape: [max_nodes, max_nbr_nodes + 1]. Last
        edge_type: Look-up table of the same shape as s_next containing integer values for edge types.
        {0: No edge exists, 1: successor edge, 2: proximal edge, 3: terminal edge}
        """
        num_nbrs = [len(e_succ[i]) + len(e_prox[i]) for i in range(len(e_succ))]
        max_nbrs = max(num_nbrs) if len(num_nbrs) > 0 else 1

        s_next = np.zeros((num_nodes, max_nbrs + 1))
        edge_type = np.zeros((num_nodes, max_nbrs + 1), dtype=int)

        for src_node in range(len(e_succ)):
            nbr_idx = 0
            successors = e_succ[src_node]
            prox_nodes = e_prox[src_node]

            # Populate successor edges
            for successor in successors:
                s_next[src_node, nbr_idx] = successor
                edge_type[src_node, nbr_idx] = 1
                nbr_idx += 1

            # Populate proximal edges
            for prox_node in prox_nodes:
                s_next[src_node, nbr_idx] = prox_node
                edge_type[src_node, nbr_idx] = 2
                nbr_idx += 1

            # Populate terminal edge
            s_next[src_node, -1] = src_node + num_nodes
            edge_type[src_node, -1] = 3

        return s_next, edge_type
