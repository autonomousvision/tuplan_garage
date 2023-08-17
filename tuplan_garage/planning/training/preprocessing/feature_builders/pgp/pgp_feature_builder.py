from __future__ import annotations

from typing import List, Tuple, Type

import numpy as np
from nuplan.planning.scenario_builder.abstract_scenario import AbstractScenario
from nuplan.planning.simulation.planner.abstract_planner import (
    PlannerInitialization,
    PlannerInput,
)
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
from nuplan.planning.training.preprocessing.feature_builders.abstract_feature_builder import (
    AbstractFeatureBuilder,
)
from nuplan.planning.training.preprocessing.features.abstract_model_feature import (
    AbstractModelFeature,
)
from nuplan.planning.training.preprocessing.features.trajectory_utils import (
    _convert_absolute_to_relative_states,
)
from scipy.spatial.distance import cdist

from tuplan_garage.planning.training.preprocessing.feature_builders.pgp.pgp_ego_agents_feature_builder import (
    PGPEgoAgentsFeatureBuilder,
)
from tuplan_garage.planning.training.preprocessing.feature_builders.pgp.pgp_graph_map_feature_builder import (
    PGPGraphMapFeatureBuilder,
)
from tuplan_garage.planning.training.preprocessing.features.pgp.pgp_agent_node_masks import (
    PGPAgentNodeMasks,
)
from tuplan_garage.planning.training.preprocessing.features.pgp.pgp_ego_agents import (
    PGPEgoAgents,
)
from tuplan_garage.planning.training.preprocessing.features.pgp.pgp_features import (
    PGPFeatures,
)
from tuplan_garage.planning.training.preprocessing.features.pgp.pgp_graph_map import (
    PGPGraphMap,
)
from tuplan_garage.planning.training.preprocessing.features.pgp.pgp_traversals import (
    PGPTraversals,
)


class PGPFeatureBuilder(AbstractFeatureBuilder):
    """
    Abstract class that creates model input features from database samples.
    """

    def __init__(
        self,
        pgp_graph_map_feature_builder: PGPGraphMapFeatureBuilder,
        pgp_ego_agents_feature_builder: PGPEgoAgentsFeatureBuilder,
        agent_node_att_dist_thresh: int,
        traversal_trajectory_sampling: TrajectorySampling,
    ):
        self.pgp_map_feature_builder = pgp_graph_map_feature_builder
        self.pgp_ego_agents_feature_builder = pgp_ego_agents_feature_builder
        self.agent_node_att_dist_thresh = agent_node_att_dist_thresh

        self.traversal_num_poses = traversal_trajectory_sampling.num_poses
        self.traversal_time_horizon = traversal_trajectory_sampling.time_horizon

    @classmethod
    def get_feature_type(cls) -> Type[AbstractModelFeature]:
        """Type of the built feature."""
        return PGPFeatures

    @classmethod
    def get_feature_unique_name(cls) -> str:
        """Unique string identifier of the built feature."""
        return "pgp_features"

    def get_features_from_simulation(
        self, current_input: PlannerInput, initialization: PlannerInitialization
    ) -> PGPFeatureBuilder:
        """
        Constructs model input features from simulation history.
        :param current_input: Iteration specific inputs for building the feature.
        :param initialization: Additional data require for building the feature.
        :return: Constructed features.
        """
        pgp_graph_map = self.pgp_map_feature_builder.get_features_from_simulation(
            current_input, initialization
        )
        pgp_ego_agents_feature = (
            self.pgp_ego_agents_feature_builder.get_features_from_simulation(
                current_input, initialization
            )
        )

        agent_node_attention_masks = self.get_agent_node_attention_masks(
            pgp_graph_map, pgp_ego_agents_feature
        )
        route_node_idcs = np.where(pgp_graph_map.nodes_on_route_flag[:, 0, 0])

        traversal_features = PGPTraversals(
            init_node=self.get_init_node(pgp_graph_map, route_node_idcs),
            node_seq_gt=np.zeros(self.traversal_num_poses),
            visited_edges=np.zeros_like(pgp_graph_map.s_next),
        )

        return PGPFeatures(
            ego_agent_features=pgp_ego_agents_feature,
            graph_map=pgp_graph_map,
            att_node_masks=agent_node_attention_masks,
            traversal_features=traversal_features,
        )

    def get_features_from_planner(
        self,
        current_input: PlannerInput,
        initialization: PlannerInitialization,
        route_roadblock_ids: List[str],
    ) -> PGPFeatureBuilder:
        """
        Constructs model input features from simulation history.
        :param current_input: Iteration specific inputs for building the feature.
        :param initialization: Additional data require for building the feature.
        :return: Constructed features.
        """
        pgp_graph_map = self.pgp_map_feature_builder.get_features_from_planner(
            current_input, initialization, route_roadblock_ids
        )
        pgp_ego_agents_feature = (
            self.pgp_ego_agents_feature_builder.get_features_from_planner(
                current_input, initialization, route_roadblock_ids
            )
        )

        agent_node_attention_masks = self.get_agent_node_attention_masks(
            pgp_graph_map, pgp_ego_agents_feature
        )

        route_node_idcs = np.where(pgp_graph_map.nodes_on_route_flag[:, 0, 0])

        traversal_features = PGPTraversals(
            init_node=self.get_init_node(pgp_graph_map, route_node_idcs),
            node_seq_gt=np.zeros(self.traversal_num_poses),
            visited_edges=np.zeros_like(pgp_graph_map.s_next),
        )

        return PGPFeatures(
            ego_agent_features=pgp_ego_agents_feature,
            graph_map=pgp_graph_map,
            att_node_masks=agent_node_attention_masks,
            traversal_features=traversal_features,
        )

    def get_features_from_scenario(
        self, scenario: AbstractScenario
    ) -> AbstractModelFeature:
        """
        Constructs model input features from a database samples.
        :param scenario: Generic scenario
        :return: Constructed features
        """
        pgp_graph_map = self.pgp_map_feature_builder.get_features_from_scenario(
            scenario
        )
        pgp_ego_agents_feature = (
            self.pgp_ego_agents_feature_builder.get_features_from_scenario(scenario)
        )

        agent_node_attention_masks = self.get_agent_node_attention_masks(
            pgp_graph_map, pgp_ego_agents_feature
        )
        ego_future = scenario.get_ego_future_trajectory(
            iteration=0,
            time_horizon=self.traversal_time_horizon,
            num_samples=self.traversal_num_poses,
        )

        ego_future = _convert_absolute_to_relative_states(
            scenario.initial_ego_state.rear_axle, [s.rear_axle for s in ego_future]
        )
        ego_future = [(s.x, s.y, s.heading) for s in ego_future]

        # traversed_node_idcs = self.get_expert_visited_lane_ids(scenario=scenario, lane_ids=pgp_graph_map.lane_ids)
        # traversal_features = self.get_traversal_features(pgp_graph_map, ego_future, traversed_node_idcs)
        node_on_route_flag = pgp_graph_map.nodes_on_route_flag[
            :, 0, 0
        ]  # use 0th feature of 0th pose as feature is equal for all poses of node
        route_node_idcs = np.where(node_on_route_flag)
        traversal_features = self.get_traversal_features(
            pgp_graph_map, ego_future, route_node_idcs
        )

        return PGPFeatures(
            ego_agent_features=pgp_ego_agents_feature,
            graph_map=pgp_graph_map,
            att_node_masks=agent_node_attention_masks,
            traversal_features=traversal_features,
        )

    # def get_expert_visited_lane_ids(self, scenario: AbstractScenario, lane_ids: List[float]):
    #     expert_states = scenario.get_expert_ego_trajectory()
    #     expert_poses = extract_ego_center(expert_states)

    #     # Get expert's route and simplify it by removing repeated consecutive route objects
    #     expert_route = get_route_unfiltered(map_api=scenario.map_api, poses=expert_poses)

    #     expert_route_node_idcs = [
    #         node_id
    #         for obj_list in expert_route
    #         for obj in obj_list
    #         for node_id in np.where(lane_ids==float(obj.id))[0]
    #     ]
    #     return list(set(expert_route_node_idcs))

    def get_traversal_features(
        self,
        pgp_graph_map: PGPGraphMap,
        ego_future_trajectory: List[Tuple[float, float, float]],
        route_node_idcs=None,
    ) -> PGPTraversals:
        init_node = self.get_init_node(pgp_graph_map, route_node_idcs)
        node_seq, visited_edges = self.get_visited_edges(
            pgp_graph_map, ego_future_trajectory, route_node_idcs
        )

        return PGPTraversals(
            init_node=init_node,
            node_seq_gt=node_seq,
            visited_edges=visited_edges,
        )

    def get_init_node(self, pgp_graph_map: PGPGraphMap, route_node_idcs: List[int]):
        """
        Returns initial node probabilities for initializing the graph traversal policy
        :param pgp_graph_map: PGPGraphMap with lane node features and edge look-up tables
        """

        # Unpack lane node poses
        node_feats = pgp_graph_map.lane_node_feats
        node_feat_lens = np.sum(1 - pgp_graph_map.lane_node_masks[:, :, 0], axis=1)
        node_poses = []
        for i, node_feat in enumerate(node_feats):
            if node_feat_lens[i] != 0:
                node_poses.append(node_feat[: int(node_feat_lens[i]), :3])

        assigned_nodes = self.assign_pose_to_node(
            node_poses,
            np.asarray([0, 0, 0]),
            dist_thresh=3,
            yaw_thresh=np.pi / 4,
            return_multiple=True,
            route_node_idcs=route_node_idcs,
        )

        init_node = np.zeros(pgp_graph_map.lane_node_feats.shape[0])
        init_node[assigned_nodes] = 1 / len(assigned_nodes)
        return init_node

    @staticmethod
    def assign_pose_to_node(
        node_poses,
        query_pose,
        dist_thresh=5,
        yaw_thresh=np.pi / 3,
        return_multiple=False,
        route_node_idcs=None,
    ):
        """
        Assigns a given agent pose to a lane node. Takes into account distance from the lane centerline as well as
        direction of motion.
        """
        dist_vals = []
        yaw_diffs = []

        for i in range(len(node_poses)):
            distances = np.linalg.norm(node_poses[i][:, :2] - query_pose[:2], axis=1)
            dist_vals.append(np.min(distances))
            idx = np.argmin(distances)
            yaw_lane = node_poses[i][idx, 2]
            yaw_query = query_pose[2]
            yaw_diffs.append(
                np.arctan2(np.sin(yaw_lane - yaw_query), np.cos(yaw_lane - yaw_query))
            )

        idcs_yaw = np.where(np.absolute(np.asarray(yaw_diffs)) <= yaw_thresh)[0]
        idcs_dist = np.where(np.asarray(dist_vals) <= dist_thresh)[0]
        idcs = np.intersect1d(idcs_dist, idcs_yaw)
        yaw_candidates = idcs_yaw
        if route_node_idcs is not None:
            idcs = np.intersect1d(idcs, route_node_idcs)
            yaw_candidates = np.intersect1d(yaw_candidates, route_node_idcs)

        if len(idcs) > 0:
            if return_multiple:
                return idcs
            else:
                return idcs[int(np.argmin(np.asarray(dist_vals)[idcs]))]
        elif len(yaw_candidates) > 0:
            # use closest node that statisifies yaw (and route) constraint
            filtered_dist_vals = [dist_vals[idx] for idx in yaw_candidates]
            idx = np.argmin(np.asarray(filtered_dist_vals))
            assigned_node_id = yaw_candidates[idx]
            if return_multiple:
                return np.asarray([assigned_node_id])
            else:
                return assigned_node_id
        else:
            # use closest node as a fallback
            assigned_node_id = np.argmin(np.asarray(dist_vals))
            if return_multiple:
                return np.asarray([assigned_node_id])
            else:
                return assigned_node_id

    def get_agent_node_attention_masks(
        self, pgp_graph_map: PGPGraphMap, pgp_ego_agents_feature: PGPEgoAgents
    ) -> PGPAgentNodeMasks:
        """
        Returns key/val masks for agent-node attention layers. All agents except those within a distance threshold of
        the lane node are masked. The idea is to incorporate local agent context at each lane node.
        """

        vehicle_node_masks = np.ones(
            (
                len(pgp_graph_map.lane_node_feats),
                len(pgp_ego_agents_feature.vehicle_agent_feats),
            )
        )
        ped_node_masks = np.ones(
            (
                len(pgp_graph_map.lane_node_feats),
                len(pgp_ego_agents_feature.pedestrians_agent_feats),
            )
        )

        # new implementation
        valid_lane_feature_mask = pgp_graph_map.lane_node_masks[..., 0] == 0
        valid_vehicle_mask = (pgp_ego_agents_feature.vehicle_agent_masks == 0).any(
            (1, 2)
        )
        valid_pedestrian_mask = (
            pgp_ego_agents_feature.pedestrians_agent_masks == 0
        ).any((1, 2))

        num_lanes, num_samples, num_features = pgp_graph_map.lane_node_feats.shape

        lane_coords = pgp_graph_map.lane_node_feats[..., :2]
        vehicle_coords = pgp_ego_agents_feature.vehicle_agent_feats[..., -1, :2]
        pedestrian_coords = pgp_ego_agents_feature.pedestrians_agent_feats[..., -1, :2]

        lane_vehicle_distances = cdist(
            lane_coords.reshape(-1, 2), vehicle_coords
        ).reshape(num_lanes, num_samples, len(vehicle_coords))
        lane_pedestrian_distances = cdist(
            lane_coords.reshape(-1, 2), pedestrian_coords
        ).reshape(num_lanes, num_samples, len(pedestrian_coords))

        lane_vehicle_distances[~valid_lane_feature_mask] = np.inf
        lane_vehicle_distances[..., ~valid_vehicle_mask] = np.inf

        lane_pedestrian_distances[~valid_lane_feature_mask] = np.inf
        lane_pedestrian_distances[..., ~valid_pedestrian_mask] = np.inf

        lane_vehicle_assign_mask = (
            lane_vehicle_distances.min(axis=1) <= self.agent_node_att_dist_thresh
        )
        lane_pedestrian_assign_mask = (
            lane_pedestrian_distances.min(axis=1) <= self.agent_node_att_dist_thresh
        )

        vehicle_node_masks[lane_vehicle_assign_mask] = 0
        ped_node_masks[lane_pedestrian_assign_mask] = 0

        return PGPAgentNodeMasks(
            vehicle_node_masks=vehicle_node_masks, pedestrian_node_masks=ped_node_masks
        )

    def _pose_is_in_map_extent(self, query_pose) -> bool:
        if self.pgp_map_feature_builder.map_extent is None:
            return True
        else:
            padding = (
                self.pgp_map_feature_builder.polyline_length
                * self.pgp_map_feature_builder.polyline_resolution
                / 2
            )
            return (
                self.pgp_map_feature_builder.map_extent[0] - padding <= query_pose[0]
                and query_pose[0]
                <= self.pgp_map_feature_builder.map_extent[1] + padding
                and self.pgp_map_feature_builder.map_extent[2] - padding
                <= query_pose[1]
                and query_pose[1]
                <= self.pgp_map_feature_builder.map_extent[3] + padding
            )

    # def assign_pose_to_successor_route_node(
    #     self,
    #     node_poses,
    #     query_pose,
    #     route_node_idcs,
    #     successor_idcs,
    # ) -> int:
    #     filter_idcs = np.intersect1d(route_node_idcs, successor_idcs)
    #     # try to assign to a node that is a successor and route node at the same time
    #     assigned_node = self.assign_pose_to_node(
    #         node_poses=node_poses,
    #         query_pose=query_pose,
    #         route_node_idcs=filter_idcs,
    #         dist_thresh=2.0,
    #         yaw_thresh=np.pi/8,
    #     )
    #     # if this fails: try to assign to a route node (fallback: closest node)
    #     if assigned_node not in filter_idcs:
    #         assigned_node = self.assign_pose_to_node(
    #             node_poses=node_poses,
    #             query_pose=query_pose,
    #             route_node_idcs=route_node_idcs,
    #             dist_thresh=2.0,
    #             yaw_thresh=np.pi/8,
    #         )
    #     return assigned_node

    def get_visited_edges(
        self,
        pgp_graph_map: PGPGraphMap,
        ego_future_trajectory: List[Tuple[float, float, float]],
        route_node_idcs: np.ndarray = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Returns nodes and edges of the lane graph visited by the actual target vehicle in the future. This serves as
        ground truth for training the graph traversal policy pi_route.
        :param idx: dataset index
        :param lane_graph: lane graph dictionary with lane node features and edge look-up tables
        :return: node_seq: Sequence of visited node ids.
                 evf: Look-up table of visited edges.
        """

        # Unpack lane graph dictionary
        node_feats = pgp_graph_map.lane_node_feats
        s_next = pgp_graph_map.s_next
        edge_type = pgp_graph_map.edge_type

        node_feat_lens = np.sum(1 - pgp_graph_map.lane_node_masks[:, :, 0], axis=1)
        node_poses = []
        for i, node_feat in enumerate(node_feats):
            if node_feat_lens[i] != 0:
                node_poses.append(node_feat[: int(node_feat_lens[i]), :3])

        # Initialize outputs
        current_step = 0
        node_seq = np.zeros(self.traversal_num_poses)
        evf = np.zeros_like(s_next)

        # Loop over future trajectory poses
        query_pose = np.asarray(ego_future_trajectory[0])
        current_node = self.assign_pose_to_node(
            node_poses=node_poses,
            query_pose=query_pose,
            route_node_idcs=route_node_idcs,
            dist_thresh=2.0,
            yaw_thresh=np.pi / 8,
        )
        node_seq[current_step] = current_node
        for n in range(1, len(ego_future_trajectory)):
            query_pose = np.asarray(ego_future_trajectory[n])
            dist_from_current_node = np.min(
                np.linalg.norm(node_poses[current_node][:, :2] - query_pose[:2], axis=1)
            )

            # If pose has deviated sufficiently from current node and is within area of interest, assign to a new node
            if dist_from_current_node >= 1.5 and self._pose_is_in_map_extent(
                query_pose
            ):

                # successor_idcs = s_next[current_node, edge_type[current_node]>0].astype(int)
                # assigned_node = self.assign_pose_to_successor_route_node(
                #     node_poses=node_poses,
                #     query_pose=query_pose,
                #     route_node_idcs=route_node_idcs,
                #     successor_idcs=successor_idcs,
                # )
                # # Assign new node to node sequence and edge to visited edges
                # if assigned_node not in node_seq:

                #     if assigned_node in s_next[current_node]:
                #         nbr_idx = np.where(s_next[current_node] == assigned_node)[0]
                #         nbr_valid = np.where(edge_type[current_node] > 0)[0]
                #         nbr_idx = np.intersect1d(nbr_idx, nbr_valid)
                #         if edge_type[current_node, nbr_idx] > 0:
                #             evf[current_node, nbr_idx] = 1

                #     current_node = assigned_node
                #     if current_step < self.traversal_num_poses-1:
                #         current_step += 1
                #         node_seq[current_step] = current_node
                assigned_node = self.assign_pose_to_node(
                    node_poses=node_poses,
                    query_pose=query_pose,
                    route_node_idcs=route_node_idcs,
                    dist_thresh=2.0,
                    yaw_thresh=np.pi / 8,
                )

                # Assign new node to node sequence and edge to visited edges
                if assigned_node != current_node:

                    if assigned_node in s_next[current_node]:
                        nbr_idx = np.where(s_next[current_node] == assigned_node)[0]
                        nbr_valid = np.where(edge_type[current_node] > 0)[0]
                        nbr_idx = np.intersect1d(nbr_idx, nbr_valid)

                        if edge_type[current_node, nbr_idx] > 0:
                            evf[current_node, nbr_idx] = 1

                    current_node = assigned_node
                    if current_step < self.traversal_num_poses - 1:
                        current_step += 1
                        node_seq[current_step] = current_node

        # Assign goal node and edge
        num_nodes = pgp_graph_map.lane_node_feats.shape[0]
        goal_node = current_node + num_nodes
        node_seq[current_step + 1 :] = goal_node
        evf[current_node, -1] = 1

        return node_seq, evf
