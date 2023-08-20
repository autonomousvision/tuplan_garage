import random
from typing import Dict, Tuple

import numpy as np
import torch
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
from nuplan.planning.training.modeling.torch_module_wrapper import TorchModuleWrapper
from nuplan.planning.training.modeling.types import TargetsType
from nuplan.planning.training.preprocessing.feature_builders.raster_feature_builder import (
    RasterFeatureBuilder,
)
from nuplan.planning.training.preprocessing.features.trajectory import Trajectory
from nuplan.planning.training.preprocessing.target_builders.ego_trajectory_target_builder import (
    EgoTrajectoryTargetBuilder,
)

from tuplan_garage.planning.training.modeling.models.pgp.pgp_aggregator import PGP
from tuplan_garage.planning.training.modeling.models.pgp.pgp_decoder import LVM
from tuplan_garage.planning.training.modeling.models.pgp.pgp_encoder import PGPEncoder
from tuplan_garage.planning.training.modeling.models.pgp.utils import (
    get_traversal_coordinates,
    smooth_centerline_trajectory,
    waypoints_to_trajectory,
)
from tuplan_garage.planning.training.preprocessing.feature_builders.pgp.pgp_ego_agents_feature_builder import (
    PGPEgoAgentsFeatureBuilder,
)
from tuplan_garage.planning.training.preprocessing.feature_builders.pgp.pgp_feature_builder import (
    PGPFeatureBuilder,
)
from tuplan_garage.planning.training.preprocessing.feature_builders.pgp.pgp_graph_map_feature_builder import (
    PGPGraphMapFeatureBuilder,
)
from tuplan_garage.planning.training.preprocessing.features.pgp.pgp_features import (
    PGPFeatures,
)
from tuplan_garage.planning.training.preprocessing.features.pgp.pgp_graph_map import (
    PGPGraphMap,
)
from tuplan_garage.planning.training.preprocessing.features.pgp.pgp_targets import (
    PGPTargets,
)
from tuplan_garage.planning.training.preprocessing.features.trajectories_multimodal import (
    MultiModalTrajectories,
)
from tuplan_garage.planning.training.preprocessing.target_builders.multimodal_trajectories_target_builder import (
    MultimodalTrajectoriesTargetBuilder,
)


class PGPModel(TorchModuleWrapper):
    """
    Single-agent prediction model
    original implementation: https://github.com/nachiket92/PGP
    """

    def __init__(
        self,
        future_trajectory_sampling: TrajectorySampling,
        past_trajectory_sampling: TrajectorySampling,
        encoder: PGPEncoder,
        aggregator: PGP,
        decoder: LVM,
        map_extent: Tuple[int, int, int, int],
        polyline_resolution: int,
        polyline_length: int,
        proximal_edges_dist_thresh: float,
        proximal_edges_yaw_thresh: float,
        agent_node_att_dist_thresh: int,
        use_raster_feature_builder: bool = False,
        return_graph_map: bool = False,
        return_traversal_coordinates: bool = False,
        filter_trajectories_by_endpoint: bool = False,
        smooth_output_trajectory: bool = False,
        interpolate_yaw: bool = False,
        average_output_trajectories: bool = False,
    ):
        feature_builders = [
            PGPFeatureBuilder(
                pgp_graph_map_feature_builder=PGPGraphMapFeatureBuilder(
                    map_extent=map_extent,
                    polyline_resolution=polyline_resolution,
                    polyline_length=polyline_length,
                    proximal_edges_dist_thresh=proximal_edges_dist_thresh,
                    proximal_edges_yaw_thresh=proximal_edges_yaw_thresh,
                ),
                pgp_ego_agents_feature_builder=PGPEgoAgentsFeatureBuilder(
                    history_sampling=past_trajectory_sampling
                ),
                agent_node_att_dist_thresh=agent_node_att_dist_thresh,
                traversal_trajectory_sampling=future_trajectory_sampling,
            )
        ]
        self.return_graph_map = return_graph_map
        self.return_traversal_coordinates = return_traversal_coordinates
        self.filter_trajectories_by_endpoint = filter_trajectories_by_endpoint
        self.smooth_output_trajectory = smooth_output_trajectory
        self.interpolate_yaw = interpolate_yaw
        self.average_output_trajectories = average_output_trajectories
        if use_raster_feature_builder:
            feature_builders.append(
                RasterFeatureBuilder(
                    map_features={
                        "LANE": 1.0,
                        "INTERSECTION": 1.0,
                        "STOP_LINE": 0.5,
                        "CROSSWALK": 0.5,
                    },
                    num_input_channels=4,
                    target_width=224,
                    target_height=224,
                    target_pixel_size=0.25,
                    ego_width=2.297,
                    ego_front_length=4.049,
                    ego_rear_length=1.127,
                    ego_longitudinal_offset=0.0,
                    baseline_path_thickness=1,
                )
            )

        super().__init__(
            feature_builders=feature_builders,
            target_builders=[
                MultimodalTrajectoriesTargetBuilder(
                    future_trajectory_sampling=future_trajectory_sampling
                ),
                EgoTrajectoryTargetBuilder(
                    future_trajectory_sampling=future_trajectory_sampling
                ),
            ],
            future_trajectory_sampling=future_trajectory_sampling,
        )

        assert encoder.target_agent_enc_size == aggregator.target_agent_enc_size
        assert encoder.node_enc_size == aggregator.node_enc_size
        assert (
            aggregator.agg_enc_size == decoder.encoding_size
        ), f"Output size of aggregator ({aggregator.agg_enc_size}) has to match input size of decoder ({decoder.encoding_size})"
        assert decoder.op_len == future_trajectory_sampling.num_poses
        if aggregator.num_samples != 1:
            assert (
                aggregator.num_samples == aggregator.num_traversals
            ), """ Aggregator can only return all traversals or only the most frequent one.
                Selecting only a subset > 1 is not yet supported. """
        if aggregator.num_samples:
            assert aggregator.num_samples == decoder.num_samples
        if aggregator.horizon:
            assert aggregator.horizon == future_trajectory_sampling.num_poses

        self.encoder = encoder
        self.aggregator = aggregator
        self.decoder = decoder

    def silently_seed_everything(self, seed) -> None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    def get_final_waypoint_on_route_mask(
        self,
        trajectories: torch.Tensor,
        graph_map: PGPGraphMap,
        threshold: float,
    ) -> torch.Tensor:
        """
        Filters trajectories of shape [B, M, T, 2] to only keep the ones which have
        an endpoint close to the route.
        The route is given by a mask for lane graph nodes of shape [B, N, P, C].
        B: batch_size, M: number of predicted trajectories, T: prediction horizon,
        N: number of nodes, P: number of poses per node, C: number of features per pose

        return mask for trajectories [B, M]
        """
        endpoints = trajectories[:, :, -1, :]
        lane_graph_poses = graph_map.lane_node_feats[:, :, :, :2].flatten(
            start_dim=1, end_dim=2
        )

        pairwise_distance = torch.cdist(endpoints, lane_graph_poses, p=2)

        # mask for on route poses [B, 1, N*P]
        on_route_poses_mask = (
            graph_map.nodes_on_route_flag[..., 0]
            .flatten(start_dim=1, end_dim=2)
            .unsqueeze(1)
        )

        # set distance for padded poses and offroad poses to threshold
        pairwise_distance: torch.Tensor = (
            on_route_poses_mask * pairwise_distance
            + torch.where(on_route_poses_mask.bool(), 0.0, np.inf)
        )
        min_pairwise_distance = pairwise_distance.min(dim=-1).values
        endpoint_on_route_mask = (min_pairwise_distance < threshold).float()

        at_least_one_endpoint_one_route_mask = (
            endpoint_on_route_mask.sum(dim=-1) > 0
        ).float()
        smallest_distance_endpoint_mask = (
            min_pairwise_distance == min_pairwise_distance.min(dim=-1).values
        ).float()
        # if at least one endpoint is on route, return trajectories according to mask
        # if no endpoint is within the threshold, return trajectory with smallest distance
        return (
            at_least_one_endpoint_one_route_mask * endpoint_on_route_mask
            + (1 - at_least_one_endpoint_one_route_mask)
            * smallest_distance_endpoint_mask
        )

    def forward(self, features: Dict[str, PGPFeatures]) -> TargetsType:
        """
        Forward pass for prediction model
        :param inputs: Dictionary with
            'target_agent_representation': target agent history
            'surrounding_agent_representation': surrounding agent history
            'map_representation': HD map representation
        :return outputs: K Predicted trajectories and/or their probabilities
        """
        self.silently_seed_everything(0)
        encodings = self.encoder(features)
        agg_encoding = self.aggregator(encodings)
        outputs = self.decoder(agg_encoding)

        if self.average_output_trajectories:
            most_likely_trajectory = outputs["traj_none_clustered"].mean(dim=1)
        else:
            if self.filter_trajectories_by_endpoint:
                probs_mask = self.get_final_waypoint_on_route_mask(
                    trajectories=outputs["traj"],
                    graph_map=features["pgp_features"].graph_map,
                    threshold=5.0,  # same as in feature builder
                )
                probs = outputs["probs"] * probs_mask
            else:
                probs = outputs["probs"]

            most_likely_trajectory = (
                outputs["traj"]
                .take_along_dim(
                    indices=probs.argmax(dim=1, keepdim=True)[..., None, None],
                    dim=1,
                )
                .squeeze(dim=1)
            )

        current_velocity = features["pgp_features"].ego_agent_features.ego_feats[
            :, 0, -1, 2
        ]
        if self.interpolate_yaw:
            most_likely_trajectory = waypoints_to_trajectory(
                most_likely_trajectory, current_velocity
            )
        else:
            batch_size, num_poses = most_likely_trajectory.shape[:2]
            dummy_heading = torch.zeros(
                [batch_size, num_poses, 1], device=most_likely_trajectory.device
            )
            most_likely_trajectory = torch.cat(
                [most_likely_trajectory, dummy_heading], dim=-1
            )

        if self.smooth_output_trajectory:
            most_likely_trajectory = smooth_centerline_trajectory(
                most_likely_trajectory
            )

        if self.return_traversal_coordinates:
            # calculate_traversal_coordinates
            traversal_coordinates = get_traversal_coordinates(
                traversals=agg_encoding["sampled_traversals"],
                lane_node_feats=features["pgp_features"].graph_map.lane_node_feats,
                lane_node_masks=features["pgp_features"].graph_map.lane_node_masks,
            )
        else:
            traversal_coordinates = None

        predictions = {
            "pgp_targets": PGPTargets(
                pi=agg_encoding["pi"],
                trajectory_probabilities=outputs["probs"],
                visited_edges=features["pgp_features"].traversal_features.visited_edges,
                trajectories=outputs["traj"],
                traversal_coordinates=traversal_coordinates,
            ),
            "multimodal_trajectories": MultiModalTrajectories(
                trajectories=outputs["traj"], probabilities=outputs["probs"]
            ),
            "trajectory": Trajectory(data=most_likely_trajectory),
        }
        if self.return_graph_map:
            predictions["pgp_graph_map"] = features["pgp_features"].graph_map

        return predictions
