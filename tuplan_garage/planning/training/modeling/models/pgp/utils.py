import numpy as np
import ray
import torch
from nuplan.common.actor_state.ego_state import EgoState
from nuplan.common.actor_state.state_representation import (
    ProgressStateSE2,
    StateVector2D,
)
from nuplan.common.actor_state.vehicle_parameters import get_pacifica_parameters
from nuplan.planning.simulation.path.interpolated_path import InterpolatedPath
from scipy.interpolate._bsplines import make_interp_spline
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans


def bivariate_gaussian_activation(ip: torch.Tensor) -> torch.Tensor:
    """
    Activation function to output parameters of bivariate Gaussian distribution
    """
    mu_x = ip[..., 0:1]
    mu_y = ip[..., 1:2]
    sig_x = ip[..., 2:3]
    sig_y = ip[..., 3:4]
    rho = ip[..., 4:5]
    sig_x = torch.exp(sig_x)
    sig_y = torch.exp(sig_y)
    rho = torch.tanh(rho)
    out = torch.cat([mu_x, mu_y, sig_x, sig_y, rho], dim=-1)
    return out


@ray.remote
def cluster_and_rank_ray(k: int, data: np.ndarray, random_state: int = None):
    return cluster_and_rank(k, data, random_state)


def cluster_and_rank(k: int, data: np.ndarray, random_state: int = None):
    """
    Combines the clustering and ranking steps so that ray.remote gets called just once
    """

    def cluster(n_clusters: int, x: np.ndarray, random_state=None):
        """
        Cluster using Scikit learn
        """
        clustering_op = KMeans(
            n_clusters=n_clusters,
            n_init=1,
            max_iter=100,
            init="random",
            random_state=random_state,
        ).fit(x)
        return clustering_op.labels_, clustering_op.cluster_centers_

    def rank_clusters(cluster_counts, cluster_centers):
        """
        Rank the K clustered trajectories using Ward's criterion. Start with K cluster centers and cluster counts.
        Find the two clusters to merge based on Ward's criterion. Smaller of the two will get assigned rank K.
        Merge the two clusters. Repeat process to assign ranks K-1, K-2, ..., 2.
        """

        num_clusters = len(cluster_counts)
        cluster_ids = np.arange(num_clusters)
        ranks = np.ones(num_clusters)

        for i in range(num_clusters, 0, -1):
            # Compute Ward distances:
            centroid_dists = cdist(cluster_centers, cluster_centers)
            n1 = cluster_counts.reshape(1, -1).repeat(len(cluster_counts), axis=0)
            n2 = n1.transpose()
            wts = n1 * n2 / (n1 + n2)
            dists = wts * centroid_dists + np.diag(
                np.inf * np.ones(len(cluster_counts))
            )

            # Get clusters with min Ward distance and select cluster with fewer counts
            c1, c2 = np.unravel_index(dists.argmin(), dists.shape)
            c = c1 if cluster_counts[c1] <= cluster_counts[c2] else c2
            c_ = c2 if cluster_counts[c1] <= cluster_counts[c2] else c1

            # Assign rank i to selected cluster
            ranks[cluster_ids[c]] = i

            # Merge clusters and update identity of merged cluster
            cluster_centers[c_] = (
                cluster_counts[c_] * cluster_centers[c_]
                + cluster_counts[c] * cluster_centers[c]
            ) / (cluster_counts[c_] + cluster_counts[c])
            cluster_counts[c_] += cluster_counts[c]

            # Discard merged cluster
            cluster_ids = np.delete(cluster_ids, c)
            cluster_centers = np.delete(cluster_centers, c, axis=0)
            cluster_counts = np.delete(cluster_counts, c)

        return ranks

    cluster_lbls, cluster_ctrs = cluster(k, data, random_state)
    cluster_cnts = np.unique(cluster_lbls, return_counts=True)[1]
    cluster_ranks = rank_clusters(cluster_cnts.copy(), cluster_ctrs.copy())
    return {"lbls": cluster_lbls, "ranks": cluster_ranks, "counts": cluster_cnts}


def cluster_traj(
    k: int, traj: torch.Tensor, use_ray: bool = False, random_state: int = None
):
    """
    clusters sampled trajectories to output K modes.
    :param k: number of clusters
    :param traj: set of sampled trajectories, shape [batch_size, num_samples, traj_len, 2]
    :return: traj_clustered:  set of clustered trajectories, shape [batch_size, k, traj_len, 2]
             scores: scores for clustered trajectories (basically 1/rank), shape [batch_size, k]
    """

    # Initialize output tensors
    batch_size = traj.shape[0]
    num_samples = traj.shape[1]
    traj_len = traj.shape[2]
    device = traj.device

    # Down-sample traj along time dimension for faster clustering
    data = traj[:, :, 0::3, :]
    data = data.reshape(batch_size, num_samples, -1).detach().cpu().numpy()

    # Cluster and rank
    if use_ray:
        cluster_ops = ray.get(
            [
                cluster_and_rank_ray.remote(k, data_slice, random_state)
                for data_slice in data
            ]
        )
    else:
        cluster_ops = [
            cluster_and_rank(k, data_slice, random_state) for data_slice in data
        ]
    cluster_lbls = np.array([cluster_op["lbls"] for cluster_op in cluster_ops])
    cluster_counts = np.array([cluster_op["counts"] for cluster_op in cluster_ops])
    cluster_ranks = np.array([cluster_op["ranks"] for cluster_op in cluster_ops])

    # Compute mean (clustered) traj and scores
    lbls = (
        torch.as_tensor(cluster_lbls, device=device)
        .unsqueeze(-1)
        .unsqueeze(-1)
        .repeat(1, 1, traj_len, 2)
        .long()
    )
    traj_summed = torch.zeros(batch_size, k, traj_len, 2, device=device).scatter_add(
        1, lbls, traj
    )
    cnt_tensor = (
        torch.as_tensor(cluster_counts, device=device)
        .unsqueeze(-1)
        .unsqueeze(-1)
        .repeat(1, 1, traj_len, 2)
    )
    traj_clustered = traj_summed / cnt_tensor
    scores = 1 / torch.as_tensor(cluster_ranks, device=device)
    scores = scores / torch.sum(scores, dim=1)[0]

    return traj_clustered, scores


def waypoints_to_trajectory(
    waypoints: torch.Tensor, current_velocity: torch.Tensor, supersampling_ratio=100
) -> torch.Tensor:
    """
    Generates a yaw angle from a series of waypoints. Therefore the trajectory is interpolated with a spline,
        wich is then supersampled to calculate the yaw angle. Kinematic feasibility is not guaranteed.
        Waypoints are expected to be in local coordinates, with ego being located at (0,0) with heading 0.
        The x-axis is assumed to be the cars longitudinal axis, and the y-axis is assumed to point to the left in direction of travel
    Note:
        Gradients are preserved for waypoints. Heading is calculated without gradient information.
    :waypoints: [batch_size, num_poses, 2]
    :current_velocity: [batch_size]
    :supersampling_ratio: number of intermediate waypoints per original waypoint
    :returns: [batch_size, num_poses, 3]
    """
    batch_size = waypoints.shape[0]
    device = waypoints.device
    # prepend current ego position
    initial_position = torch.zeros([2], device=device).expand([batch_size, 1, 2])
    stationary_points = waypoints[:, :, 0] < 0.2
    waypoints[stationary_points] = 0.0
    waypoints = torch.cat([initial_position, waypoints], dim=1)
    # need to move waypoints to cpu for interpolation. Original tensor is kept to preserve gradients.
    current_velocity_cpu = current_velocity.detach().cpu()
    waypoints_cpu = waypoints.detach().cpu()
    num_waypoints = waypoints.shape[-2]

    time_steps_orig = torch.arange(0, num_waypoints)
    time_steps_interp = torch.arange(
        0, num_waypoints - 1 + 1.0 / supersampling_ratio, 1.0 / supersampling_ratio
    )
    output_slice = slice(
        0, (num_waypoints - 1) * supersampling_ratio + 1, supersampling_ratio
    )

    trajectories = []
    for sample_idx in range(batch_size):
        x_interp_func = make_interp_spline(
            time_steps_orig,
            waypoints_cpu[sample_idx, :, 0],
            k=3,
            bc_type=([(1, current_velocity_cpu[sample_idx])], [(2, 0.0)]),
        )

        y_interp_func = make_interp_spline(
            time_steps_orig,
            waypoints_cpu[sample_idx, :, 1],
            k=3,
            bc_type=([(1, 0.0)], [(2, 0.0)]),
        )

        x_interp = torch.tensor(x_interp_func(time_steps_interp), device=device)
        y_interp = torch.tensor(y_interp_func(time_steps_interp), device=device)
        yaw_interp = torch.cat(
            [
                torch.tensor([0], device=device),
                torch.atan2(y_interp[1:] - y_interp[:-1], x_interp[1:] - x_interp[:-1]),
            ]
        )

        yaw = yaw_interp[output_slice]

        poses = torch.cat([waypoints[sample_idx], yaw.unsqueeze(-1)], dim=-1)

        # Surpress noise when ego is not moving
        MINIMAL_TRAVELED_DISTANCE = 0.5
        valid_poses = [poses[0]]
        for pose in poses[1:]:
            traveled_distance = torch.norm(valid_poses[-1][:2] - pose[:2], p=2)
            last_valid_pose = (
                pose
                if traveled_distance > MINIMAL_TRAVELED_DISTANCE
                else valid_poses[-1]
            )
            valid_poses.append(last_valid_pose)
        poses = torch.stack(valid_poses)

        trajectories.append(poses[1:])

    trajectories = torch.stack(trajectories)
    trajectories[stationary_points, :] = 0.0
    return trajectories


def get_traversal_coordinates(
    traversals: torch.Tensor,
    lane_node_feats: torch.Tensor,
    lane_node_masks: torch.Tensor,
) -> torch.Tensor:
    """
    Calculates coordinates of each node of a traversal. The coordinates of a node are estimated as the mean of all poses within the node
    :param traversals: Tensor [batch_size, num_traversals, num_nodes_per_traversal] containing the index of each visited node
    :param lane_node_feats: Tensor [batch_size, num_nodes, num_poses_per_node, num_feats_per_node] with feature 0 (1) being x (y) coordinate
    :param lane_node_masks: Tensor [batch_size, num_nodes, num_poses_per_node, num_feats_per_node], 1 if feature is unavailable, 0, else
    Returns:
    :Tensor [batch_size, num_traversals, num_nodes_per_traversal, 2=(xy)]
    """
    lane_node_xy_feats = lane_node_feats[:, :, :, :2]
    lane_node_xy_masks = lane_node_masks[:, :, :, :2].any(dim=-1)

    # Set coordinates of missing poses to 0, s.t. they don't change the mean
    lane_node_xy_feats[lane_node_xy_masks, :] = 0
    # num_poses_per_node [batch_size, num_nodes] containing number of poses for each node
    num_poses_per_node = (~lane_node_xy_masks).int().sum(dim=-1)
    # lane_node_coords [batch_size, num_nodes, 2=(xy)] containing number of poses for each node
    lane_node_coords = lane_node_xy_feats.sum(dim=-2) / num_poses_per_node.unsqueeze(-1)

    # indices > num_nodes refer to a goal state. Its coordinates a set to be equal to the corresponding lane node
    num_nodes = lane_node_coords.shape[1]
    traversals[traversals >= num_nodes] -= num_nodes

    num_nodes_per_traversal = traversals.shape[-1]
    traversal_coordinates = torch.take_along_dim(
        input=lane_node_coords[:, :, None, :].repeat(1, 1, num_nodes_per_traversal, 1),
        indices=traversals[..., None].repeat(1, 1, 1, 2),
        dim=1,
    )

    return traversal_coordinates


def smooth_centerline_trajectory(trajectories: torch.Tensor):
    """
    trajectories: [batch_size, num_poses, 3]
    """
    traj_shape = trajectories.shape
    assert torch.all(torch.isfinite(trajectories)), f"nans / infs in {trajectories}"
    device = trajectories.device
    batch_size = trajectories.shape[0]
    vehicle_parameters = get_pacifica_parameters()

    # append current position
    trajectories = torch.cat(
        [torch.zeros([batch_size, 1, 3], device=device), trajectories], dim=1
    )
    smoothed_trajectories = []
    for trajectory in trajectories:
        _, idxs = np.unique(
            trajectory.detach().cpu().numpy(), axis=0, return_index=True
        )
        idxs = np.sort(idxs)
        unique_waypoints = trajectory[idxs, :]
        if unique_waypoints.shape[0] == 1:
            # only one unique point -> cannot interpolate
            # and no smoothing necessary
            smoothed_trajectories.append(trajectory[1:, :])
        else:
            original_waypoints_progress = [
                torch.norm(current_waypoint[:2] - last_waypoint[:2], p=2)
                for last_waypoint, current_waypoint in zip(
                    trajectory[:-1], trajectory[1:]
                )
            ]
            original_waypoints_progress = list(np.cumsum(original_waypoints_progress))
            # remove duplicate points as they can cause nans
            unique_waypoints_progress = [
                torch.norm(current_waypoint[:2] - last_waypoint[:2], p=2)
                for last_waypoint, current_waypoint in zip(
                    unique_waypoints[:-1], unique_waypoints[1:]
                )
            ]
            unique_waypoints_progress = list(np.cumsum(unique_waypoints_progress))
            predicted_rear_axle_trajectory = [
                ProgressStateSE2(
                    x=current_waypoint[0],
                    y=current_waypoint[1],
                    heading=current_waypoint[2],
                    progress=prog,
                )
                for current_waypoint, prog in zip(
                    unique_waypoints[1:], unique_waypoints_progress
                )
            ]
            # append a state to make sure path is long enough to sample all center states
            final_heading = unique_waypoints[-1, 2]
            extrapolated_point = (
                unique_waypoints[-1, :]
                + torch.tensor([torch.cos(final_heading), torch.sin(final_heading), 0])
                * vehicle_parameters.rear_axle_to_center
            )
            predicted_rear_axle_trajectory.append(
                ProgressStateSE2(
                    x=extrapolated_point[0],
                    y=extrapolated_point[1],
                    heading=trajectory[-1, 2],
                    progress=predicted_rear_axle_trajectory[-1].progress
                    + vehicle_parameters.rear_axle_to_center,
                )
            )
            predicted_rear_axle_trajectory = InterpolatedPath(
                path=predicted_rear_axle_trajectory
            )
            # infer_centerline trajectory by moving states along the path
            # note: last progress is excluded as this refers to an appended state
            infered_smooth_centerline_trajectory = [
                predicted_rear_axle_trajectory.get_state_at_progress(
                    p + vehicle_parameters.rear_axle_to_center
                )
                for p in original_waypoints_progress
            ]
            infered_smooth_centerline_trajectory = [
                EgoState.build_from_center(
                    center=state,
                    center_acceleration_2d=StateVector2D(0, 0),
                    center_velocity_2d=StateVector2D(0, 0),
                    tire_steering_angle=0.0,
                    time_point=None,
                    vehicle_parameters=vehicle_parameters,
                )
                for state in infered_smooth_centerline_trajectory
            ]
            smooth_rear_axle_trajectory = [
                torch.tensor(
                    [e.rear_axle.x, e.rear_axle.y, e.center.heading], device=device
                )
                for e in infered_smooth_centerline_trajectory
            ]
            smoothed_trajectories.append(torch.stack(smooth_rear_axle_trajectory))
            assert not torch.all(
                trajectory[1:, :] == torch.stack(smooth_rear_axle_trajectory)
            ), f"{trajectory}, {torch.stack(smooth_rear_axle_trajectory)}"
    smoothed_trajectories = torch.stack(smoothed_trajectories)
    assert torch.all(
        torch.isfinite(smoothed_trajectories)
    ), f"""
        nans / infs in smoothed_trajectories
        {smoothed_trajectories}
        where input trajectories was
        {trajectories}
    """
    assert (
        smoothed_trajectories.shape == traj_shape
    ), f"{smoothed_trajectories.shape}, {traj_shape}"
    return smoothed_trajectories
