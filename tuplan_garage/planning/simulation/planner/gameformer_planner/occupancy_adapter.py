import numpy as np
import scipy
from nuplan.planning.metrics.utils.expert_comparisons import principal_value


def occupancy_adpter(
    predictions,
    scores,
    neighbors,
    ref_path,
    max_path_length: int,
    vehicle_width: float,
    future_horizon: int,
):
    best_mode = np.argmax(scores.cpu().numpy(), axis=-1)
    predictions = predictions.cpu().numpy()
    neighbors = neighbors.cpu().numpy()

    best_predictions = [
        predictions[i, best_mode[i], :, :2] for i in range(predictions.shape[0])
    ]
    prediction_F = [transform_to_Frenet(a, ref_path) for a in best_predictions]
    len_path = ref_path.shape[0]
    if len_path < max_path_length * 10:
        ref_path = np.append(
            ref_path,
            np.repeat(
                ref_path[np.newaxis, -1], max_path_length * 10 - len(ref_path), axis=0
            ),
            axis=0,
        )

    time_occupancy = np.stack(
        future_horizon * 10 * [ref_path[:, -1]], axis=0
    )  # (timestep, path_len)

    for t in range(future_horizon * 10):
        for n, a in enumerate(prediction_F):
            if neighbors[n][0] == 0:
                continue

            if a[0][0] <= 0:
                continue

            # intersect threshold
            aw = neighbors[n][7]
            threshold = aw * 0.5 + vehicle_width * 0.5 + 0.3

            # project to the path
            if a[t][0] > 0 and np.abs(a[t][1]) < threshold:
                al = neighbors[n][6]
                backward = 0.5 * al + 3
                forward = 0.5 * al
                os = np.clip(a[t][0] - backward, 0, max_path_length)
                oe = np.clip(a[t][0] + forward, 0, max_path_length)
                time_occupancy[t][int(os * 10) : int(oe * 10)] = 1

        if len_path < max_path_length * 10:
            time_occupancy[t][len_path:] = 1

    time_occupancy = np.reshape(time_occupancy, (future_horizon * 10, -1, 10))
    time_occupancy = np.max(time_occupancy, axis=-1)

    return time_occupancy


def transform_to_Frenet(traj, ref_path):
    distance_to_ref_path = scipy.spatial.distance.cdist(traj[:, :2], ref_path[:, :2])
    frenet_idx = np.argmin(distance_to_ref_path, axis=-1)
    ref_points = ref_path[frenet_idx]
    interval = 0.1

    frenet_s = interval * frenet_idx
    e = np.sign(
        (traj[:, 1] - ref_points[:, 1]) * np.cos(ref_points[:, 2])
        - (traj[:, 0] - ref_points[:, 0]) * np.sin(ref_points[:, 2])
    )
    frenet_l = np.linalg.norm(traj[:, :2] - ref_points[:, :2], axis=-1) * e

    if traj.shape[-1] == 3:
        frenet_h = principal_value(ref_points[:, 2] - traj[:, 2])
        frenet_traj = np.column_stack([frenet_s, frenet_l, frenet_h])
    else:
        frenet_traj = np.column_stack([frenet_s, frenet_l])

    return frenet_traj
