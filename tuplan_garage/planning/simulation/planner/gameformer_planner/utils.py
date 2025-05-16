import numpy as np
import scipy


def annotate_occupancy(occupancy, ego_path, red_light_lane):
    ego_path_red_light = scipy.spatial.distance.cdist(ego_path[:, :2], red_light_lane)

    if len(red_light_lane) < 80:
        pass
    else:
        occupancy[np.any(ego_path_red_light < 0.5, axis=-1)] = 1

    return occupancy


def annotate_speed(ref_path, speed_limit):
    speed = np.ones(len(ref_path)) * speed_limit

    # get the turning point
    turning_idx = np.argmax(np.abs(ref_path[:, 3]) > 1 / 10)

    # set speed limit to 3 m/s for turning
    if turning_idx > 0:
        speed[turning_idx:] = 3

    return speed[:, None]


def wrap_to_pi(theta):
    return (theta + np.pi) % (2 * np.pi) - np.pi


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
