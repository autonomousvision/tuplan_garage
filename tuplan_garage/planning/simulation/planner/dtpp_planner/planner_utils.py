import numpy as np
import scipy


def check_path(path):
    refine_path = [path[0]]

    for i in range(1, path.shape[0]):
        if np.linalg.norm(path[i] - path[i - 1]) < 0.1:
            continue
        else:
            refine_path.append(path[i])

    line = np.array(refine_path)

    return line


def calculate_path_heading(path):
    heading = np.arctan2(path[1:, 1] - path[:-1, 1], path[1:, 0] - path[:-1, 0])
    heading = np.append(heading, heading[-1])

    return heading


def trajectory_smoothing(trajectory):
    x = trajectory[:, 0]
    y = trajectory[:, 1]
    h = trajectory[:, 2]

    window_length = 40
    x = scipy.signal.savgol_filter(x, window_length=window_length, polyorder=3)
    y = scipy.signal.savgol_filter(y, window_length=window_length, polyorder=3)
    h = scipy.signal.savgol_filter(h, window_length=window_length, polyorder=3)

    return np.column_stack([x, y, h])


def wrap_to_pi(theta):
    return (theta + np.pi) % (2 * np.pi) - np.pi
