from typing import List

import numpy as np
import numpy.typing as npt
from nuplan.common.actor_state.state_representation import StateSE2
from shapely import Polygon
from shapely.vectorized import contains

from tuplan_garage.planning.training.preprocessing.feature_builders.route_utils import (
    normalize_angle,
)


def calculate_lane_progress(state_se2_array: npt.NDArray[np.float64]):

    """
    Calculate the cumulative progress of a given path

    :param path: a path consisting of StateSE2 as waypoints
    :return: a cumulative list of progress
    """

    position_diff_xy = state_se2_array[:-1, :2] - state_se2_array[1:, :2]
    progress_diff_norm = np.append(0.0, np.linalg.norm(position_diff_xy, axis=-1))

    return np.cumsum(progress_diff_norm, dtype=np.float64)


def points_in_polygons(
    point: npt.NDArray[np.float64], polygons: List[Polygon]
) -> npt.NDArray[np.bool_]:

    out = np.zeros((len(polygons), len(point)), dtype=bool)
    for i, polygon in enumerate(polygons):
        out[i] = contains(polygon, point[:, 0], point[:, 1])

    return out


def convert_absolute_to_relative_array(
    origin: StateSE2, poses: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:

    theta = -origin.heading
    origin_array = np.array([[origin.x, origin.y, origin.heading]], dtype=np.float64)

    R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])

    points_rel = poses - origin_array
    points_rel[..., :2] = points_rel[..., :2] @ R.T
    points_rel[:, 2] = normalize_angle(points_rel[:, 2])

    return points_rel
