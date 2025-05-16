import numpy as np
import scipy
from nuplan.common.actor_state.tracked_objects_types import TrackedObjectType
from nuplan.planning.metrics.utils.expert_comparisons import principal_value
from nuplan.planning.simulation.observation.idm.utils import path_to_linestring
from shapely import LineString, Point
from shapely.geometry.base import CAP_STYLE

from tuplan_garage.planning.simulation.planner.dtpp_planner.bezier_path import (
    calc_4points_bezier_path,
)
from tuplan_garage.planning.simulation.planner.gameformer_planner.cubic_spline_planner import (
    calc_spline_course,
)


class LatticePlanner:
    def __init__(self, route_ids, max_len, vehicle_width: float):
        self.target_depth = max_len
        self.candidate_lane_edge_ids = route_ids
        self.max_path_len = max_len
        self.vehicle_width = vehicle_width

    def get_candidate_paths(self, edges):
        """Get candidate paths using depth first search"""
        # get all paths
        paths = []
        for edge in edges:
            paths.extend(self.depth_first_search(edge))

        candidate_paths = {}

        # extract path polyline
        for i, path in enumerate(paths):
            path_polyline = []
            for edge in path:
                path_polyline.extend(edge.baseline_path.discrete_path)

            path_polyline = self.check_path(
                np.array(path_to_linestring(path_polyline).coords)
            )
            dist_to_ego = scipy.spatial.distance.cdist([self.ego_point], path_polyline)
            path_polyline = path_polyline[dist_to_ego.argmin() :]
            if len(path_polyline) < 3:
                continue

            path_len = len(path_polyline) * 0.25
            polyline_heading = self.calculate_path_heading(path_polyline)
            path_polyline = np.stack(
                [path_polyline[:, 0], path_polyline[:, 1], polyline_heading], axis=1
            )
            candidate_paths[i] = (path_len, dist_to_ego.min(), path, path_polyline)

        if len(candidate_paths) == 0:
            return None

        # trim paths by length
        self.path_len = max([v[0] for v in candidate_paths.values()])
        acceptable_path_len = (
            self.max_path_len * 0.2
            if self.path_len > self.max_path_len * 0.2
            else self.path_len
        )
        candidate_paths = {
            k: v for k, v in candidate_paths.items() if v[0] >= acceptable_path_len
        }

        # sort paths by distance to ego
        candidate_paths = sorted(candidate_paths.items(), key=lambda x: x[1][1])

        return candidate_paths

    def get_candidate_edges(self, starting_block, ego_state):
        """Get candidate edges from the starting block"""
        edges = []
        edges_distance = []
        self.ego_point = (ego_state.rear_axle.x, ego_state.rear_axle.y)
        self.num_edges = len(starting_block.interior_edges)

        for edge in starting_block.interior_edges:
            edges_distance.append(edge.polygon.distance(Point(self.ego_point)))
            if edge.polygon.distance(Point(self.ego_point)) < 4:
                edges.append(edge)

        # if no edge is close to ego, use the closest edge
        if len(edges) == 0:
            edges.append(starting_block.interior_edges[np.argmin(edges_distance)])

        return edges

    def plan(self, ego_state, starting_block, observation, traffic_light_data):
        # Get candidate paths
        edges = self.get_candidate_edges(starting_block, ego_state)
        candidate_paths = self.get_candidate_paths(edges)

        if candidate_paths is None:
            return None

        # Get obstacles
        object_types = [
            TrackedObjectType.VEHICLE,
            TrackedObjectType.BARRIER,
            TrackedObjectType.CZONE_SIGN,
            TrackedObjectType.TRAFFIC_CONE,
            TrackedObjectType.GENERIC_OBJECT,
        ]
        objects = observation.tracked_objects.get_tracked_objects_of_types(object_types)

        obstacles = []
        vehicles = []
        for obj in objects:
            if obj.box.geometry.distance(ego_state.car_footprint.geometry) > 30:
                continue

            if obj.tracked_object_type == TrackedObjectType.VEHICLE:
                if obj.velocity.magnitude() < 0.01:
                    obstacles.append(obj.box)
                else:
                    vehicles.append(obj.box)
            else:
                obstacles.append(obj.box)

        # Generate paths using state lattice
        paths = self.generate_paths(ego_state, candidate_paths)

        # disable lane change in large intersections
        if len(traffic_light_data) > 0:
            self._just_stay_current = True
        elif (
            self.num_edges >= 4
            and ego_state.dynamic_car_state.rear_axle_velocity_2d.x <= 3
        ):
            self._just_stay_current = True
        else:
            self._just_stay_current = False

        # Calculate costs and choose the optimal path
        optimal_path = None
        min_cost = np.inf

        for path in paths:
            cost = self.calculate_cost(path, obstacles, vehicles)
            if cost < min_cost:
                min_cost = cost
                optimal_path = path[0]

        # Post-process the path
        ref_path = self.post_process(optimal_path, ego_state)

        return ref_path

    def generate_paths(self, ego_state, paths):
        """Generate paths from state lattice"""
        new_paths = []
        ego_state = (
            ego_state.rear_axle.x,
            ego_state.rear_axle.y,
            ego_state.rear_axle.heading,
        )

        for _, (path_len, dist, path, path_polyline) in paths:
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
                    ego_state[0],
                    ego_state[1],
                    ego_state[2],
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
                new_paths.append((path_polyline, dist, path, path_len))

        return new_paths

    def calculate_cost(self, path, obstacles, vehicles):
        # path curvature
        curvature = self.calculate_path_curvature(path[0][:100])
        curvature = np.max(curvature)

        # lane change
        lane_change = path[1]
        if self._just_stay_current:
            lane_change = 5 * lane_change

        # go to the target lane
        target = self.check_target_lane(
            path[0][:50], path[3], vehicles, width=self.vehicle_width
        )

        # check obstacles
        obstacles = self.check_obstacles(
            path[0][:100], obstacles, width=self.vehicle_width
        )

        # out of boundary
        out_boundary = self.check_out_boundary(
            path[0][:100], path[2], width=self.vehicle_width
        )

        # final cost
        cost = (
            10 * obstacles
            + 2 * out_boundary
            + 1 * lane_change
            + 0.1 * curvature
            - 5 * target
        )

        return cost

    def post_process(self, path, ego_state):
        index = np.arange(0, len(path), 10)
        x = path[:, 0][index]
        y = path[:, 1][index]
        rx, ry, ryaw, rk = calc_spline_course(x, y)
        spline_path = np.stack([rx, ry, ryaw, rk], axis=1)
        ref_path = self.transform_to_ego_frame(spline_path, ego_state)
        ref_path = ref_path[: self.max_path_len * 10]

        return ref_path

    def depth_first_search(self, starting_edge, depth=0):
        if depth >= self.target_depth:
            return [[starting_edge]]
        else:
            traversed_edges = []
            child_edges = [
                edge
                for edge in starting_edge.outgoing_edges
                if edge.id in self.candidate_lane_edge_ids
            ]

            if child_edges:
                for child in child_edges:
                    edge_len = len(child.baseline_path.discrete_path) * 0.25
                    traversed_edges.extend(
                        self.depth_first_search(child, depth + edge_len)
                    )

            if len(traversed_edges) == 0:
                return [[starting_edge]]

            edges_to_return = []

            for edge_seq in traversed_edges:
                edges_to_return.append([starting_edge] + edge_seq)

            return edges_to_return

    def check_target_lane(self, path, path_len, vehicles, width: float):
        if np.abs(path_len - self.path_len) > 5:
            return 0

        expanded_path = LineString(path).buffer((width / 2), cap_style=CAP_STYLE.square)
        min_distance_to_vehicles = np.inf

        for v in vehicles:
            d = expanded_path.distance(v.geometry)
            if d < min_distance_to_vehicles:
                min_distance_to_vehicles = d

        if min_distance_to_vehicles < 5:
            return 0

        return 1

    @staticmethod
    def check_path(path):
        refine_path = [path[0]]

        for i in range(1, path.shape[0]):
            if np.linalg.norm(path[i] - path[i - 1]) < 0.1:
                continue
            else:
                refine_path.append(path[i])

        line = np.array(refine_path)

        return line

    @staticmethod
    def calculate_path_heading(path):
        heading = np.arctan2(path[1:, 1] - path[:-1, 1], path[1:, 0] - path[:-1, 0])
        heading = np.append(heading, heading[-1])

        return heading

    @staticmethod
    def check_obstacles(path, obstacles, width: float):
        expanded_path = LineString(path).buffer((width / 2), cap_style=CAP_STYLE.square)

        for obstacle in obstacles:
            obstacle_polygon = obstacle.geometry
            if expanded_path.intersects(obstacle_polygon):
                return 1

        return 0

    @staticmethod
    def check_out_boundary(polyline, path, width: float):
        line = LineString(polyline).buffer((width / 2), cap_style=CAP_STYLE.square)

        for edge in path:
            left, right = edge.adjacent_edges
            if (left is None and line.intersects(edge.left_boundary.linestring)) or (
                right is None and line.intersects(edge.right_boundary.linestring)
            ):
                return 1

        return 0

    @staticmethod
    def calculate_path_curvature(path):
        dx = np.gradient(path[:, 0])
        dy = np.gradient(path[:, 1])
        d2x = np.gradient(dx)
        d2y = np.gradient(dy)
        curvature = np.abs(dx * d2y - d2x * dy) / (dx**2 + dy**2) ** (3 / 2)

        return curvature

    @staticmethod
    def transform_to_ego_frame(path, ego_state):
        ego_x, ego_y, ego_h = (
            ego_state.rear_axle.x,
            ego_state.rear_axle.y,
            ego_state.rear_axle.heading,
        )
        path_x, path_y, path_h, path_k = path[:, 0], path[:, 1], path[:, 2], path[:, 3]
        ego_path_x = np.cos(ego_h) * (path_x - ego_x) + np.sin(ego_h) * (path_y - ego_y)
        ego_path_y = -np.sin(ego_h) * (path_x - ego_x) + np.cos(ego_h) * (
            path_y - ego_y
        )
        ego_path_h = principal_value(path_h - ego_h)
        ego_path = np.stack([ego_path_x, ego_path_y, ego_path_h, path_k], axis=-1)

        return ego_path
