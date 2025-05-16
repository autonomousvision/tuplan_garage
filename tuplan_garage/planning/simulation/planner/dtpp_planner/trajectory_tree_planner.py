import itertools
import random
from collections import defaultdict

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import scipy
import torch
from nuplan.common.actor_state.tracked_objects_types import TrackedObjectType
from nuplan.common.actor_state.vehicle_parameters import get_pacifica_parameters
from nuplan.planning.simulation.observation.idm.utils import path_to_linestring
from shapely import LineString, Point
from shapely.geometry.base import CAP_STYLE
from torch.nn.utils.rnn import pad_sequence

from tuplan_garage.planning.simulation.planner.dtpp_planner.bezier_path import (
    calc_4points_bezier_path,
)
from tuplan_garage.planning.simulation.planner.dtpp_planner.common_utils import (
    MAX_LEN,
    WIDTH,
)
from tuplan_garage.planning.simulation.planner.dtpp_planner.path_planner import (
    calc_spline_course,
)
from tuplan_garage.planning.simulation.planner.dtpp_planner.planner_utils import (
    calculate_path_heading,
    check_path,
    trajectory_smoothing,
)
from tuplan_garage.planning.simulation.planner.dtpp_planner.spline_planner import (
    SplinePlanner,
)


class TrajTree:
    def __init__(self, traj, parent, depth):
        self.traj = traj
        self.state = traj[-1, :5]
        self.children = list()
        self.parent = parent
        self.depth = depth
        self.attribute = dict()
        if parent is not None:
            self.total_traj = torch.cat((parent.total_traj, traj), 0)
        else:
            self.total_traj = traj

    def expand(self, child):
        self.children.append(child)

    def expand_set(self, children):
        self.children += children

    def expand_children(self, paths, horizon, speed_limit, planner):
        trajs = planner.gen_trajectories(
            self.state, horizon, paths, speed_limit, self.isroot()
        )
        children = [TrajTree(traj, self, self.depth + 1) for traj in trajs]
        self.expand_set(children)

    def isroot(self):
        return self.parent is None

    def isleaf(self):
        return len(self.children) == 0

    def get_subseq_trajs(self):
        return [child.traj for child in self.children]

    def get_all_leaves(self, leaf_set=[]):
        if self.isleaf():
            print(self.state)
            leaf_set.append(self)
        else:
            for child in self.children:
                leaf_set = child.get_all_leaves(leaf_set)

        return leaf_set

    @staticmethod
    def get_children(obj):
        if isinstance(obj, TrajTree):
            return obj.children

        elif isinstance(obj, list):
            children = [node.children for node in obj]
            children = list(itertools.chain.from_iterable(children))
            return children

        else:
            raise TypeError("obj must be a TrajTree or a list")

    def plot_tree(self, ax=None, msize=12):
        if ax is None:
            fig, ax = plt.subplots(figsize=(20, 10))
        state = self.state.cpu().detach().numpy()

        ax.plot(state[0], state[1], marker="o", color="b", markersize=msize)

        if self.traj.shape[0] > 1:
            if self.parent is not None:
                traj_l = torch.cat((self.parent.traj[-1:], self.traj), 0)
                traj = traj_l.cpu().detach().numpy()
            else:
                traj = self.traj.cpu().detach().numpy()

            ax.plot(traj[:, 0], traj[:, 1], color="k")

        for child in self.children:
            child.plot_tree(ax)

        return ax

    @staticmethod
    def get_children_index_torch(nodes):
        indices = dict()
        for depth, nodes_d in nodes.items():
            if depth + 1 in nodes:
                childs_d = nodes[depth + 1]
                indices_d = list()
                for node in nodes_d:
                    indices_d.append(
                        torch.tensor([childs_d.index(child) for child in node.children])
                    )
                indices[depth] = pad_sequence(
                    indices_d, batch_first=True, padding_value=-1
                )

        return indices

    @staticmethod
    def get_nodes_by_level(obj, depth, nodes=None, trim_short_branch=True):
        assert obj.depth <= depth
        if nodes is None:
            nodes = defaultdict(lambda: list())

        if obj.depth == depth:
            nodes[depth].append(obj)

            return nodes, True
        else:
            if obj.isleaf():
                return nodes, False
            else:
                flag = False
                children_flags = dict()
                for child in obj.children:
                    nodes, child_flag = TrajTree.get_nodes_by_level(child, depth, nodes)
                    children_flags[child] = child_flag
                    flag = flag or child_flag

                if trim_short_branch:
                    obj.children = [
                        child for child in obj.children if children_flags[child]
                    ]
                if flag:
                    nodes[obj.depth].append(obj)

                return nodes, flag


class TreePlanner:
    def __init__(
        self, device, encoder, decoder, n_candidates_expand=5, n_candidates_max=30
    ):
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        self.max_path_len = 120  # [m]
        self.target_depth = MAX_LEN  # [m]
        self.target_speed = 13  # [m/s]
        self.horizon = 8  # [s]
        self.first_stage_horizon = 3  # [s]
        self.n_candidates_expand = n_candidates_expand  # second stage
        self.n_candidates_max = n_candidates_max  # max number of candidates
        self.planner = SplinePlanner(self.first_stage_horizon, self.horizon)

    def get_candidate_paths(self, edges):
        # get all paths
        paths = []
        for edge in edges:
            paths.extend(self.depth_first_search(edge))

        # extract path polyline
        candidate_paths = []

        for i, path in enumerate(paths):
            path_polyline = []
            for edge in path:
                path_polyline.extend(edge.baseline_path.discrete_path)

            path_polyline = check_path(
                np.array(path_to_linestring(path_polyline).coords)
            )
            dist_to_ego = scipy.spatial.distance.cdist([self.ego_point], path_polyline)
            path_polyline = path_polyline[dist_to_ego.argmin() :]
            if len(path_polyline) < 3:
                continue

            path_len = len(path_polyline) * 0.25
            polyline_heading = calculate_path_heading(path_polyline)
            path_polyline = np.stack(
                [path_polyline[:, 0], path_polyline[:, 1], polyline_heading], axis=1
            )
            candidate_paths.append((path_len, dist_to_ego.min(), path_polyline))

        # trim paths by length
        max_path_len = max([v[0] for v in candidate_paths])
        acceptable_path_len = (
            MAX_LEN / 2 if max_path_len > MAX_LEN / 2 else max_path_len
        )
        paths = [v for v in candidate_paths if v[0] >= acceptable_path_len]

        return paths

    def get_candidate_edges(self, starting_block):
        edges = []
        edges_distance = []
        self.ego_point = (self.ego_state.rear_axle.x, self.ego_state.rear_axle.y)

        for edge in starting_block.interior_edges:
            edges_distance.append(edge.polygon.distance(Point(self.ego_point)))
            if edge.polygon.distance(Point(self.ego_point)) < 4:
                edges.append(edge)

        # if no edge is close to ego, use the closest edge
        if len(edges) == 0:
            edges.append(starting_block.interior_edges[np.argmin(edges_distance)])

        return edges

    def generate_paths(self, routes):
        ego_state = (
            self.ego_state.rear_axle.x,
            self.ego_state.rear_axle.y,
            self.ego_state.rear_axle.heading,
        )

        # generate paths
        new_paths = []
        path_distance = []
        for (path_len, dist, path_polyline) in routes:
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
                new_paths.append(path_polyline)
                path_distance.append(dist)

        # evaluate paths
        candiate_paths = {}
        for path, dist in zip(new_paths, path_distance):
            cost = self.calculate_cost(path, dist)
            candiate_paths[cost] = path

        # sort paths by cost
        candidate_paths = []
        for cost in sorted(candiate_paths.keys())[:3]:
            path = candiate_paths[cost]
            path = self.post_process(path)
            candidate_paths.append(path)

        return candidate_paths

    def calculate_cost(self, path, dist):
        # path curvature
        curvature = self.calculate_path_curvature(path[0:100])
        curvature = np.max(curvature)

        # lane change
        lane_change = dist

        # check obstacles
        obstacles = self.check_obstacles(path[0:100:10], self.obstacles)

        # final cost
        cost = 10 * obstacles + 1 * lane_change + 0.1 * curvature

        return cost

    def post_process(self, path):
        path = self.transform_to_ego_frame(path)
        index = np.arange(0, len(path), 10)
        x = path[:, 0][index]
        y = path[:, 1][index]

        # spline interpolation
        rx, ry, ryaw, rk = calc_spline_course(x, y)
        spline_path = np.stack([rx, ry, ryaw, rk], axis=1)
        ref_path = spline_path[: self.max_path_len * 10]

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

    @staticmethod
    def calculate_path_curvature(path):
        dx = np.gradient(path[:, 0])
        dy = np.gradient(path[:, 1])
        d2x = np.gradient(dx)
        d2y = np.gradient(dy)
        curvature = np.abs(dx * d2y - d2x * dy) / (dx**2 + dy**2) ** (3 / 2)

        return curvature

    @staticmethod
    def check_obstacles(path, obstacles):
        expanded_path = LineString(path).buffer((WIDTH / 2), cap_style=CAP_STYLE.square)

        for obstacle in obstacles:
            obstacle_polygon = obstacle.geometry
            if expanded_path.intersects(obstacle_polygon):
                return 1

        return 0

    def predict(self, encoder_outputs, traj_inputs, agent_states, timesteps):
        ego_trajs = torch.zeros((self.n_candidates_max, self.horizon * 10, 6)).to(
            self.device
        )
        for i, traj in enumerate(traj_inputs):
            ego_trajs[i, : len(traj)] = traj[..., :6].float()

        ego_trajs = ego_trajs.unsqueeze(0)
        agent_trajs, scores, _, _ = self.decoder(
            encoder_outputs, ego_trajs, agent_states, timesteps
        )

        return agent_trajs, scores

    def transform_to_ego_frame(self, path):
        x = path[:, 0] - self.ego_state.rear_axle.x
        y = path[:, 1] - self.ego_state.rear_axle.y
        x_e = x * np.cos(-self.ego_state.rear_axle.heading) - y * np.sin(
            -self.ego_state.rear_axle.heading
        )
        y_e = x * np.sin(-self.ego_state.rear_axle.heading) + y * np.cos(
            -self.ego_state.rear_axle.heading
        )
        path = np.column_stack([x_e, y_e])

        return path

    def plan(
        self,
        iteration,
        ego_state,
        env_inputs,
        starting_block,
        route_roadblocks,
        candidate_lane_edge_ids,
        traffic_light,
        observation,
        debug=False,
    ):
        # get environment information
        self.ego_state = ego_state
        self.candidate_lane_edge_ids = candidate_lane_edge_ids
        self.route_roadblocks = route_roadblocks
        self.traffic_light = traffic_light
        object_types = [
            TrackedObjectType.VEHICLE,
            TrackedObjectType.BARRIER,
            TrackedObjectType.CZONE_SIGN,
            TrackedObjectType.TRAFFIC_CONE,
            TrackedObjectType.GENERIC_OBJECT,
        ]
        objects = observation.tracked_objects.get_tracked_objects_of_types(object_types)
        self.obstacles = []
        for obj in objects:
            if obj.tracked_object_type == TrackedObjectType.VEHICLE:
                if obj.velocity.magnitude() < 0.1:
                    self.obstacles.append(obj.box)
            else:
                self.obstacles.append(obj.box)

        # initial tree (root node)
        # x, y, heading, velocity, acceleration, curvature, time
        state = torch.tensor(
            [
                [
                    0,
                    0,
                    0,  # x, y, heading
                    ego_state.dynamic_car_state.rear_axle_velocity_2d.x,
                    ego_state.dynamic_car_state.rear_axle_acceleration_2d.x,
                    0,
                    0,
                ]
            ],
            dtype=torch.float32,
        )
        tree = TrajTree(state, None, 0)

        # environment encoding
        encoder_outputs = self.encoder(env_inputs)
        agent_states = env_inputs["neighbor_agents_past"]

        # get candidate map lanes
        edges = self.get_candidate_edges(starting_block)
        candidate_paths = self.get_candidate_paths(edges)
        paths = self.generate_paths(candidate_paths)
        self.speed_limit = edges[0].speed_limit_mps or self.target_speed

        # expand tree
        tree.expand_children(
            paths, self.first_stage_horizon, self.speed_limit, self.planner
        )
        leaves = TrajTree.get_children(tree)

        # query the model
        parent_scores = {}
        trajs = [leaf.total_traj[1:] for leaf in leaves]
        agent_trajectories, scores = self.predict(
            encoder_outputs, trajs, agent_states, self.first_stage_horizon * 10
        )
        indices = torch.topk(scores, self.n_candidates_expand)[1][0]
        pruned_leaves = []
        for i in indices:
            if i.item() < len(leaves):
                pruned_leaves.append(leaves[i])
                parent_scores[leaves[i]] = scores[0, i].item()

        # expand leaves with higher scores
        for leaf in pruned_leaves:
            leaf.expand_children(
                paths,
                self.horizon - self.first_stage_horizon,
                self.speed_limit,
                self.planner,
            )

        # get all leaves
        leaves = TrajTree.get_children(leaves)
        if len(leaves) > self.n_candidates_max:
            leaves = random.sample(leaves, self.n_candidates_max)

        # query the model
        trajs = [leaf.total_traj[1:] for leaf in leaves]
        agent_trajectories, scores = self.predict(
            encoder_outputs, trajs, agent_states, self.horizon * 10
        )

        # calculate scores
        children_scores = {}
        for i, leaf in enumerate(leaves):
            if leaf.parent in children_scores:
                children_scores[leaf.parent].append(scores[0, i].item())
            else:
                children_scores[leaf.parent] = [scores[0, i].item()]

        # get the best parent
        best_parent = None
        best_child_index = None
        best_score = -np.inf
        for parent in parent_scores.keys():
            score = parent_scores[parent] + np.max(children_scores[parent])
            if score > best_score:
                best_parent = parent
                best_score = score
                best_child_index = np.argmax(children_scores[parent])

        # get the best trajectory
        best_traj = best_parent.children[best_child_index].total_traj[1:, :3]

        # plot
        if debug:
            for i, traj in enumerate(trajs):
                self.plot(iteration, env_inputs, traj, agent_trajectories[0, i])

        return best_traj

    def plot(self, iteration, env_inputs, ego_future, agents_future):
        fig = plt.gcf()
        dpi = 100
        size_inches = 800 / dpi
        fig.set_size_inches([size_inches, size_inches])
        fig.set_dpi(dpi)
        fig.set_tight_layout(True)

        # plot map
        map_lanes = env_inputs["map_lanes"][0]
        for i in range(map_lanes.shape[0]):
            lane = map_lanes[i].cpu().numpy()
            if lane[0, 0] != 0:
                plt.plot(lane[:, 0], lane[:, 1], color="gray", linewidth=20, zorder=1)
                plt.plot(lane[:, 0], lane[:, 1], "k--", linewidth=1, zorder=2)

        map_crosswalks = env_inputs["map_crosswalks"][0]
        for crosswalk in map_crosswalks:
            pts = crosswalk.cpu().numpy()
            plt.plot(pts[:, 0], pts[:, 1], "b:", linewidth=2)

        # plot ego
        front_length = get_pacifica_parameters().front_length
        rear_length = get_pacifica_parameters().rear_length
        width = get_pacifica_parameters().width
        rect = plt.Rectangle(
            (0 - rear_length, 0 - width / 2),
            front_length + rear_length,
            width,
            linewidth=2,
            color="r",
            alpha=0.9,
            zorder=3,
        )
        plt.gca().add_patch(rect)

        # plot agents
        agents = env_inputs["neighbor_agents_past"][0]
        for agent in agents:
            agent = agent[-1].cpu().numpy()
            if agent[0] != 0:
                rect = plt.Rectangle(
                    (agent[0] - agent[6] / 2, agent[1] - agent[7] / 2),
                    agent[6],
                    agent[7],
                    linewidth=2,
                    color="m",
                    alpha=0.9,
                    zorder=3,
                    transform=mpl.transforms.Affine2D().rotate_around(
                        *(agent[0], agent[1]), agent[2]
                    )
                    + plt.gca().transData,
                )
                plt.gca().add_patch(rect)

        # plot ego and agents future trajectories
        ego = ego_future.cpu().numpy()
        agents = agents_future.cpu().numpy()
        plt.plot(ego[:, 0], ego[:, 1], color="r", linewidth=3)
        plt.gca().add_patch(
            plt.Circle((ego[29, 0], ego[29, 1]), 0.5, color="r", zorder=4)
        )
        plt.gca().add_patch(
            plt.Circle((ego[79, 0], ego[79, 1]), 0.5, color="r", zorder=4)
        )

        for agent in agents:
            if np.abs(agent[0, 0]) > 1:
                agent = trajectory_smoothing(agent)
                plt.plot(agent[:, 0], agent[:, 1], color="m", linewidth=3)
                plt.gca().add_patch(
                    plt.Circle((agent[29, 0], agent[29, 1]), 0.5, color="m", zorder=4)
                )
                plt.gca().add_patch(
                    plt.Circle((agent[79, 0], agent[79, 1]), 0.5, color="m", zorder=4)
                )

        # plot
        plt.gca().margins(0)
        plt.gca().set_aspect("equal")
        plt.gca().axes.get_yaxis().set_visible(False)
        plt.gca().axes.get_xaxis().set_visible(False)
        plt.gca().axis([-50, 50, -50, 50])
        plt.show()
