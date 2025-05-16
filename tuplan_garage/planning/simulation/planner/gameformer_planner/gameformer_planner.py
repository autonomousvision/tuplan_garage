import math
import warnings

import numpy as np
import torch
from hydra.utils import instantiate
from nuplan.common.actor_state.vehicle_parameters import get_pacifica_parameters
from nuplan.common.maps.maps_datatypes import SemanticMapLayer, TrafficLightStatusType
from nuplan.planning.simulation.observation.observation_type import DetectionsTracks
from nuplan.planning.simulation.planner.abstract_planner import (
    AbstractPlanner,
    PlannerInitialization,
    PlannerInput,
)
from nuplan.planning.simulation.planner.ml_planner.transform_utils import (
    transform_predictions_to_states,
)
from nuplan.planning.simulation.trajectory.interpolated_trajectory import (
    InterpolatedTrajectory,
)
from nuplan.planning.training.modeling.lightning_module_wrapper import (
    LightningModuleWrapper,
)
from omegaconf import DictConfig
from shapely import Point

from tuplan_garage.planning.simulation.planner.dtpp_planner.obs_adapter import (
    observation_adapter,
)
from tuplan_garage.planning.simulation.planner.gameformer_planner.state_lattice_path_planner import (
    LatticePlanner,
)
from tuplan_garage.planning.simulation.planner.gameformer_planner.trajectory_planner import (
    TrajectoryPlanner,
)
from tuplan_garage.planning.simulation.planner.gameformer_planner.utils import (
    annotate_occupancy,
    annotate_speed,
    transform_to_ego_frame,
)
from tuplan_garage.planning.training.modeling.models.gameformer.gameformer_model import (
    GameFormer,
)
from tuplan_garage.planning.training.preprocessing.features.gameformer.gameformer_features import (
    GameFormerFeatures,
)

warnings.filterwarnings("error", message=".*Converting mask without.*")


class GameFormerPlanner(AbstractPlanner):
    def __init__(
        self,
        checkpoint_path: str,
        model_config: DictConfig,
        max_path_length: int = 120,
        future_horizon: int = 8,
        step_interval: float = 0.1,
        target_speed: float = 13.0,
        vehicle_width: float = None,
        vehicle_length: float = None,
    ):
        # MAX_LEN = 120 # [m] max length of the path
        self._max_path_length = max_path_length  # [m]
        # T = 8 # [s] planning horizon
        self._future_horizon = future_horizon  # [s]
        # DT = 0.1 # [s] time interval
        self._step_interval = step_interval  # [s]
        self._target_speed = target_speed  # [m/s]
        self._N_points = int(future_horizon / step_interval)
        if vehicle_width is None:
            self._vehicle_width = get_pacifica_parameters().width
        else:
            self._vehicle_width = vehicle_width
        if vehicle_length is None:
            self._vehicle_length = get_pacifica_parameters().front_length
        else:
            self._vehicle_length = vehicle_length
        self._checkpoint_path = checkpoint_path
        self._model_config = model_config

    def name(self) -> str:
        return "GameFormer Planner"

    def observation_type(self):
        return DetectionsTracks

    def initialize(self, initialization: PlannerInitialization):
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self._map_api = initialization.map_api
        self._goal = initialization.mission_goal
        self._route_roadblock_ids = initialization.route_roadblock_ids
        self._initialize_route_plan(self._route_roadblock_ids)
        self._initialize_model()
        self._trajectory_planner = TrajectoryPlanner(
            max_path_length=self._max_path_length,
            future_horizon=self._future_horizon,
            step_interval=self._step_interval,
            vehicle_width=self._vehicle_width,
            vehicle_length=self._vehicle_length,
        )
        self._path_planner = LatticePlanner(
            self._candidate_lane_edge_ids,
            self._max_path_length,
            vehicle_width=self._vehicle_width,
        )

    def _initialize_model(self):
        # Load trained model
        model: GameFormer = instantiate(self._model_config)
        self._model = LightningModuleWrapper.load_from_checkpoint(
            self._checkpoint_path, model=model, map_location=self._device
        ).model
        self._model.eval()

    def _initialize_route_plan(self, route_roadblock_ids):
        self._route_roadblocks = []

        for id_ in route_roadblock_ids:
            block = self._map_api.get_map_object(id_, SemanticMapLayer.ROADBLOCK)
            block = block or self._map_api.get_map_object(
                id_, SemanticMapLayer.ROADBLOCK_CONNECTOR
            )
            self._route_roadblocks.append(block)

        self._candidate_lane_edge_ids = [
            edge.id
            for block in self._route_roadblocks
            if block
            for edge in block.interior_edges
        ]

    def _get_reference_path(self, ego_state, traffic_light_data, observation):
        # Get starting block
        starting_block = None
        min_target_speed = 3
        max_target_speed = 15
        cur_point = (ego_state.rear_axle.x, ego_state.rear_axle.y)
        closest_distance = math.inf

        for block in self._route_roadblocks:
            for edge in block.interior_edges:
                distance = edge.polygon.distance(Point(cur_point))
                if distance < closest_distance:
                    starting_block = block
                    closest_distance = distance

            if np.isclose(closest_distance, 0):
                break

        # In case the ego vehicle is not on the route, return None
        if closest_distance > 5:
            return None

        # Get reference path, handle exception
        try:
            ref_path = self._path_planner.plan(
                ego_state, starting_block, observation, traffic_light_data
            )
        except Exception:
            ref_path = None

        if ref_path is None:
            return None

        # Annotate red light to occupancy
        occupancy = np.zeros(shape=(ref_path.shape[0], 1))
        for data in traffic_light_data:
            id_ = str(data.lane_connector_id)
            if (
                data.status == TrafficLightStatusType.RED
                and id_ in self._candidate_lane_edge_ids
            ):
                lane_conn = self._map_api.get_map_object(
                    id_, SemanticMapLayer.LANE_CONNECTOR
                )
                conn_path = lane_conn.baseline_path.discrete_path
                conn_path = np.array([[p.x, p.y] for p in conn_path])
                red_light_lane = transform_to_ego_frame(conn_path, ego_state)
                occupancy = annotate_occupancy(occupancy, ref_path, red_light_lane)

        # Annotate max speed along the reference path
        target_speed = (
            starting_block.interior_edges[0].speed_limit_mps or self._target_speed
        )
        target_speed = np.clip(target_speed, min_target_speed, max_target_speed)
        max_speed = annotate_speed(ref_path, target_speed)

        # Finalize reference path
        ref_path = np.concatenate(
            [ref_path, max_speed, occupancy], axis=-1
        )  # [x, y, theta, k, v_max, occupancy]
        if len(ref_path) < self._max_path_length * 10:
            ref_path = np.append(
                ref_path,
                np.repeat(
                    ref_path[np.newaxis, -1],
                    self._max_path_length * 10 - len(ref_path),
                    axis=0,
                ),
                axis=0,
            )

        return ref_path.astype(np.float32)

    def _get_prediction(self, features: GameFormerFeatures):
        out = self._model({"gameformer_features": features})
        predictions = out["gameformer_targets"].decoder_outputs
        plan = out["trajectory"]
        K = len(predictions) // 2 - 1
        final_predictions = predictions[f"level_{K}_interactions"][:, 1:]
        final_scores = predictions[f"level_{K}_scores"]
        ego_current = features.ego_agent_past[:, -1]
        neighbors_current = features.neighbor_agents_past[:, :, -1]

        return plan, final_predictions, final_scores, ego_current, neighbors_current

    def _plan(self, ego_state, history, traffic_light_data, observation):
        # Construct input features
        features = observation_adapter(
            history,
            traffic_light_data,
            self._map_api,
            self._route_roadblock_ids,
            self._device,
        )
        features = GameFormerFeatures(
            ego_agent_past=features["ego_agent_past"],
            ego_agent_future=None,
            neighbor_agents_past=features["neighbor_agents_past"],
            neighbor_agents_future=None,
            crosswalks=features["map_crosswalks"],
            route_lanes=features["route_lanes"],
            lanes=features["map_lanes"],
        )

        # Get reference path
        ref_path = self._get_reference_path(ego_state, traffic_light_data, observation)

        # Infer prediction model
        with torch.no_grad():
            (
                plan,
                predictions,
                scores,
                ego_state_transformed,
                neighbors_state_transformed,
            ) = self._get_prediction(features)

        # Trajectory refinement
        with torch.no_grad():
            plan = self._trajectory_planner.plan(
                ego_state,
                ego_state_transformed,
                neighbors_state_transformed,
                predictions,
                plan.data,
                scores,
                ref_path,
                observation,
            )

        states = transform_predictions_to_states(
            plan, history.ego_states, self._future_horizon, self._step_interval
        )
        trajectory = InterpolatedTrajectory(states)

        return trajectory

    def compute_planner_trajectory(self, current_input: PlannerInput):
        history = current_input.history
        traffic_light_data = list(current_input.traffic_light_data)
        ego_state, observation = history.current_state
        trajectory = self._plan(ego_state, history, traffic_light_data, observation)

        return trajectory
