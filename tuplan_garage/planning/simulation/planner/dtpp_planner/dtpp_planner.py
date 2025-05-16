import math
import time

import numpy as np
import torch
from hydra.utils import instantiate
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
from shapely import Point

from tuplan_garage.planning.simulation.planner.dtpp_planner.common_utils import DT, T
from tuplan_garage.planning.simulation.planner.dtpp_planner.obs_adapter import (
    observation_adapter,
)
from tuplan_garage.planning.simulation.planner.dtpp_planner.scenario_tree_prediction import (
    Decoder,
    Encoder,
)
from tuplan_garage.planning.simulation.planner.dtpp_planner.trajectory_tree_planner import (
    TreePlanner,
)
from tuplan_garage.planning.training.modeling.models.dtpp.dtpp_model import DTPPModel


class DTPPPlanner(AbstractPlanner):
    def __init__(self, model_path, model_config):
        self._future_horizon = T  # [s]
        self._step_interval = DT  # [s]
        self._N_points = int(T / DT)
        self._model_path = model_path
        self._model_config = model_config

    def name(self) -> str:
        return "DTPP Planner"

    def observation_type(self):
        return DetectionsTracks

    def initialize(self, initialization: PlannerInitialization):
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._map_api = initialization.map_api
        self._goal = initialization.mission_goal
        self._route_roadblock_ids = initialization.route_roadblock_ids
        self._initialize_route_plan(self._route_roadblock_ids)
        self._initialize_model()
        self._trajectory_planner = TreePlanner(
            self._device, self._encoder, self._decoder
        )

    def _initialize_model(self):
        model: DTPPModel = instantiate(self._model_config)
        model = LightningModuleWrapper.load_from_checkpoint(
            self._model_path, model=model, map_location=self._device
        ).model
        model.eval()

        self._encoder: Encoder = model.encoder
        self._decoder: Decoder = model.decoder

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

    def compute_planner_trajectory(self, current_input: PlannerInput):
        # Extract iteration, history, and traffic light
        iteration = current_input.iteration.index
        history = current_input.history
        traffic_light_data = list(current_input.traffic_light_data)
        ego_state, observation = history.current_state

        # Construct input features
        time.perf_counter()
        features = observation_adapter(
            history,
            traffic_light_data,
            self._map_api,
            self._route_roadblock_ids,
            self._device,
        )

        # Get starting block
        starting_block = None
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

        # Get traffic light lanes
        traffic_light_lanes = []
        for data in traffic_light_data:
            id_ = str(data.lane_connector_id)
            if (
                data.status == TrafficLightStatusType.RED
                and id_ in self._candidate_lane_edge_ids
            ):
                lane_conn = self._map_api.get_map_object(
                    id_, SemanticMapLayer.LANE_CONNECTOR
                )
                traffic_light_lanes.append(lane_conn)

        # Tree policy planner
        try:
            plan = self._trajectory_planner.plan(
                iteration,
                ego_state,
                features,
                starting_block,
                self._route_roadblocks,
                self._candidate_lane_edge_ids,
                traffic_light_lanes,
                observation,
            )
        except Exception as e:
            print("Error in planning")
            print(e)
            plan = np.zeros((self._N_points, 3))

        # Convert relative poses to absolute states and wrap in a trajectory object
        states = transform_predictions_to_states(
            plan, history.ego_states, self._future_horizon, DT
        )
        trajectory = InterpolatedTrajectory(states)

        return trajectory
