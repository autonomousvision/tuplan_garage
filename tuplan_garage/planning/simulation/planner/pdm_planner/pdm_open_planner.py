import gc
import logging
import warnings
from typing import Type, cast

import torch
from nuplan.planning.simulation.observation.observation_type import (
    DetectionsTracks,
    Observation,
)
from nuplan.planning.simulation.planner.abstract_planner import (
    PlannerInitialization,
    PlannerInput,
)
from nuplan.planning.simulation.planner.ml_planner.transform_utils import (
    transform_predictions_to_states,
)
from nuplan.planning.simulation.trajectory.abstract_trajectory import AbstractTrajectory
from nuplan.planning.simulation.trajectory.interpolated_trajectory import (
    InterpolatedTrajectory,
)
from nuplan.planning.training.modeling.lightning_module_wrapper import (
    LightningModuleWrapper,
)
from nuplan.planning.training.modeling.torch_module_wrapper import TorchModuleWrapper
from nuplan.planning.utils.serialization.scene import Trajectory

from tuplan_garage.planning.simulation.planner.pdm_planner.abstract_pdm_planner import (
    AbstractPDMPlanner,
)
from tuplan_garage.planning.simulation.planner.pdm_planner.observation.pdm_observation_utils import (
    get_drivable_area_map,
)
from tuplan_garage.planning.simulation.planner.pdm_planner.utils.pdm_feature_utils import (
    create_pdm_feature,
)
from tuplan_garage.planning.simulation.planner.pdm_planner.utils.pdm_path import PDMPath

warnings.filterwarnings("ignore", category=RuntimeWarning)

logger = logging.getLogger(__name__)


class PDMOpenPlanner(AbstractPDMPlanner):
    """PDM-Open planner class."""

    # Inherited property, see superclass.
    requires_scenario: bool = False

    def __init__(
        self,
        model: TorchModuleWrapper,
        checkpoint_path: str,
        map_radius: float,
    ):
        """
        Constructor for PDMOpenPlanner
        :param model: torch model
        :param checkpoint_path: path to checkpoint for model as string
        :param map_radius: radius around ego to consider
        """
        super(PDMOpenPlanner, self).__init__(map_radius)

        self._device = "cpu"

        self._model = LightningModuleWrapper.load_from_checkpoint(
            checkpoint_path,
            model=model,
            map_location=self._device,
        ).model

        self._model.eval()
        torch.set_grad_enabled(False)

    def initialize(self, initialization: PlannerInitialization) -> None:
        """Inherited, see superclass."""
        self._iteration = 0
        self._map_api = initialization.map_api
        self._load_route_dicts(initialization.route_roadblock_ids)
        gc.collect()

    def name(self) -> str:
        """Inherited, see superclass."""
        return self.__class__.__name__

    def observation_type(self) -> Type[Observation]:
        """Inherited, see superclass."""
        return DetectionsTracks  # type: ignore

    def compute_planner_trajectory(
        self, current_input: PlannerInput
    ) -> AbstractTrajectory:
        """Inherited, see superclass."""

        gc.disable()
        ego_state, _ = current_input.history.current_state

        # Apply route correction on first iteration (ego_state required)
        if self._iteration == 0:
            self._route_roadblock_correction(ego_state)

        # Update/Create drivable area polygon map
        self._drivable_area_map = get_drivable_area_map(
            self._map_api, ego_state, self._map_radius
        )

        # Create centerline
        current_lane = self._get_starting_lane(ego_state)
        self._centerline = PDMPath(self._get_discrete_centerline(current_lane))

        # feature building & model forward
        pdm_feature = create_pdm_feature(
            self._model, current_input, self._centerline, None, self._device
        )
        predictions = self._model.forward({"pdm_features": pdm_feature})

        # convert to absolute
        trajectory_data = cast(Trajectory, predictions["trajectory"]).data
        trajectory = trajectory_data.cpu().detach().numpy()[0]

        trajectory = InterpolatedTrajectory(
            transform_predictions_to_states(
                trajectory,
                current_input.history.ego_states,
                self._model.trajectory_sampling.time_horizon,
                self._model.trajectory_sampling.step_time,
            )
        )

        self._iteration += 1
        return trajectory
