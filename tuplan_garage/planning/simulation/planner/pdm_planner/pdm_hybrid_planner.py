import gc
import logging
import warnings
from typing import List, Optional, Type, cast

import torch
from nuplan.common.actor_state.ego_state import EgoState
from nuplan.common.actor_state.state_representation import TimeDuration, TimePoint
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
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
from nuplan.planning.training.modeling.lightning_module_wrapper import (
    LightningModuleWrapper,
)
from nuplan.planning.training.modeling.torch_module_wrapper import TorchModuleWrapper
from nuplan.planning.utils.serialization.scene import Trajectory

from tuplan_garage.planning.simulation.planner.pdm_planner.abstract_pdm_closed_planner import (
    AbstractPDMClosedPlanner,
)
from tuplan_garage.planning.simulation.planner.pdm_planner.observation.pdm_observation_utils import (
    get_drivable_area_map,
)
from tuplan_garage.planning.simulation.planner.pdm_planner.proposal.batch_idm_policy import (
    BatchIDMPolicy,
)
from tuplan_garage.planning.simulation.planner.pdm_planner.utils.pdm_feature_utils import (
    create_pdm_feature,
)
from tuplan_garage.planning.simulation.planner.pdm_planner.utils.pdm_path import PDMPath

warnings.filterwarnings("ignore", category=RuntimeWarning)

logger = logging.getLogger(__name__)


class PDMHybridPlanner(AbstractPDMClosedPlanner):
    """PDM-Closed planner class."""

    # Inherited property, see superclass.
    requires_scenario: bool = False

    def __init__(
        self,
        trajectory_sampling: TrajectorySampling,
        proposal_sampling: TrajectorySampling,
        idm_policies: BatchIDMPolicy,
        lateral_offsets: Optional[List[float]],
        map_radius: float,
        model: TorchModuleWrapper,
        correction_horizon: float,
        checkpoint_path: str,
    ):
        """
        Constructor for PDM-Hybrid.
        :param trajectory_sampling: sampling parameters for final trajectory
        :param proposal_sampling: sampling parameters for proposals
        :param idm_policies: BatchIDMPolicy class
        :param lateral_offsets: centerline offsets for proposals (optional)
        :param map_radius: radius around ego to consider
        :param model: torch model
        :param correction_horizon: time to apply open-loop correction [s]
        :param checkpoint_path: path to checkpoint for model as string
        """

        super(PDMHybridPlanner, self).__init__(
            trajectory_sampling,
            proposal_sampling,
            idm_policies,
            lateral_offsets,
            map_radius,
        )

        self._device = "cpu"

        self._model = LightningModuleWrapper.load_from_checkpoint(
            checkpoint_path,
            model=model,
            map_location=self._device,
        ).model
        self._model.eval()
        torch.set_grad_enabled(False)

        self._correction_horizon: float = correction_horizon  # [s]

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

        # trajectory of PDM-Closed
        closed_loop_trajectory = self._get_closed_loop_trajectory(current_input)
        uncorrected_states = closed_loop_trajectory.get_sampled_trajectory()

        # trajectory of PDM-Offset
        pdm_feature = create_pdm_feature(
            self._model,
            current_input,
            self._centerline,
            closed_loop_trajectory,
            self._device,
        )
        predictions = self._model.forward({"pdm_features": pdm_feature})
        trajectory_data = (
            cast(Trajectory, predictions["trajectory"]).data.cpu().detach().numpy()[0]
        )
        corrected_states = transform_predictions_to_states(
            trajectory_data,
            current_input.history.ego_states,
            self._model.trajectory_sampling.time_horizon,
            self._model.trajectory_sampling.step_time,
        )

        # apply correction by fusing
        trajectory = self._apply_trajectory_correction(
            uncorrected_states, corrected_states
        )
        self._iteration += 1
        return trajectory

    def _apply_trajectory_correction(
        self,
        uncorrected_states: List[EgoState],
        corrected_states: List[EgoState],
    ) -> InterpolatedTrajectory:
        """
        Applies open-loop correction and fuses to a single trajectory.
        :param uncorrected_states: ego vehicles states of PDM-Closed trajectory
        :param corrected_states: ego-vehicles states of PDM-Offset trajectory
        :return: trajectory after applying correction.
        """

        # split trajectory
        uncorrected_duration: TimeDuration = TimeDuration.from_s(
            self._correction_horizon
        )
        cutting_time_point: TimePoint = (
            uncorrected_states[0].time_point + uncorrected_duration
        )

        uncorrected_split = [
            ego_state
            for ego_state in uncorrected_states
            if ego_state.time_point <= cutting_time_point
        ]

        corrected_split = [
            ego_state
            for ego_state in corrected_states
            if ego_state.time_point > cutting_time_point
        ]

        return InterpolatedTrajectory(uncorrected_split + corrected_split)
