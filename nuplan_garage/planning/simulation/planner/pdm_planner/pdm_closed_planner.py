import gc
import logging
from typing import List, Optional, Type

import numpy as np

from nuplan.planning.simulation.observation.observation_type import Observation
from nuplan.planning.simulation.planner.abstract_planner import PlannerInitialization, PlannerInput
from nuplan_garage.planning.simulation.planner.pdm_planner.abstract_pdm_planner import AbstractPDMPlanner
from nuplan_garage.planning.simulation.planner.pdm_planner.proposal.batch_idm_policy import BatchIDMPolicy
from nuplan_garage.planning.simulation.planner.pdm_planner.utils.route_utils import (
    route_roadblock_correction,
)
from nuplan.planning.simulation.trajectory.abstract_trajectory import AbstractTrajectory


import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning) 

logger = logging.getLogger(__name__)


class PDMClosedPlanner(AbstractPDMPlanner):
    """PDM-Closed planner class."""

    # Inherited property, see superclass.
    requires_scenario: bool = False

    def __init__(
        self,
        idm_policies: BatchIDMPolicy,
        lateral_offsets: Optional[List[float]],
        trajectory_samples: int,
        proposal_samples: int,
        sample_interval: float,
        map_radius: float,
    ):
        """
        Constructor for PDMClosedPlanner
        :param idm_policies: BatchIDMPolicy class
        :param lateral_offsets: centerline offsets for proposals (optional)
        :param trajectory_samples: number of trajectory samples
        :param proposal_samples: number of proposal samples
        :param sample_interval: interval of trajectory/proposal samples
        :param map_radius: radius around ego to consider
        """
        super(PDMClosedPlanner, self).__init__(
            idm_policies,
            lateral_offsets,
            trajectory_samples,
            proposal_samples,
            sample_interval,
            map_radius,
        )

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

    def compute_planner_trajectory(self, current_input: PlannerInput) -> AbstractTrajectory:
        """Inherited, see superclass."""

        gc.disable()
        ego_state, observation = current_input.history.current_state

        # Apply route correction on first iteration (ego_state required)
        if self._iteration == 0:
            route_roadblock_ids = route_roadblock_correction(
                ego_state, self._map_api, self._route_roadblock_dict
            )
            self._load_route_dicts(route_roadblock_ids)

        # 1. Environment forecast and observation update
        self._observation.update(
            ego_state,
            observation,
            current_input.traffic_light_data,
            self._route_lane_dict,
            self._map_api,
        )

        # 2. Centerline extraction and proposal update
        self._update_proposal_manager(ego_state)

        # 3. Generate/Unroll proposals
        proposals_array = self._generator.generate_proposals(
            ego_state, self._observation, self._proposal_manager
        )

        # 4. Simulate proposals
        simulated_proposals_array = self._simulator.simulate_proposals(proposals_array, ego_state)

        # 5. Score proposals
        proposal_scores = self._scorer.score_proposals(
            simulated_proposals_array,
            ego_state,
            self._observation,
            self._centerline,
            self._route_lane_dict,
            self._map_api,
        )

        # 6.a Apply brake if emergency is expected
        trajectory = self._emergency_brake.brake_if_emergency(
            ego_state, proposal_scores, self._scorer
        )

        # 6.b Otherwise, extend and output best proposal
        if trajectory is None:
            trajectory = self._generator.generate_trajectory(np.argmax(proposal_scores))

        self._iteration += 1
        return trajectory
