import numpy as np
import scipy
import torch

from tuplan_garage.planning.simulation.planner.gameformer_planner.occupancy_adapter import (
    occupancy_adpter,
)
from tuplan_garage.planning.simulation.planner.gameformer_planner.refinement_planner import (
    RefinementPlanner,
)


class TrajectoryPlanner:
    def __init__(
        self,
        max_path_length: int,
        vehicle_width: float,
        vehicle_length: float,
        future_horizon: int,
        step_interval: float,
        device="cpu",
    ):
        self._future_horizon = future_horizon
        self.N = int(future_horizon / step_interval)
        self.step_interval = step_interval
        self.ts = step_interval
        self._max_path_length = max_path_length
        self._vehicle_width = vehicle_width
        self._device = device
        self.planner = RefinementPlanner(
            max_path_length=max_path_length,
            future_horizon=future_horizon,
            step_interval=step_interval,
            vehicle_length=vehicle_length,
            device=device,
        )

    def plan(
        self,
        ego_state,
        ego_state_transformed,
        neighbors_state_transformed,
        predictions,
        plan,
        scores,
        ref_path,
        observation,
    ):
        # Get the plan from the prediction model
        plan = plan[0].cpu().numpy()

        # Get the plan in the reference path
        if ref_path is not None:
            distance_to_ref = scipy.spatial.distance.cdist(plan[:, :2], ref_path[:, :2])
            i = np.argmin(distance_to_ref, axis=1)
            plan = ref_path[i, :3]
            s = np.concatenate([[0], i]) * 0.1
            speed = np.diff(s) / self.step_interval
        else:
            speed = np.diff(plan[:, :2], axis=0) / self.step_interval
            speed = np.linalg.norm(speed, axis=-1)
            speed = np.concatenate([speed, [speed[-1]]])

        # Refine planning
        if ref_path is None:
            pass
        else:
            occupancy = occupancy_adpter(
                predictions[0],
                scores[0, 1:],
                neighbors_state_transformed[0],
                ref_path,
                self._max_path_length,
                self._vehicle_width,
                self._future_horizon,
            )
            ego_plan_ds = torch.from_numpy(speed).float().unsqueeze(0).to(self._device)
            ego_plan_s = torch.from_numpy(s).float().unsqueeze(0).to(self._device)
            ego_state_transformed = ego_state_transformed.to(self._device)
            ref_path = torch.from_numpy(ref_path).unsqueeze(0).to(self._device)
            occupancy = torch.from_numpy(occupancy).unsqueeze(0).to(self._device)

            s, speed = self.planner.plan(
                ego_state_transformed, ego_plan_ds, ego_plan_s, occupancy, ref_path
            )
            s = s.squeeze(0).cpu().numpy()
            speed = speed.squeeze(0).cpu().numpy()

            # Convert to Cartesian trajectory
            ref_path = ref_path.squeeze(0).cpu().numpy()
            i = (s * 10).astype(np.int32).clip(0, len(ref_path) - 1)
            plan = ref_path[i, :3]

        return plan
