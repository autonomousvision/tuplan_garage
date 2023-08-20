from typing import List, Optional, Tuple

import numpy as np
from nuplan.planning.scenario_builder.abstract_scenario import AbstractScenario
from nuplan.planning.training.data_augmentation.kinematic_agent_augmentation import (
    KinematicAgentAugmentor,
)
from nuplan.planning.training.modeling.types import FeaturesType, TargetsType
from nuplan.planning.training.preprocessing.features.agents import Agents


class PGPKinematicAgentAugmentor(KinematicAgentAugmentor):
    def __init__(
        self,
        trajectory_length: int,
        dt: float,
        mean: List[float],
        std: List[float],
        low: List[float],
        high: List[float],
        augment_prob: float,
        use_uniform_noise: bool = False,
    ) -> None:
        """
        Initialize the augmentor.
        :param trajectory_length: Length of trajectory to be augmented.
        :param dt: Time interval between trajecotry points.
        :param mean: Parameter to set mean vector of the Gaussian noise on [x, y, yaw].
        :param std: Parameter to set standard deviation vector of the Gaussian noise on [x, y, yaw].
        :param low: Parameter to set lower bound vector of the Uniform noise on [x, y, yaw]. Used only if use_uniform_noise == True.
        :param high: Parameter to set upper bound vector of the Uniform noise on [x, y, yaw]. Used only if use_uniform_noise == True.
        :param augment_prob: Probability between 0 and 1 of applying the data augmentation.
        :param use_uniform_noise: Parameter to decide to use uniform noise instead of gaussian noise if true.
        """
        super().__init__(
            trajectory_length=trajectory_length,
            dt=dt,
            mean=mean,
            std=std,
            low=low,
            high=high,
            augment_prob=augment_prob,
            use_uniform_noise=use_uniform_noise,
        )

    def augment(
        self,
        features: FeaturesType,
        targets: TargetsType,
        scenario: Optional[AbstractScenario] = None,
    ) -> Tuple[FeaturesType, TargetsType]:
        """Inherited, see superclass."""
        if np.random.rand() >= self._augment_prob:
            return features, targets
        # pgp has the following attributes: x,y,v,a,theta_dot
        # agents has the following attributes: x,y,theta
        # the augmentor changes perturbs x,y,theta at t=0 with random noise and then solves for a smooth trajectory close to the original target
        # theta for t=0 is 0 (reference frame is ego at t=0)
        #
        # v, a remain unchanged. theta_dot is augmented according to theta

        # make ego feature from pgp features
        pgp_ego_features = features["pgp_features"].ego_agent_features.ego_feats
        if pgp_ego_features.ndim == 4:
            raise RuntimeError("pgp feature must not have batch dimension augmentation")
        num_frames = pgp_ego_features.shape[-2]
        dummy_heading = np.zeros([num_frames, 1])

        agents_feature = Agents(
            # turn batch dimension into list, squeeze "num_agents" dimension, select feature 0,1,2 = (x,y,theta_dot)
            # use empty Agents feature to pass post_init test
            ego=[np.concatenate([pgp_ego_features[0, :, :2], dummy_heading], axis=-1)],
            agents=[np.empty([num_frames, 1, 8])],
        )
        # augment feature
        features.update({"agents": agents_feature})
        features, targets = super().augment(
            features=features, targets=targets, scenario=scenario
        )

        # make pgp feature from augmented feature, overwrite trajectory and multimodal trajectories
        new_curr_pos = features["agents"].ego[0][-1, :2]

        # trajectory is already augmented within super().augment
        features["pgp_features"].ego_agent_features.ego_feats[0, -1, :2] = new_curr_pos
        targets["multimodal_trajectories"].trajectories = targets["trajectory"].data[
            None, :, :
        ]
        del features["agents"]

        return features, targets

    @property
    def required_features(self) -> List[str]:
        """Inherited, see superclass."""
        return ["pgp_features"]

    @property
    def required_targets(self) -> List[str]:
        """Inherited, see superclass."""
        return ["trajectory", "multimodal_trajectories"]
