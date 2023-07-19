from typing import List, Optional, Tuple

import numpy as np
from nuplan.planning.scenario_builder.abstract_scenario import AbstractScenario
from nuplan.planning.training.data_augmentation.abstract_data_augmentation import (
    AbstractAugmentor,
)
from nuplan.planning.training.data_augmentation.data_augmentation_util import (
    ParameterToScale,
    ScalingDirection,
)
from nuplan.planning.training.modeling.types import FeaturesType, TargetsType


class PGPAgentDropoutAugmentor(AbstractAugmentor):
    def __init__(self, augment_prob: float, dropout_rate: float) -> None:
        """
        Initialize the augmentor.
        :param augment_prob: Probability between 0 and 1 of applying the data augmentation.
        :param dropout_rate: Rate of agents in the scenes to drop out - 0 means no dropout.
        """
        self._augment_prob = augment_prob
        self._dropout_rate = dropout_rate

    def augment(
        self,
        features: FeaturesType,
        targets: TargetsType,
        scenario: Optional[AbstractScenario] = None,
    ) -> Tuple[FeaturesType, TargetsType]:
        """Inherited, see superclass."""
        if np.random.rand() >= self._augment_prob:
            return features, targets
        num_vehicles = features[
            "pgp_features"
        ].ego_agent_features.vehicle_agent_masks.shape[-3]
        keep_mask = np.random.choice(
            [True, False],
            num_vehicles,
            p=[1.0 - self._dropout_rate, self._dropout_rate],
        )
        features["pgp_features"].ego_agent_features.vehicle_agent_masks[
            keep_mask, :, :
        ] = 1
        features["pgp_features"].ego_agent_features.vehicle_agent_feats[
            keep_mask, :, :
        ] = 0

        num_pedestrians = features[
            "pgp_features"
        ].ego_agent_features.pedestrians_agent_masks.shape[-3]
        keep_mask = np.random.choice(
            [True, False],
            num_pedestrians,
            p=[1.0 - self._dropout_rate, self._dropout_rate],
        )
        features["pgp_features"].ego_agent_features.pedestrians_agent_masks[
            keep_mask, :, :
        ] = 1
        features["pgp_features"].ego_agent_features.pedestrians_agent_feats[
            keep_mask, :, :
        ] = 0

        # if no agents are left in the scene, the agents feature tensor and mask
        # has to be overwritten with an empty one to prevent missing gradients
        if np.all(features["pgp_features"].ego_agent_features.vehicle_agent_masks == 1):
            num_past_poses = features[
                "pgp_features"
            ].ego_agent_features.vehicle_agent_masks.shape[1]
            features["pgp_features"].ego_agent_features.vehicle_agent_masks = np.full(
                shape=(0, num_past_poses, 5), fill_value=0.0, dtype=np.float16
            )
            features["pgp_features"].ego_agent_features.vehicle_agent_feats = np.empty(
                shape=(0, num_past_poses, 5), dtype=np.float32
            )
        if np.all(
            features["pgp_features"].ego_agent_features.pedestrians_agent_masks == 1
        ):
            num_past_poses = features[
                "pgp_features"
            ].ego_agent_features.pedestrians_agent_masks.shape[1]
            features[
                "pgp_features"
            ].ego_agent_features.pedestrians_agent_masks = np.full(
                shape=(0, num_past_poses, 5), fill_value=0.0, dtype=np.float16
            )
            features[
                "pgp_features"
            ].ego_agent_features.pedestrians_agent_feats = np.empty(
                shape=(0, num_past_poses, 5), dtype=np.float32
            )
        return features, targets

    @property
    def required_features(self) -> List[str]:
        """Inherited, see superclass."""
        return ["pgp_features"]

    @property
    def required_targets(self) -> List[str]:
        """Inherited, see superclass."""
        return []

    @property
    def augmentation_probability(self) -> ParameterToScale:
        """Inherited, see superclass."""
        return ParameterToScale(
            param=self._augment_prob,
            param_name=f"{self._augment_prob=}".partition("=")[0].split(".")[1],
            scaling_direction=ScalingDirection.MAX,
        )
