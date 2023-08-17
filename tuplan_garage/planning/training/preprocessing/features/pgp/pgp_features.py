from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

import torch
from nuplan.planning.training.preprocessing.features.abstract_model_feature import (
    AbstractModelFeature,
)

from tuplan_garage.planning.training.preprocessing.features.pgp.pgp_agent_node_masks import (
    PGPAgentNodeMasks,
)
from tuplan_garage.planning.training.preprocessing.features.pgp.pgp_ego_agents import (
    PGPEgoAgents,
)
from tuplan_garage.planning.training.preprocessing.features.pgp.pgp_graph_map import (
    PGPGraphMap,
)
from tuplan_garage.planning.training.preprocessing.features.pgp.pgp_traversals import (
    PGPTraversals,
)


@dataclass
class PGPFeatures(AbstractModelFeature):
    """
    : ego_agent_features: See PGPEgoAgents for details
    : graph_map: See PGPGraphMap for details
    : att_node_masks: See PGPAgentNodeMasks for details
    : traversal_features: See PGPTraversals for details
    """

    ego_agent_features: PGPEgoAgents
    graph_map: PGPGraphMap
    att_node_masks: PGPAgentNodeMasks
    traversal_features: PGPTraversals

    def __post_init__(self) -> None:
        # TODO: assert types are correct
        if any(
            [
                self.ego_agent_features.batch_size,
                self.graph_map.batch_size,
                self.att_node_masks.batch_size,
            ]
        ):
            batch_size = self.ego_agent_features.batch_size
            same_batch_size = (self.graph_map.batch_size == batch_size) and (
                self.att_node_masks.batch_size == batch_size
            )
            if not same_batch_size:
                raise AssertionError(
                    "All feature tensors need to have the same batch_size!"
                )

    @property
    def has_init_node(self) -> bool:
        # TODO: implement
        return True

    def to_feature_tensor(self) -> PGPFeatures:
        """
        :return object which will be collated into a batch
        """
        return PGPFeatures(
            ego_agent_features=self.ego_agent_features.to_feature_tensor(),
            graph_map=self.graph_map.to_feature_tensor(),
            att_node_masks=self.att_node_masks.to_feature_tensor(),
            traversal_features=self.traversal_features.to_feature_tensor(),
        )

    def to_device(self, device: torch.device) -> PGPFeatures:
        """Implemented. See interface."""
        return PGPFeatures(
            ego_agent_features=self.ego_agent_features.to_device(device=device),
            graph_map=self.graph_map.to_device(device=device),
            att_node_masks=self.att_node_masks.to_device(device=device),
            traversal_features=self.traversal_features.to_device(device=device),
        )

    @classmethod
    def deserialize(cls, data: Dict[str, Any]) -> PGPFeatures:
        """
        :return: Return dictionary of data that can be serialized
        """
        return PGPFeatures(
            ego_agent_features=PGPEgoAgents.deserialize(data["ego_agent_features"]),
            graph_map=PGPGraphMap.deserialize(data=data["graph_map"]),
            att_node_masks=PGPAgentNodeMasks.deserialize(data=data["att_node_masks"]),
            traversal_features=PGPTraversals.deserialize(
                data=data["traversal_features"]
            ),
        )

    def unpack(self) -> List[PGPFeatures]:
        """
        :return: Unpack a batched feature to a list of features.
        """
        return [
            PGPFeatures(
                ego_agent_features=ego_agent_features,
                graph_map=graph_map,
                att_node_masks=att_node_masks,
                traversal_features=traversal_features,
            )
            for ego_agent_features, graph_map, att_node_masks, traversal_features in zip(
                self.ego_agent_features.unpack(),
                self.graph_map.unpack(),
                self.att_node_masks.unpack(),
                self.traversal_features.unpack(),
            )
        ]

    @property
    def batch_size(self) -> int:
        """
        :return: number of batches
        """
        # batch size is equal for all sub-features (see post_init)
        return self.ego_agent_features.batch_size

    @classmethod
    def collate(cls, batch: List[PGPFeatures]) -> PGPFeatures:
        """
        Implemented. See interface.
        Collates a list of features that each have batch size of 1.
        """
        return PGPFeatures(
            ego_agent_features=PGPEgoAgents.collate(
                [item.ego_agent_features for item in batch]
            ),
            graph_map=PGPGraphMap.collate([item.graph_map for item in batch]),
            att_node_masks=PGPAgentNodeMasks.collate(
                [item.att_node_masks for item in batch]
            ),
            traversal_features=PGPTraversals.collate(
                [item.traversal_features for item in batch]
            ),
        )
