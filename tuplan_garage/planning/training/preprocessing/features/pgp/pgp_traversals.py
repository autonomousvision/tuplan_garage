from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

import torch
from nuplan.planning.script.builders.utils.utils_type import validate_type
from nuplan.planning.training.preprocessing.features.abstract_model_feature import (
    AbstractModelFeature,
    FeatureDataType,
    to_tensor,
)


@dataclass
class PGPTraversals(AbstractModelFeature):
    """
    Contains features for sampling of traversals in PGP Aggregator
    node_seq_gt and visited edges are ground truths that can be used for teacher-forcing pre-training
        they are computed in the FeaturesBuilder instead of as a separate target to reduce computational effort
    :init_node: shape [num_nodes]
    :node_seq_gt: shape [num_poses]
    :visited_edges: shape [num_nodes, num_edges] Note: in the original implementation this is called evf
    """

    init_node: FeatureDataType
    node_seq_gt: FeatureDataType
    visited_edges: FeatureDataType

    def __post_init__(self) -> None:
        pass

    def to_feature_tensor(self) -> PGPTraversals:
        """
        :return object which will be collated into a batch
        """
        return PGPTraversals(
            init_node=to_tensor(self.init_node),
            node_seq_gt=to_tensor(self.node_seq_gt),
            visited_edges=to_tensor(self.visited_edges),
        )

    def to_device(self, device: torch.device) -> PGPTraversals:
        """Implemented. See interface."""
        validate_type(self.init_node, torch.Tensor)
        validate_type(self.node_seq_gt, torch.Tensor)
        validate_type(self.visited_edges, torch.Tensor)
        return PGPTraversals(
            init_node=self.init_node.to(device=device),
            node_seq_gt=self.node_seq_gt.to(device=device),
            visited_edges=self.visited_edges.to(device=device),
        )

    @classmethod
    def deserialize(cls, data: Dict[str, Any]) -> PGPTraversals:
        """
        :return: Return dictionary of data that can be serialized
        """
        return PGPTraversals(
            init_node=data["init_node"],
            node_seq_gt=data["node_seq_gt"],
            visited_edges=data["visited_edges"],
        )

    def unpack(self) -> List[PGPTraversals]:
        """
        :return: Unpack a batched feature to a list of features.
        """
        return [
            PGPTraversals(
                init_node=init_node[None],
                node_seq_gt=node_seq_gt[None],
                visited_edges=visited_edges[None],
            )
            for init_node, node_seq_gt, visited_edges in zip(
                self.init_node, self.node_seq_gt, self.visited_edges
            )
        ]

    @classmethod
    def collate(cls, batch: List[PGPTraversals]) -> PGPTraversals:
        """
        Implemented. See interface.
        Collates a list of features that each have batch size of 1.

        :init_node: shape [num_nodes]
        :node_seq_gt: shape [num_poses]
        :visited_edges: shape [num_nodes, num_edges]

        """
        device = batch[0].init_node.device
        max_nodes = max([item.init_node.shape[0] for item in batch])
        max_edges = max([item.visited_edges.shape[1] for item in batch])

        collated_init_node = torch.zeros([len(batch), max_nodes], device=device)
        collated_visited_edges = torch.zeros(
            [len(batch), max_nodes, max_edges], device=device
        )
        collated_node_seq_gt = torch.zeros(
            [len(batch), batch[0].node_seq_gt.shape[-1]], device=device
        )

        for i, item in enumerate(batch):
            num_nodes, num_edges = item.visited_edges.shape
            assert item.init_node.shape[0] == num_nodes
            collated_init_node[i, :num_nodes] = item.init_node
            collated_visited_edges[i, :num_nodes, :num_edges] = item.visited_edges

            # the index of the goal node (=node_idx + num_nodes) has to be changed
            # since the index of a nodes goal state is node_idx + num_nodes
            node_seq_gt = item.node_seq_gt.clone()
            node_seq_gt[node_seq_gt >= num_nodes] += max_nodes - num_nodes
            collated_node_seq_gt[i, :] = node_seq_gt

            # the last column of visited_edges refers to a goal state
            # when increasing the number of edges, the flag for the last visited node
            # (i.e. the node from where ego moved to the "goal state") has to be moved
            # to make sure it stays in the last column
            item_goal_node_idx = item.node_seq_gt[-1]
            last_visited_node_idx = item_goal_node_idx - num_nodes
            collated_visited_edges[i, last_visited_node_idx.long(), num_edges - 1] = 0
            collated_visited_edges[i, last_visited_node_idx.long(), -1] = 1

        return PGPTraversals(
            init_node=collated_init_node,
            node_seq_gt=collated_node_seq_gt,
            visited_edges=collated_visited_edges,
        )
