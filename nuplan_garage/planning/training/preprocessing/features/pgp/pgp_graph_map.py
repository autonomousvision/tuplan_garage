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
class PGPGraphMap(AbstractModelFeature):
    """
    Graph Map Structure for pgp model

    :lane_node_feats: shape [num_nodes (i.e. num polylines), num_poses (per polyline), num_feats_per_pose]
    :lane_node_masks:shape [num_nodes (i.e. num polylines), num_poses (per polyline), num_feats_per_pose]
    :lane_ids:shape [num_nodes]: maping the node indices to the nuplan map lane ids
    :s_next: shape [num_nodes, num_edges]
    :edge_type: shape [num_nodes, num_edges]: values {0: No edge exists, 1: successor edge, 2: proximal edge, 3: terminal edge}
    :edge_on_route_mask: shape [num_nodes, num_edges]: values {0: target node of edge is not on route, 1: target node of edge is on oute}
    :nodes_on_route_flag: shape [num_nodes (i.e. num polylines), num_poses (per polyline), 1]: values {0: not on route, 1: on route}
    :red_light_mask: shape [num_nodes, num_edges]: values {0: following edge is prohibied by red light, 1: else}
    :red_light_flag: shape [num_nodes (i.e. num polylines), num_poses (per polyline), 1]: values {0: unknown / green, 1: lane segment is prohibited by red light}
    """

    lane_node_feats: FeatureDataType
    lane_node_masks: FeatureDataType
    lane_ids: FeatureDataType
    s_next: FeatureDataType
    edge_type: FeatureDataType
    edge_on_route_mask: FeatureDataType
    nodes_on_route_flag: FeatureDataType
    red_light_mask: FeatureDataType
    red_light_flag: FeatureDataType

    def __post_init__(self) -> None:
        if self.lane_node_feats.shape != self.lane_node_masks.shape:
            raise RuntimeError(
                "lane node feats and lane node mask have to be of equal shape"
            )

        if self.lane_node_feats.shape[:2] != self.nodes_on_route_flag.shape[:2]:
            raise RuntimeError(
                "lane_node_feats and nodes_on_route_flag have to have same dimensions except for last"
            )

    def to_feature_tensor(self) -> PGPGraphMap:
        """
        :return object which will be collated into a batch
        """
        return PGPGraphMap(
            lane_node_feats=to_tensor(self.lane_node_feats),
            lane_node_masks=to_tensor(self.lane_node_masks),
            lane_ids=to_tensor(self.lane_ids),
            s_next=to_tensor(self.s_next),
            edge_type=to_tensor(self.edge_type),
            edge_on_route_mask=to_tensor(self.edge_on_route_mask),
            nodes_on_route_flag=to_tensor(self.nodes_on_route_flag),
            red_light_mask=to_tensor(self.red_light_mask),
            red_light_flag=to_tensor(self.red_light_flag),
        )

    def to_device(self, device: torch.device) -> PGPGraphMap:
        """Implemented. See interface."""
        validate_type(self.lane_node_feats, torch.Tensor)
        validate_type(self.lane_node_masks, torch.Tensor)
        validate_type(self.lane_ids, torch.Tensor)
        validate_type(self.s_next, torch.Tensor)
        validate_type(self.edge_type, torch.Tensor)
        validate_type(self.edge_on_route_mask, torch.Tensor)
        validate_type(self.nodes_on_route_flag, torch.Tensor)
        validate_type(self.red_light_mask, torch.Tensor)
        validate_type(self.red_light_flag, torch.Tensor)
        return PGPGraphMap(
            lane_node_feats=self.lane_node_feats.to(device=device),
            lane_node_masks=self.lane_node_masks.to(device=device),
            lane_ids=self.lane_ids.to(device=device),
            s_next=self.s_next.to(device=device),
            edge_type=self.edge_type.to(device=device),
            edge_on_route_mask=self.edge_on_route_mask.to(device=device),
            nodes_on_route_flag=self.nodes_on_route_flag.to(device=device),
            red_light_mask=self.red_light_mask.to(device=device),
            red_light_flag=self.red_light_flag.to(device=device),
        )

    @classmethod
    def deserialize(cls, data: Dict[str, Any]) -> PGPGraphMap:
        """
        :return: Return dictionary of data that can be serialized
        """
        return PGPGraphMap(
            lane_node_feats=data["lane_node_feats"],
            lane_node_masks=data["lane_node_masks"],
            lane_ids=data["lane_ids"],
            s_next=data["s_next"],
            edge_type=data["edge_type"],
            edge_on_route_mask=data["edge_on_route_mask"],
            nodes_on_route_flag=data["nodes_on_route_flag"],
            red_light_mask=data["red_light_mask"],
            red_light_flag=data["red_light_flag"],
        )

    def unpack(self) -> List[PGPGraphMap]:
        """
        :return: Unpack a batched feature to a list of features.
        """
        return [
            PGPGraphMap(
                lane_node_feats[None],
                lane_node_masks[None],
                lane_ids[None],
                s_next[None],
                edge_type[None],
                edge_on_route_mask[None],
                nodes_on_route_flag[None],
                red_light_mask[None],
                red_light_flag[None],
            )
            for (
                lane_node_feats,
                lane_node_masks,
                lane_ids,
                s_next,
                edge_type,
                edge_on_route_mask,
                nodes_on_route_flag,
                red_light_mask,
                red_light_flag,
            ) in zip(
                self.lane_node_feats,
                self.lane_node_masks,
                self.lane_ids,
                self.s_next,
                self.edge_type,
                self.edge_on_route_mask,
                self.nodes_on_route_flag,
                self.red_light_mask,
                self.red_light_flag,
            )
        ]

    @property
    def batch_size(self) -> int:
        """
        :return: number of batches
        """
        if self.lane_node_feats.ndim == 4:
            return self.lane_node_feats.shape[0]
        else:
            return None

    @classmethod
    def collate(cls, batch: List[PGPGraphMap]) -> PGPGraphMap:
        """
        Implemented. See interface.
        Collates a list of features that each have batch size of 1.
        """
        # lane_node_feats and lane_node_mask have same shape so checking one is enough
        assert batch[0].lane_node_feats.ndim == 3
        device = batch[0].lane_node_feats.device
        max_nodes = max([item.lane_node_feats.shape[0] for item in batch])
        num_poses_per_polyline = batch[0].lane_node_feats.shape[1]
        max_edges = max([item.s_next.shape[1] for item in batch])

        collated_lane_node_feats = torch.zeros(
            [len(batch), max_nodes, num_poses_per_polyline, 7], device=device
        )
        collated_lane_node_masks = torch.ones_like(collated_lane_node_feats)
        collated_lane_ids = torch.zeros([len(batch), max_nodes, 1], device=device)
        collated_s_next = torch.zeros(
            [len(batch), max_nodes, max_edges],
            device=device,
        )
        collated_edge_type = torch.zeros_like(collated_s_next)
        collated_on_route_mask = torch.zeros(
            [len(batch), max_nodes, max_edges],
            device=device,
        )
        collated_nodes_on_route_flag = torch.zeros(
            [len(batch), max_nodes, num_poses_per_polyline, 1],
            device=device,
        )
        collated_red_light_mask = torch.ones_like(collated_on_route_mask)
        collated_red_light_flag = torch.zeros_like(collated_nodes_on_route_flag)
        for i, item in enumerate(batch):
            num_nodes = item.lane_node_feats.shape[0]
            num_edges = item.s_next.shape[1]
            collated_lane_node_feats[i, :num_nodes, :, :] = item.lane_node_feats
            collated_lane_node_masks[i, :num_nodes, :, :] = item.lane_node_masks
            collated_lane_ids[i, :num_nodes, :] = item.lane_ids
            # the last edge refers to an end state. The end states index (=node_idx+num_nodes) depends on the number of nodes in the sample
            #   therefore the last edge has to be collated separatly
            collated_s_next[i, :num_nodes, : num_edges - 1] = item.s_next[
                :, : num_edges - 1
            ]
            collated_s_next[i, :num_nodes, -1] = item.s_next[:, -1] + (
                max_nodes - num_nodes
            )

            collated_edge_type[i, :num_nodes, : num_edges - 1] = item.edge_type[
                :, : num_edges - 1
            ]
            collated_edge_type[i, :num_nodes, -1] = 3

            collated_on_route_mask[
                i, :num_nodes, : num_edges - 1
            ] = item.edge_on_route_mask[:, : num_edges - 1]
            collated_nodes_on_route_flag[i, :num_nodes, :, :] = item.nodes_on_route_flag

            collated_red_light_mask[
                i, :num_nodes, : num_edges - 1
            ] = item.red_light_mask[:, : num_edges - 1]
            collated_red_light_flag[i, :num_nodes, :, :] = item.red_light_flag
        # staying in the same node has to be considered to be "on route" to prevent all probabilities being 0
        collated_on_route_mask[i, :, -1] = 1
        collated_red_light_mask[i, :, -1] = 1

        return PGPGraphMap(
            lane_node_feats=collated_lane_node_feats,
            lane_node_masks=collated_lane_node_masks,
            lane_ids=collated_lane_ids,
            s_next=collated_s_next,
            edge_type=collated_edge_type,
            edge_on_route_mask=collated_on_route_mask,
            nodes_on_route_flag=collated_nodes_on_route_flag,
            red_light_mask=collated_red_light_mask,
            red_light_flag=collated_red_light_flag,
        )
