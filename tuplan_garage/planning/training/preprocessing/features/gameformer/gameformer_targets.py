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
class GameFormerTargets(AbstractModelFeature):
    ego_agent_future: FeatureDataType
    neighbor_agents_future: FeatureDataType
    decoder_outputs: Dict[str, FeatureDataType]

    def to_feature_tensor(self) -> GameFormerTargets:
        """
        :return object which will be collated into a batch
        """
        return GameFormerTargets(
            ego_agent_future=to_tensor(self.ego_agent_future),
            neighbor_agents_future=to_tensor(self.neighbor_agents_future),
            decoder_outputs={k: to_tensor(v) for k, v in self.decoder_outputs.items()},
        )

    def to_device(self, device: torch.device) -> GameFormerTargets:
        """Implemented. See interface."""
        validate_type(self.ego_agent_future, torch.Tensor)
        validate_type(self.neighbor_agents_future, torch.Tensor)

        return GameFormerTargets(
            ego_agent_future=self.ego_agent_future.to(device=device),
            neighbor_agents_future=self.neighbor_agents_future.to(device=device),
            decoder_outputs={
                k: v.to(device=device) for k, v in self.decoder_outputs.items()
            },
        )

    @classmethod
    def deserialize(cls, data: Dict[str, Any]) -> GameFormerTargets:
        """
        :return: Return dictionary of data that can be serialized
        """
        return GameFormerTargets(
            ego_agent_future=data["ego_agent_future"],
            neighbor_agents_future=data["neighbor_agents_future"],
            decoder_outputs=data["decoder_outputs"],
        )

    def unpack(self) -> List[GameFormerTargets]:
        """
        :return: Unpack a batched feature to a list of features.
        """
        batch_size = self.ego_agent_future.shape[0]
        unpacked_decoder_outputs = [
            {k: v[i : i + 1] for k, v in self.decoder_outputs.items()}
            for i in range(batch_size)
        ]
        return [
            GameFormerTargets(
                ego_agent_future[None],
                neighbor_agents_future[None],
                decoder_outputs,
            )
            for (ego_agent_future, neighbor_agents_future, decoder_outputs) in zip(
                self.ego_agent_future,
                self.neighbor_agents_future,
                unpacked_decoder_outputs,
            )
        ]

    @classmethod
    def collate(cls, batch: List[GameFormerTargets]) -> GameFormerTargets:
        """
        Implemented. See interface.
        Collates a list of features that each have batch size of 1.
        """
        device = batch[0].ego_agent_future.device

        collated_decoder_outputs = {}
        for k in batch[0].decoder_outputs:
            collated_decoder_outputs[k] = torch.stack(
                [item.decoder_outputs[k] for item in batch], dim=0
            ).to(device)

        collated_ego_agent_future = torch.stack(
            [item.ego_agent_future for item in batch], dim=0
        ).to(device)

        collated_neighbor_agents_future = torch.stack(
            [item.neighbor_agents_future for item in batch], dim=0
        ).to(device)

        return GameFormerTargets(
            ego_agent_future=collated_ego_agent_future,
            neighbor_agents_future=collated_neighbor_agents_future,
        )
