from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Union
import torch

from nuplan.planning.training.preprocessing.features.abstract_model_feature import (
    AbstractModelFeature,
    FeatureDataType,
    to_tensor
)


@dataclass
class GPTDriverFeatures(AbstractModelFeature):
    """
    Dataclass that holds map/environment signals that can be consumed by a LLM behavior planner
    All params are wrapped in a list representing the batch dimension
    :param ego_state: Tensor containing only the current velocity
    :param ego_poses: Tensor containing the past ego poses (x,y,theta)
    :param agents: List of length N_agents, with each entry being a Tensor containing (x,y,theta,vx,vy)
    :param objects: List of length N_objects, with each entry being a Tensor containing (x,y,theta)
    :param centerlines: Dict with centerlines 'left', 'right', 'current' with each value being None or a centerline List of length N_points 
        containing a Tensor of shape [3=(x,y,theta)]
    all params are wrapped in a list that represents the batch dimesion
    """

    ego_state: List[FeatureDataType]
    ego_poses: List[FeatureDataType]
    centerlines: List[Dict[str, Union[List[FeatureDataType], None]]]
    agents: List[List[FeatureDataType]]
    objects: List[List[FeatureDataType]]

    def to_feature_tensor(self) -> AbstractModelFeature:
        """Implemented. See interface."""
        return GPTDriverFeatures(
            ego_state=[to_tensor(sample) for sample in self.ego_state],
            agents=[[to_tensor(agent) for agent in sample] for sample in self.agents],
            objects=[[to_tensor(obj) for obj in sample] for sample in self.objects],
            centerlines=[
                {k:([to_tensor(pose) for pose in lane] if lane else None) for k,lane in sample.items()}
                for sample in self.centerlines
            ],
        ) 

    def to_device(self, device: torch.device) -> GPTDriverFeatures:
        """Implemented. See interface."""
        return GPTDriverFeatures(
            ego_state=[sample.to(device) for sample in self.ego_state],
            ego_poses=[sample.to(device) for sample in self.ego_poses],
            objects=[sample.to(device) for sample in self.objects],
            agents=[[agent.to(device) for agent in sample] for sample in self.agents],
            centerlines=[
                {k:([pose.to(device) for pose in lane] if lane else None) for k,lane in sample.items()} 
                for sample in self.centerlines
            ],
        )

    @classmethod
    def deserialize(cls, data: Dict[str, Any]) -> GPTDriverFeatures:
        """Implemented. See interface."""
        return GPTDriverFeatures(
            ego_state=data["ego_state"],
            ego_poses=data["ego_poses"],
            objects=data["objects"],
            centerlines=data["centerlines"],
            agents=data["agents"],
        )

    def unpack(self) -> List[GPTDriverFeatures]:
        """Implemented. See interface."""
        return [
            GPTDriverFeatures(
                ego_state=[ego_state],
                ego_poses=[ego_pose],
                objects=[obj],
                centerlines=[centerlines],
                agents=[agents]
            )
            for ego_state, ego_pose, obj, centerlines, agents
            in zip(
                self.ego_state, 
                self.ego_poses,
                self.objects,
                self.centerlines,
                self.agents
            )
        ]
    
    @classmethod
    def collate(cls, batch: List[GPTDriverFeatures]) -> GPTDriverFeatures:
        def _flatten_list(l: List[List[Any]]) -> List[Any]:
            return [j for i in l for j in i]
        
        return GPTDriverFeatures(
            ego_state=_flatten_list([es.ego_state for es in batch]),
            ego_poses=_flatten_list([es.ego_poses for es in batch]),
            objects=_flatten_list([es.objects for es in batch]),
            agents=_flatten_list([es.agents for es in batch]),
            centerlines=_flatten_list([es.centerlines for es in batch]),
        )