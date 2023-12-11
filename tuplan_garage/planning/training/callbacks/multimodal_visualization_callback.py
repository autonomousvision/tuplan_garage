from typing import Any, List, Optional
import numpy as np
import numpy.typing as npt
import torch
from dataclasses import dataclass
from matplotlib.pyplot import get_cmap

from nuplan.planning.training.callbacks.utils.visualization_utils import (
    get_raster_from_vector_map_with_agents,
    get_raster_with_trajectories_as_rgb,
    _draw_trajectory,
    Color
)
from nuplan.planning.training.callbacks.visualization_callback import VisualizationCallback
from nuplan.planning.training.modeling.types import FeaturesType, TargetsType
from nuplan.planning.training.preprocessing.features.trajectory import Trajectory

@dataclass
class PREDICTION_COLOR:
    def __init__(self, idx, k=5):
        self.cmap = get_cmap("gist_rainbow", lut=k)
        self.idx = idx
        idcs = np.arange(k)
        self.map = self.cmap(idcs) * 255

    @property
    def value(self):
        return self.map[self.idx]

class MultimodalVisualizationCallback(VisualizationCallback):
    def __init__(
        self,
        images_per_tile: int,
        num_train_tiles: int,
        num_val_tiles: int,
        pixel_size: float,
    ):
        super().__init__(
            images_per_tile=images_per_tile,
            num_train_tiles=num_train_tiles,
            num_val_tiles=num_val_tiles,
            pixel_size=pixel_size,
        )

    
    def _log_batch(
        self,
        loggers: List[Any],
        features: FeaturesType,
        targets: TargetsType,
        predictions: TargetsType,
        batch_idx: int,
        training_step: int,
        prefix: str,
    ) -> None:
        """
        Visualizes and logs a batch of data (features, targets, predictions) from the model.

        :param loggers: list of loggers from the trainer
        :param features: tensor of model features
        :param targets: tensor of model targets
        :param predictions: tensor of model predictions
        :param batch_idx: index of total batches to visualize
        :param training_step: global trainign step
        :param prefix: prefix to add to the log tag
        """
        if 'multimodal_trajectories' not in targets or 'multimodal_trajectories' not in predictions:
            return
        if 'raster' in features:
            image_batch = self._get_images_from_raster_features(features, targets, predictions)
        elif 'vector_map' in features and 'agents' in features:
            image_batch = self._get_images_from_vector_features(features, targets, predictions)
        else:
            return

        tag = f'{prefix}_multimodal_visualization_{batch_idx}'

        for logger in loggers:
            if isinstance(logger, torch.utils.tensorboard.writer.SummaryWriter):
                logger.add_images(
                    tag=tag,
                    img_tensor=torch.from_numpy(image_batch),
                    global_step=training_step,
                    dataformats='NHWC',
                )

    def _get_images_from_raster_features(
        self, features: FeaturesType, targets: TargetsType, predictions: TargetsType
    ) -> npt.NDArray[np.uint8]:
        """
        Create a list of RGB raster images from a batch of model data of raster features.

        :param features: tensor of model features
        :param targets: tensor of model targets
        :param predictions: tensor of model predictions
        :return: list of raster images
        """
        images = list()

        for raster, target_trajectory, predicted_trajectories in zip(
            features['raster'].unpack(), targets['multimodal_trajectories'].unpack(), predictions['multimodal_trajectories'].unpack()
        ):
            image = get_raster_with_trajectories_as_rgb(
                raster,
                Trajectory(data=target_trajectory.trajectories.squeeze(1)),
                None,
                pixel_size=self.pixel_size,
            )
            for col_idx, predicted_trajectory in enumerate(predicted_trajectories.trajectories[0]):
                color=PREDICTION_COLOR(col_idx, k=predicted_trajectories.trajectories.shape[1])
                _draw_trajectory(
                    image=image,
                    trajectory=self._trajectory_to_waypoints_for_visualization(predicted_trajectory),
                    color=color,
                    pixel_size=self.pixel_size,
                    thickness=1,
                    radius=2,
                )

            images.append(image)

        return np.asarray(images)

    def _get_images_from_vector_features(
        self, features: FeaturesType, targets: TargetsType, predictions: TargetsType
    ) -> npt.NDArray[np.uint8]:
        """
        Create a list of RGB raster images from a batch of model data of vectormap and agent features.

        :param features: tensor of model features
        :param targets: tensor of model targets
        :param predictions: tensor of model predictions
        :return: list of raster images
        """
        images = list()

        for vector_map, agents, target_trajectory, predicted_trajectories in zip(
            features['vector_map'].unpack(),
            features['agents'].unpack(),
            targets['multimodal_trajectories'].unpack(),
            predictions['multimodal_trajectories'].unpack(),
        ):
            image = get_raster_from_vector_map_with_agents(
                vector_map,
                agents,
                Trajectory(data=target_trajectory.trajectories.squeeze(1)),
                None,
                pixel_size=self.pixel_size,
            )
            for predicted_trajectory in predicted_trajectories.trajectories:
                _draw_trajectory(
                    image=image,
                    trajectory=self._trajectory_to_waypoints_for_visualization(predicted_trajectory),
                    color=Color.PREDICTED_TRAJECTORY,
                    pixel_size=self.pixel_size,
                )

            images.append(image)

        return np.asarray(images)

    def _trajectory_to_waypoints_for_visualization(self, multimodal_trajectory: torch.Tensor) -> Trajectory:
        heading = torch.zeros_like(multimodal_trajectory)[...,:1]
        #print('trajbefore', multimodal_trajectory)
        #print('traj after', torch.cat([multimodal_trajectory, heading], dim=-1))
        return Trajectory(data=torch.cat([multimodal_trajectory, heading], dim=-1))