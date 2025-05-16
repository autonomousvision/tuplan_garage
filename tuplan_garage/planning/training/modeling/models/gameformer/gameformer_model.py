import warnings

import torch
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
from nuplan.planning.training.modeling.torch_module_wrapper import TorchModuleWrapper
from nuplan.planning.training.preprocessing.features.trajectory import Trajectory
from nuplan.planning.training.preprocessing.target_builders.ego_trajectory_target_builder import (
    EgoTrajectoryTargetBuilder,
)

from tuplan_garage.planning.training.modeling.models.gameformer.predictor_modules import (
    AgentEncoder,
    CrossTransformer,
    FutureEncoder,
    InitialPredictionDecoder,
    InteractionDecoder,
    VectorMapEncoder,
)
from tuplan_garage.planning.training.preprocessing.feature_builders.gameformer.game_former_feature_builder import (
    GameFormerFeatureBuilder,
)
from tuplan_garage.planning.training.preprocessing.features.gameformer.gameformer_targets import (
    GameFormerTargets,
)


class Encoder(torch.nn.Module):
    def __init__(self, dim=256, layers=6, heads=8, dropout=0.1):
        super(Encoder, self).__init__()
        self._lane_len = 50
        self._lane_feature = 7
        self._crosswalk_len = 30
        self._crosswalk_feature = 3
        self.agent_encoder = AgentEncoder(agent_dim=11)
        self.ego_encoder = AgentEncoder(agent_dim=7)
        self.lane_encoder = VectorMapEncoder(self._lane_feature, self._lane_len)
        self.crosswalk_encoder = VectorMapEncoder(
            self._crosswalk_feature, self._crosswalk_len
        )
        attention_layer = torch.nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=heads,
            dim_feedforward=dim * 4,
            activation="gelu",
            dropout=dropout,
            batch_first=True,
        )
        self.fusion_encoder = torch.nn.TransformerEncoder(
            attention_layer, layers, enable_nested_tensor=False
        )

    def forward(self, inputs):
        # agents
        ego = inputs["gameformer_features"].ego_agent_past
        neighbors = inputs["gameformer_features"].neighbor_agents_past
        actors = torch.cat([ego[:, None, :, :5], neighbors[..., :5]], dim=1)

        # agent encoding
        encoded_ego = self.ego_encoder(ego)
        encoded_neighbors = [
            self.agent_encoder(neighbors[:, i]) for i in range(neighbors.shape[1])
        ]
        encoded_actors = torch.stack([encoded_ego] + encoded_neighbors, dim=1)
        actors_mask = torch.eq(actors[:, :, -1].sum(-1), 0)

        # vector maps
        map_lanes = inputs["gameformer_features"].lanes
        map_crosswalks = inputs["gameformer_features"].crosswalks

        # map encoding
        encoded_map_lanes, lanes_mask = self.lane_encoder(map_lanes)
        encoded_map_crosswalks, crosswalks_mask = self.crosswalk_encoder(map_crosswalks)

        # attention fusion encoding
        input = torch.cat(
            [encoded_actors, encoded_map_lanes, encoded_map_crosswalks], dim=1
        )
        mask = torch.cat([actors_mask, lanes_mask, crosswalks_mask], dim=1)

        warnings.filterwarnings("ignore", message=".*Converting mask without.*")
        encoding = self.fusion_encoder(input, src_key_padding_mask=mask)

        # outputs
        encoder_outputs = {
            "actors": actors,
            "encoding": encoding,
            "mask": mask,
            "route_lanes": inputs["gameformer_features"].route_lanes,
        }

        return encoder_outputs


class Decoder(torch.nn.Module):
    def __init__(self, neighbors=10, modalities=6, levels=3):
        super(Decoder, self).__init__()
        self.levels = levels
        future_encoder = FutureEncoder()

        # initial level
        self.initial_predictor = InitialPredictionDecoder(modalities, neighbors)

        # level-k reasoning
        self.interaction_stage = torch.nn.ModuleList(
            [InteractionDecoder(modalities, future_encoder) for _ in range(levels)]
        )

    def forward(self, encoder_outputs):
        decoder_outputs = {}
        current_states = encoder_outputs["actors"][:, :, -1]
        encoding, mask = encoder_outputs["encoding"], encoder_outputs["mask"]

        # level 0 decode
        last_content, last_level, last_score = self.initial_predictor(
            current_states, encoding, mask
        )
        decoder_outputs["level_0_interactions"] = last_level
        decoder_outputs["level_0_scores"] = last_score

        # level k reasoning
        for k in range(1, self.levels + 1):
            interaction_decoder = self.interaction_stage[k - 1]
            last_content, last_level, last_score = interaction_decoder(
                current_states, last_level, last_score, last_content, encoding, mask
            )
            decoder_outputs[f"level_{k}_interactions"] = last_level
            decoder_outputs[f"level_{k}_scores"] = last_score

        env_encoding = last_content[:, 0]

        return decoder_outputs, env_encoding


class NeuralPlanner(torch.nn.Module):
    def __init__(self):
        super(NeuralPlanner, self).__init__()
        self._future_len = 80
        self.route_fusion = CrossTransformer()
        self.plan_decoder = torch.nn.Sequential(
            torch.nn.Linear(512, 256),
            torch.nn.ELU(),
            torch.nn.Dropout(0.1),
            torch.nn.Linear(256, self._future_len * 2),
        )
        self.route_encoder = VectorMapEncoder(3, 50)

    def dynamics_layer(self, controls, initial_state):
        dt = 0.1  # discrete time period [s]
        max_a = 5  # vehicle's accleration limits [m/s^2]
        max_d = 0.5  # vehicle's steering limits [rad]

        vel_init = torch.hypot(initial_state[..., 3], initial_state[..., 4])
        vel = vel_init[:, None] + torch.cumsum(
            controls[..., 0].clamp(-max_a, max_a) * dt, dim=-1
        )
        vel = torch.clamp(vel, min=0)

        yaw_rate = controls[..., 1].clamp(-max_d, max_d) * vel
        yaw = initial_state[:, None, 2] + torch.cumsum(yaw_rate * dt, dim=-1)
        yaw = torch.fmod(yaw, 2 * torch.pi)

        vel_x = vel * torch.cos(yaw)
        vel_y = vel * torch.sin(yaw)

        x = initial_state[:, None, 0] + torch.cumsum(vel_x * dt, dim=-1)
        y = initial_state[:, None, 1] + torch.cumsum(vel_y * dt, dim=-1)

        return torch.stack((x, y, yaw), dim=-1)

    def forward(self, env_encoding, route_lanes, initial_state):
        route_lanes, mask = self.route_encoder(route_lanes)
        mask[:, 0] = False
        route_encoding = self.route_fusion(env_encoding, route_lanes, route_lanes, mask)
        env_route_encoding = torch.cat([env_encoding, route_encoding], dim=-1)
        env_route_encoding = torch.max(env_route_encoding, dim=1)[
            0
        ]  # max pooling over modalities
        control = self.plan_decoder(env_route_encoding)
        control = control.reshape(-1, self._future_len, 2)
        plan = self.dynamics_layer(control, initial_state)

        return plan


class GameFormer(TorchModuleWrapper):
    def __init__(self, encoder_layers=6, decoder_levels=3, modalities=6, neighbors=10):

        trajectory_sampling = TrajectorySampling(
            time_horizon=8.0,
            interval_length=0.1,
        )
        feature_builders = [GameFormerFeatureBuilder(num_agents=neighbors)]
        target_builders = [EgoTrajectoryTargetBuilder(trajectory_sampling)]

        super(GameFormer, self).__init__(
            future_trajectory_sampling=trajectory_sampling,
            feature_builders=feature_builders,
            target_builders=target_builders,
        )

        self.encoder = Encoder(layers=encoder_layers)
        self.decoder = Decoder(neighbors, modalities, decoder_levels)
        self.planner = NeuralPlanner()

    def forward(self, inputs):
        encoder_outputs = self.encoder(inputs)
        route_lanes = encoder_outputs["route_lanes"]
        initial_state = encoder_outputs["actors"][:, 0, -1]
        decoder_outputs, env_encoding = self.decoder(encoder_outputs)
        ego_plan = self.planner(env_encoding, route_lanes, initial_state)

        return {
            "trajectory": Trajectory(ego_plan),
            "gameformer_targets": GameFormerTargets(
                ego_agent_future=inputs["gameformer_features"].ego_agent_future,
                neighbor_agents_future=inputs[
                    "gameformer_features"
                ].neighbor_agents_future,
                decoder_outputs=decoder_outputs,
            ),
        }
