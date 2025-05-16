import torch
from torch import nn

from tuplan_garage.planning.simulation.planner.dtpp_planner.prediction_modules import (
    AgentDecoder,
    AgentEncoder,
    CrossAttention,
    ScoreDecoder,
    VectorMapEncoder,
)


class Encoder(nn.Module):
    def __init__(self, dim=256, layers=3, heads=8, dropout=0.1):
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
        attention_layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=heads,
            dim_feedforward=dim * 4,
            activation="gelu",
            dropout=dropout,
            batch_first=True,
        )
        self.fusion_encoder = nn.TransformerEncoder(
            attention_layer, layers, enable_nested_tensor=False
        )

    def forward(self, inputs):
        # agents
        ego = inputs["ego_agent_past"]
        neighbors = inputs["neighbor_agents_past"]
        actors = torch.cat([ego[:, None, :, :5], neighbors[..., :5]], dim=1)

        # agent encoding
        encoded_ego = self.ego_encoder(ego)
        encoded_neighbors = [
            self.agent_encoder(neighbors[:, i]) for i in range(neighbors.shape[1])
        ]
        encoded_actors = torch.stack([encoded_ego] + encoded_neighbors, dim=1)
        actors_mask = torch.eq(actors[:, :, -1].sum(-1), 0)

        # vector maps
        map_lanes = inputs["map_lanes"]
        map_crosswalks = inputs["map_crosswalks"]

        # map encoding
        encoded_map_lanes, lanes_mask = self.lane_encoder(map_lanes)
        encoded_map_crosswalks, crosswalks_mask = self.crosswalk_encoder(map_crosswalks)

        # attention fusion encoding
        input = torch.cat(
            [encoded_actors, encoded_map_lanes, encoded_map_crosswalks], dim=1
        )
        mask = torch.cat([actors_mask, lanes_mask, crosswalks_mask], dim=1)
        encoding = self.fusion_encoder(input, src_key_padding_mask=mask)

        # outputs
        encoder_outputs = {"encoding": encoding, "mask": mask}

        return encoder_outputs


class Decoder(nn.Module):
    def __init__(
        self,
        neighbors=10,
        max_time=8,
        max_branch=30,
        n_heads=8,
        dim=256,
        variable_cost=False,
    ):
        super(Decoder, self).__init__()
        self._neighbors = neighbors
        self._nheads = n_heads
        self._time = max_time
        self._branch = max_branch

        self.environment_decoder = CrossAttention(n_heads, dim)
        self.ego_condition_decoder = CrossAttention(n_heads, dim)
        self.time_embed = nn.Embedding(max_time, dim)
        self.ego_traj_encoder = nn.Sequential(
            nn.Linear(6, 64), nn.ReLU(), nn.Linear(64, 256)
        )
        self.agent_traj_decoder = AgentDecoder(max_time, max_branch, dim * 2)
        self.ego_traj_decoder = nn.Sequential(
            nn.Linear(256, 256), nn.ELU(), nn.Linear(256, max_time * 10 * 3)
        )
        self.scorer = ScoreDecoder(variable_cost)
        self.register_buffer("casual_mask", self.generate_casual_mask())
        self.register_buffer("time_index", torch.arange(max_time).repeat(max_branch, 1))

    def pooling_trajectory(self, trajectory_tree):
        B, M, T, D = trajectory_tree.shape
        trajectory_tree = torch.reshape(trajectory_tree, (B, M, T // 10, 10, D))
        trajectory_tree = torch.max(trajectory_tree, dim=-2)[0]

        return trajectory_tree

    def generate_casual_mask(self):
        time_mask = torch.tril(torch.ones(self._time, self._time))
        casual_mask = torch.zeros(self._branch * self._time, self._branch * self._time)
        for i in range(self._branch):
            casual_mask[
                i * self._time : (i + 1) * self._time,
                i * self._time : (i + 1) * self._time,
            ] = time_mask

        return casual_mask

    def forward(self, encoder_outputs, ego_traj_inputs, agents_states, timesteps):
        # get inputs
        current_states = agents_states[:, : self._neighbors, -1]
        encoding, encoding_mask = encoder_outputs["encoding"], encoder_outputs["mask"]
        ego_traj_ori_encoding = self.ego_traj_encoder(ego_traj_inputs)
        branch_embedding = ego_traj_ori_encoding[:, :, timesteps - 1]
        ego_traj_ori_encoding = self.pooling_trajectory(ego_traj_ori_encoding)
        time_embedding = self.time_embed(self.time_index)
        tree_embedding = time_embedding[None, :, :, :] + branch_embedding[:, :, None, :]

        # get mask
        ego_traj_mask = torch.ne(ego_traj_inputs.sum(-1), 0)
        ego_traj_mask = ego_traj_mask[:, :, :: (ego_traj_mask.shape[-1] // self._time)]
        ego_traj_mask = torch.reshape(ego_traj_mask, (ego_traj_mask.shape[0], -1))
        env_mask = torch.einsum(
            "ij,ik->ijk", ego_traj_mask, encoding_mask.logical_not()
        )
        env_mask = torch.where(env_mask == 1, 0, -1e9)
        env_mask = env_mask.repeat(self._nheads, 1, 1)
        ego_condition_mask = self.casual_mask[None, :, :] * ego_traj_mask[:, :, None]
        ego_condition_mask = torch.where(ego_condition_mask == 1, 0, -1e9)
        ego_condition_mask = ego_condition_mask.repeat(self._nheads, 1, 1)

        # decode
        agents_trajecotries = []
        for i in range(self._neighbors):
            # learnable query
            query = encoding[:, i + 1, None, None] + tree_embedding
            query = torch.reshape(query, (query.shape[0], -1, query.shape[-1]))

            # decode from environment inputs
            env_decoding = self.environment_decoder(query, encoding, encoding, env_mask)

            # decode from ego trajectory inputs
            ego_traj_encoding = torch.reshape(
                ego_traj_ori_encoding,
                (ego_traj_ori_encoding.shape[0], -1, ego_traj_ori_encoding.shape[-1]),
            )
            ego_condition_decoding = self.ego_condition_decoder(
                query, ego_traj_encoding, ego_traj_encoding, ego_condition_mask
            )

            # trajectory outputs
            decoding = torch.cat([env_decoding, ego_condition_decoding], dim=-1)
            trajectory = self.agent_traj_decoder(decoding, current_states[:, i])
            agents_trajecotries.append(trajectory)

        # score outputs
        agents_trajecotries = torch.stack(agents_trajecotries, dim=2)
        scores, weights = self.scorer(
            ego_traj_inputs,
            encoding[:, 0],
            agents_trajecotries,
            current_states,
            timesteps,
        )

        # ego regularization
        ego_traj_regularization = self.ego_traj_decoder(encoding[:, 0])
        ego_traj_regularization = torch.reshape(
            ego_traj_regularization, (ego_traj_regularization.shape[0], 80, 3)
        )

        return agents_trajecotries, scores, ego_traj_regularization, weights
