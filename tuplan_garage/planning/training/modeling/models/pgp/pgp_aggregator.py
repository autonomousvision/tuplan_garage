from enum import Enum
from typing import Dict, Tuple

import torch
import torch.nn as nn
from positional_encodings.torch_encodings import PositionalEncoding1D
from torch.distributions import Categorical


class TraversalReduction(Enum):
    NO_REDUCTION: Tuple[bool, bool] = False, False
    KEEP_ONLY_MOST_FREQUENT: Tuple[bool, bool] = True, True
    REPEAT_MOST_FREQUENT: Tuple[bool, bool] = True, False


class PGP(nn.Module):
    """
    Policy header + selective aggregator from "Multimodal trajectory prediction conditioned on lane graph traversals"
    1) Outputs edge probabilities corresponding to pi_route
    2) Samples pi_route to output traversed paths
    3) Selectively aggregates context along traversed paths
    """

    def __init__(
        self,
        pre_train: bool,
        node_enc_size: int,
        target_agent_enc_size: int,
        pi_h1_size: int,
        pi_h2_size: int,
        emb_size: int,
        num_heads: int,
        num_samples: int,
        horizon: int,
        use_route_mask: bool,
        hard_masking: bool,
        num_traversals: str,
        keep_only_best_traversal: bool = False,
    ):
        """
        'pre_train': bool, whether the model is being pre-trained using ground truth node sequence.
        'node_enc_size': int, size of node encoding
        'target_agent_enc_size': int, size of target agent encoding
        'pi_h1_size': int, size of first layer of policy header
        'pi_h2_size': int, size of second layer of policy header
        'emb_size': int, embedding size for attention layer for aggregating node encodings
        'num_heads: int, number of attention heads
        'num_samples': int, number of sampled traversals (and encodings) to output
        """
        super().__init__()
        self.pre_train = pre_train
        self.use_route_mask = use_route_mask
        self.hard_masking = hard_masking

        # define for interface checking
        self.agg_enc_size = emb_size + target_agent_enc_size
        self.target_agent_enc_size = target_agent_enc_size
        self.node_enc_size = node_enc_size

        # Policy header
        self.pi_h1 = nn.Linear(
            2 * node_enc_size + target_agent_enc_size + 2, pi_h1_size
        )
        self.pi_h2 = nn.Linear(pi_h1_size, pi_h2_size)
        self.pi_op = nn.Linear(pi_h2_size, 1)
        self.pi_h1_goal = nn.Linear(node_enc_size + target_agent_enc_size, pi_h1_size)
        self.pi_h2_goal = nn.Linear(pi_h1_size, pi_h2_size)
        self.pi_op_goal = nn.Linear(pi_h2_size, 1)
        self.leaky_relu = nn.LeakyReLU()
        self.log_softmax = nn.LogSoftmax(dim=2)

        # For sampling policy
        self.horizon = horizon
        if not keep_only_best_traversal:
            assert (
                num_samples == num_traversals
            ), "if all traversals are used, number of sampled traversals has to match number of returned samples"
        self.num_samples = num_samples
        self.num_traversals = num_traversals
        self.keep_only_best_traversal = keep_only_best_traversal

        # Attention based aggregator
        self.pos_enc = PositionalEncoding1D(node_enc_size)
        self.query_emb = nn.Linear(target_agent_enc_size, emb_size)
        self.key_emb = nn.Linear(node_enc_size, emb_size)
        self.val_emb = nn.Linear(node_enc_size, emb_size)
        self.mha = nn.MultiheadAttention(emb_size, num_heads)

        if use_route_mask and not hard_masking:
            self.route_bonus = nn.Parameter(torch.ones(1))

    def forward(self, encodings: Dict) -> Dict:
        """
        Forward pass for PGP aggregator
        :param encodings: dictionary with encoder outputs
        :return: outputs: dictionary with
            'agg_encoding': aggregated encodings along sampled traversals
            'pi': discrete policy (probabilities over outgoing edges) for graph traversal
        """

        # Unpack encodings:
        target_agent_encoding = encodings["target_agent_encoding"]
        node_encodings = encodings["context_encoding"]["combined"]
        node_masks = encodings["context_encoding"]["combined_masks"]
        s_next = encodings["s_next"]
        edge_type = encodings["edge_type"]
        edge_on_route_mask = encodings["edge_on_route_mask"]
        node_on_route_mask = encodings["node_on_route_mask"]

        # Compute pi (log probs)
        pi = self.compute_policy(
            target_agent_encoding,
            node_encodings,
            node_masks,
            s_next,
            edge_type,
            edge_on_route_mask,
        )

        # If pretraining model, use ground truth node sequences
        if self.pre_train:
            sampled_traversals = (
                encodings["node_seq_gt"]
                .unsqueeze(1)
                .repeat(1, self.num_samples, 1)
                .long()
            )
        else:
            # Sample pi
            init_node = encodings["init_node"]
            sampled_traversals = self.sample_policy(
                pi, s_next, init_node, node_on_route_mask
            )

        # Selectively aggregate context along traversed paths
        agg_enc = self.aggregate(
            sampled_traversals, node_encodings, target_agent_encoding
        )

        outputs = {
            "agg_encoding": agg_enc,
            "pi": torch.log(pi + 1e-5),
            "sampled_traversals": sampled_traversals,
        }
        return outputs

    def aggregate(
        self, sampled_traversals, node_encodings, target_agent_encoding
    ) -> torch.Tensor:

        # Useful variables:
        batch_size = node_encodings.shape[0]
        max_nodes = node_encodings.shape[1]
        # num_traversals = sampled_traversals.shape[1]

        # Get unique traversals and form consolidated batch:
        counted_unique_traversals = [
            torch.unique(i, dim=0, return_counts=True) for i in sampled_traversals
        ]
        unique_traversals = [i[0] for i in counted_unique_traversals]
        traversal_counts = [i[1] for i in counted_unique_traversals]
        if self.keep_only_best_traversal:
            # only a single traversal is used and passed to the decoder
            most_frequent_traversal_idcs = [
                torch.argmax(traversal_counts_sample)
                for traversal_counts_sample in traversal_counts
            ]
            best_traversals = [
                unique_traversals[i][most_frequent_traversal_idx].unsqueeze(0)
                for i, most_frequent_traversal_idx in enumerate(
                    most_frequent_traversal_idcs
                )
            ]
            traversal_counts = [
                torch.tensor([self.num_samples], device=sampled_traversals.device)
                for _ in traversal_counts
            ]
            unique_traversals = best_traversals

        traversals_batched = torch.cat(unique_traversals, dim=0)
        counts_batched = torch.cat(traversal_counts, dim=0)
        batch_idcs = torch.cat(
            [
                j * torch.ones(len(count)).long()
                for j, count in enumerate(traversal_counts)
            ]
        )
        batch_idcs = batch_idcs.unsqueeze(1).repeat(1, self.horizon)

        # Dummy encodings for goal nodes
        dummy_enc = torch.zeros_like(node_encodings)
        node_encodings = torch.cat((node_encodings, dummy_enc), dim=1)

        # Gather node encodings along traversed paths
        node_enc_selected = node_encodings[batch_idcs, traversals_batched]

        # Add positional encodings:
        pos_enc = self.pos_enc(torch.zeros_like(node_enc_selected))
        node_enc_selected += pos_enc

        # Multi-head attention
        target_agent_enc_batched = target_agent_encoding[batch_idcs[:, 0]]
        query = self.query_emb(target_agent_enc_batched).unsqueeze(0)
        keys = self.key_emb(node_enc_selected).permute(1, 0, 2)
        vals = self.val_emb(node_enc_selected).permute(1, 0, 2)
        key_padding_mask = torch.as_tensor(traversals_batched >= max_nodes)
        att_op, _ = self.mha(query, keys, vals, key_padding_mask)

        # Repeat based on counts
        att_op = (
            att_op.squeeze(0)
            .repeat_interleave(counts_batched, dim=0)
            .view(batch_size, self.num_samples, -1)
        )

        # Concatenate target agent encoding
        agg_enc = torch.cat(
            (target_agent_encoding.unsqueeze(1).repeat(1, self.num_samples, 1), att_op),
            dim=-1,
        )

        return agg_enc

    def sample_policy(
        self,
        pi: torch.Tensor,
        s_next: torch.Tensor,
        init_node: torch.Tensor,
        node_on_route_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Sample graph traversals using discrete policy.
        :param pi: tensor with probabilities corresponding to the policy
        :param s_next: look-up table for next node for a given source node and edge
        :param init_node: initial node to start the policy at
        :return:
        """
        with torch.no_grad():

            # Useful variables:
            batch_size = pi.shape[0]
            max_nodes = pi.shape[1]
            batch_idcs = (
                torch.arange(batch_size, device=pi.device)
                .unsqueeze(1)
                .repeat(1, self.num_traversals)
                .view(-1)
            )

            # Initialize output
            sampled_traversals = torch.zeros(
                batch_size, self.num_traversals, self.horizon, device=pi.device
            ).long()

            # Set up dummy self transitions for goal states:
            pi_dummy = torch.zeros_like(pi)
            pi_dummy[:, :, -1] = 1
            s_next_dummy = torch.zeros_like(s_next)
            s_next_dummy[:, :, -1] = max_nodes + torch.arange(max_nodes).unsqueeze(
                0
            ).repeat(batch_size, 1)
            pi = torch.cat((pi, pi_dummy), dim=1)
            s_next = torch.cat((s_next, s_next_dummy), dim=1)

            # Sample initial node:
            if self.use_route_mask and self.hard_masking:
                mask = 1.0 - ((init_node * node_on_route_mask).sum(dim=-1) > 0).float()
                node_on_route_mask = node_on_route_mask + mask.unsqueeze(-1)
                init_node = init_node * node_on_route_mask
                init_node = init_node / init_node.sum(dim=-1, keepdim=True)
            pi_s = (
                init_node.unsqueeze(1)
                .repeat(1, self.num_traversals, 1)
                .view(-1, max_nodes)
            )
            s = Categorical(pi_s).sample()

            sampled_traversals[:, :, 0] = s.reshape(batch_size, self.num_traversals)

            # Sample traversed paths for a fixed horizon
            for n in range(1, self.horizon):

                # Gather policy at appropriate indices:
                pi_s = pi[batch_idcs, s]
                # pi_s[...,-1] = 0.0

                # Sample edges
                a = Categorical(pi_s).sample()

                # Look-up next node
                s = s_next[batch_idcs, s, a].long()

                # Add node indices to sampled traversals
                sampled_traversals[:, :, n] = s.reshape(batch_size, self.num_traversals)

        return sampled_traversals

    def compute_policy(
        self,
        target_agent_encoding: torch.Tensor,
        node_encodings: torch.Tensor,
        node_masks: torch.Tensor,
        s_next: torch.Tensor,
        edge_type: torch.Tensor,
        edge_on_route_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for policy header
        :param target_agent_encoding: tensor encoding the target agent's past motion
        :param node_encodings: tensor of node encodings provided by the encoder
        :param node_masks: masks indicating whether a node exists for a given index in the tensor
        :param s_next: look-up table for next node for a given source node and edge
        :param edge_type: look-up table with edge types
        :return pi: tensor with probabilities corresponding to the policy
        """
        device = target_agent_encoding.device

        # Useful variables:
        batch_size = node_encodings.shape[0]
        max_nodes = node_encodings.shape[1]
        max_nbrs = s_next.shape[2] - 1
        node_enc_size = node_encodings.shape[2]
        target_agent_enc_size = target_agent_encoding.shape[1]

        # Gather source node encodigns, destination node encodings, edge encodings and target agent encodings.
        src_node_enc = node_encodings.unsqueeze(2).repeat(1, 1, max_nbrs, 1)
        dst_idcs = s_next[:, :, :-1].reshape(batch_size, -1).long()
        batch_idcs = (
            torch.arange(batch_size).unsqueeze(1).repeat(1, max_nodes * max_nbrs)
        )
        dst_node_enc = node_encodings[batch_idcs, dst_idcs].reshape(
            batch_size, max_nodes, max_nbrs, node_enc_size
        )
        target_agent_enc = (
            target_agent_encoding.unsqueeze(1)
            .unsqueeze(2)
            .repeat(1, max_nodes, max_nbrs, 1)
        )
        edge_enc = torch.cat(
            (
                torch.as_tensor(edge_type[:, :, :-1] == 1, device=device)
                .unsqueeze(3)
                .float(),
                torch.as_tensor(edge_type[:, :, :-1] == 2, device=device)
                .unsqueeze(3)
                .float(),
            ),
            dim=3,
        )
        enc = torch.cat((target_agent_enc, src_node_enc, dst_node_enc, edge_enc), dim=3)
        enc_goal = torch.cat(
            (target_agent_enc[:, :, 0, :], src_node_enc[:, :, 0, :]), dim=2
        )

        # Form a single batch of encodings
        masks = torch.sum(edge_enc, dim=3, keepdim=True).bool()
        masks_goal = ~node_masks.unsqueeze(-1).bool()
        enc_batched = torch.masked_select(enc, masks).reshape(
            -1, target_agent_enc_size + 2 * node_enc_size + 2
        )
        enc_goal_batched = torch.masked_select(enc_goal, masks_goal).reshape(
            -1, target_agent_enc_size + node_enc_size
        )

        # Compute scores for pi_route
        pi_ = self.pi_op(
            self.leaky_relu(self.pi_h2(self.leaky_relu(self.pi_h1(enc_batched))))
        )
        pi = torch.zeros_like(masks, dtype=pi_.dtype)
        pi = pi.masked_scatter_(masks, pi_).squeeze(-1)
        pi_goal_ = self.pi_op_goal(
            self.leaky_relu(
                self.pi_h2_goal(self.leaky_relu(self.pi_h1_goal(enc_goal_batched)))
            )
        )
        pi_goal = torch.zeros_like(masks_goal, dtype=pi_goal_.dtype)
        pi_goal = pi_goal.masked_scatter_(masks_goal, pi_goal_)

        # In original implementation (https://github.com/nachiket92/PGP/blob/main/models/aggregators/pgp.py)
        #   op_masks = torch.log(torch.as_tensor(edge_type != 0).float())
        #   pi = self.log_softmax(pi + op_masks)
        # However, if edge_type == 0, op_masks = -inf which caused problems w/ nan gradients
        # Also, for goal-conditioning, further masks are applied.
        # Therefore, in this implementation probabilities are used instead of log-probabilities.
        # Note: before returning the probabilities (output name "pi") the probabilities are converted to log-probabilities
        #       to be able to apply the original log-likelihood loss

        pi = torch.cat((pi, pi_goal), dim=-1)
        op_masks = torch.as_tensor(edge_type != 0).float()

        if self.use_route_mask:
            if self.hard_masking:
                op_masks = edge_on_route_mask * op_masks
                pi = pi * edge_on_route_mask
            else:
                pi = pi + self.route_bonus * edge_on_route_mask

        return self.masked_softmax(pi, dim=2, mask=op_masks)

    def masked_softmax(
        self,
        input: torch.Tensor,
        dim: int,
        mask: torch.Tensor = None,
        tau: float = 1.0,
        eps: float = 1e-5,
    ):
        if mask is None:
            mask = torch.as_tensor(input != 0).float()
        input_exp = torch.exp(input / tau) * mask
        input_exp_sum = input_exp.sum(dim=-1, keepdim=True)
        input_exp[:, :, -1] += torch.as_tensor(input_exp_sum == 0).float().squeeze(-1)
        input_exp_sum = input_exp.sum(dim=dim, keepdim=True)
        return input_exp / input_exp_sum
