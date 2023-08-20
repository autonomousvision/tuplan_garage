from typing import Dict, Union

import torch
import torch.nn as nn

from tuplan_garage.planning.training.modeling.models.pgp.utils import cluster_traj


class LVM(nn.Module):
    def __init__(
        self,
        agg_type: str,
        num_samples: int,
        op_len: int,
        lv_dim: int,
        encoding_size: int,
        hidden_size: int,
        num_clusters: int,
        use_ray: bool,
    ):
        """
        Latent variable conditioned decoder.
        args to include:
        agg_type: 'combined' or 'sample_specific'. Whether we have a single aggregated context vector or sample-specific
        num_samples: int Number of trajectories to sample
        op_len: int Length of predicted trajectories
        lv_dim: int Dimension of latent variable
        encoding_size: int Dimension of encoded scene + agent context
        hidden_size: int Size of output mlp hidden layer
        num_clusters: int Number of final clustered trajectories to output
        """
        super().__init__()
        if agg_type not in ["combined", "sample_specific"]:
            raise ValueError(
                f"argument agg_type has to be either 'combine' or 'sample_specific' but {agg_type} was given"
            )
        self.encoding_size = encoding_size
        self.agg_type = agg_type
        self.num_samples = num_samples
        self.op_len = op_len
        self.lv_dim = lv_dim
        self.hidden = nn.Linear(encoding_size + lv_dim, hidden_size)
        self.op_traj = nn.Linear(hidden_size, op_len * 2)
        self.leaky_relu = nn.LeakyReLU()
        self.num_clusters = num_clusters
        self.use_ray = use_ray

    def forward(self, inputs: Union[Dict, torch.Tensor]) -> Dict:
        """
        Forward pass for latent variable model.
        :param inputs: aggregated context encoding,
         shape for combined encoding: [batch_size, encoding_size]
         shape if sample specific encoding: [batch_size, num_samples, encoding_size]
        :return: predictions
        """

        if type(inputs) is torch.Tensor:
            agg_encoding = inputs
        else:
            agg_encoding = inputs["agg_encoding"]

        if self.agg_type == "combined":
            agg_encoding = agg_encoding.unsqueeze(1).repeat(1, self.num_samples, 1)
        else:
            if (
                len(agg_encoding.shape) != 3
                or agg_encoding.shape[1] != self.num_samples
            ):
                raise Exception(
                    "Expected "
                    + str(self.num_samples)
                    + "encodings for each train/val data"
                )

        # Sample latent variable and concatenate with aggregated encoding
        batch_size = agg_encoding.shape[0]
        z = torch.randn(
            batch_size, self.num_samples, self.lv_dim, device=agg_encoding.device
        )
        agg_encoding = torch.cat((agg_encoding, z), dim=2)
        h = self.leaky_relu(self.hidden(agg_encoding))

        # Output trajectories
        traj = self.op_traj(h)
        traj = traj.reshape(batch_size, self.num_samples, self.op_len, 2)

        # Cluster
        traj_clustered, probs = cluster_traj(
            k=self.num_clusters, traj=traj, random_state=0, use_ray=self.use_ray
        )

        predictions = {
            "traj": traj_clustered,
            "probs": probs,
            "traj_none_clustered": traj,
        }

        if type(inputs) is dict:
            for key, val in inputs.items():
                if key != "agg_encoding":
                    predictions[key] = val

        return predictions
