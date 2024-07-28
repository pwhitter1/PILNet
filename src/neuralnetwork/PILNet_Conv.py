import torch
import torch.nn as nn

import dgl
import dgl.function as fn


class PILNet_Conv(nn.Module):
    """
    PIL-Net model convolutional layer.

    Parameters
    ----------
    num_node_feats : int
        Number of node features.
    num_edge_feats: int
        Number of edge features.
    hidden_dim: int
        Number of neural network hidden neurons.

    Returns
    -------
    PILNet_Conv
        PILNet_Conv convolutional layer object.
    """

    def __init__(
        self, num_node_feats: int, num_edge_feats: int, hidden_dim: int
    ) -> None:
        """Initialize PILNet_Conv object."""

        super(PILNet_Conv, self).__init__()

        num_coord_feats = 3

        activation = nn.CELU()

        self.node_expansion = nn.Sequential(
            nn.Linear(num_node_feats, hidden_dim),
            activation,
        )

        self.edge_expansion = nn.Sequential(
            nn.Linear(num_edge_feats, hidden_dim),
            activation,
        )

        self.coord_expansion = nn.Sequential(
            nn.Linear(num_coord_feats, hidden_dim),
            activation,
        )

        self.combination_reduction_nfeats = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            activation,
            nn.Linear(hidden_dim, num_node_feats),
            activation,
        )

        self.combination_reduction_efeats = nn.Sequential(
            nn.Linear(hidden_dim, num_edge_feats), activation
        )

        self.combination_reduction_cfeats = nn.Sequential(
            nn.Linear(hidden_dim, num_coord_feats), activation
        )

    def edge_udf(self, edges: dgl.udf.EdgeBatch) -> dict:
        """Perform edge convolution."""

        return {
            "x": (
                torch.abs(edges.dst["c"] - edges.src["c"])
                * (edges.src["h"] * edges.data["e"])
            )
        }

    def forward(
        self,
        bgs: dgl.DGLGraph,
        hfeats: torch.Tensor,
        cfeats: torch.Tensor,
        efeats: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """'PILNet_Conv function for forward pass through convolutional layer."""

        # Apply linear layer (increase dimension) and non-linear activation to each feature
        bgs.ndata["h"] = self.node_expansion(hfeats)
        bgs.edata["e"] = self.edge_expansion(efeats)
        bgs.ndata["c"] = self.coord_expansion(cfeats)

        # Graph convolution (update each feature based on the features' of its neighbors)
        bgs.apply_edges(self.edge_udf)
        bgs.update_all(fn.copy_e("x", "m"), fn.sum("m", "k"))

        del bgs.ndata["h"]
        del bgs.edata["x"]

        # Apply linear layer (reduce dimension to original) and non-linear activation to each feature
        # Apply a skip connection
        hfeats = hfeats + self.combination_reduction_nfeats(bgs.srcdata["k"])
        efeats = efeats + self.combination_reduction_efeats(bgs.edata["e"])
        cfeats = cfeats + self.combination_reduction_cfeats(bgs.srcdata["c"])

        del bgs.ndata["k"]
        del bgs.edata["e"]
        del bgs.ndata["c"]

        return hfeats, cfeats, efeats
