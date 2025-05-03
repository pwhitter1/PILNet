"""Contains class definition for PIL-Net model convolutional layer.
"""

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

    def edge_udf(self, edges: dgl.udf.EdgeBatch) -> dict[str, torch.Tensor]:  # type: ignore
        """
        Function to perform message passing across the edges of the graphs.
        
            With respect to each node, the created feature is a function of the following: 
                a) the node features of the neighboring nodes
                b) the adjacent edge features
                c) the distance between itself and its neighbors

        Parameters
        ----------
        edges: dgl.udf.EdgeBatch
            Batch of DGL graph edges.

        Returns
        -------
        dict[str, torch.Tensor]: 
            Contains updated feature following the graph convolution.
        
        """

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
        """
        PIL-Net function for forward pass through convolutional layer.

        Parameters
        ----------
        bgs: dgl.DGLGraph
            Batched group of DGL graphs.
        hfeats: torch.Tensor
            Node features associated with batched graphs.
        cfeats: torch.Tensor
            Coordinate features associated with batched graphs.
        efeats: torch.Tensor
            Edge features associated with batched graphs.


        Returns
        -------
        hfeats: torch.Tensor
            Updated node features following graph convolution.
        cfeats: torch.Tensor
            Updated coordinate features following graph convolution.
        efeats: torch.Tensor
            Updated edge features following graph convolution.

        """

        # Apply linear layer (increase dimension) and non-linear activation to each feature
        bgs.ndata["h"] = self.node_expansion(hfeats)
        bgs.edata["e"] = self.edge_expansion(efeats)
        bgs.ndata["c"] = self.coord_expansion(cfeats)

        # Graph convolution - message passing across the edges
        # (Each node feature will be updated based on this combination of the features' of its neighbors)
        bgs.apply_edges(self.edge_udf)
        bgs.update_all(fn.copy_e("x", "m"), fn.sum("m", "k"))  # type: ignore

        del bgs.ndata["h"]
        del bgs.edata["x"]

        # Apply linear layer (reduce dimension to original)
        # and non-linear activation function to each feature.
        # Apply a skip connection.
        hfeats = hfeats + self.combination_reduction_nfeats(bgs.srcdata["k"])
        efeats = efeats + self.combination_reduction_efeats(bgs.edata["e"])
        cfeats = cfeats + self.combination_reduction_cfeats(bgs.srcdata["c"])

        del bgs.ndata["k"]
        del bgs.edata["e"]
        del bgs.ndata["c"]

        return hfeats, cfeats, efeats
