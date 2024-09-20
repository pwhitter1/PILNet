"""Contains class definition for PIL-Net model architecture.
"""

import torch
import torch.nn as nn

import dgl

from .PILNet_Conv import PILNet_Conv


class PILNet(nn.Module):
    """
    PIL-Net model network architecture.

    Parameters
    ----------
    num_node_feats : int
        Number of node features.
    num_edge_feats: int
        Number of edge features.
    hidden_dim: int
        Number of neural network hidden neurons.
    model_type: str
        Whether model enforces physics-informed constrains or not.

    Returns
    -------
    PILNet
        PILNet model object.
    """

    def __init__(
        self, num_node_feats: int, num_edge_feats: int, hidden_dim: int, model_type: str
    ) -> None:
        """Initialize PILNet object."""

        super(PILNet, self).__init__()

        self.model_type = model_type

        # Five convolutional layers per multipole type
        self.conv1_mon = PILNet_Conv(num_node_feats, num_edge_feats, hidden_dim)
        self.conv2_mon = PILNet_Conv(num_node_feats, num_edge_feats, hidden_dim)
        self.conv3_mon = PILNet_Conv(num_node_feats, num_edge_feats, hidden_dim)
        self.conv4_mon = PILNet_Conv(num_node_feats, num_edge_feats, hidden_dim)
        self.conv5_mon = PILNet_Conv(num_node_feats, num_edge_feats, hidden_dim)

        self.conv1_dip = PILNet_Conv(num_node_feats, num_edge_feats, hidden_dim)
        self.conv2_dip = PILNet_Conv(num_node_feats, num_edge_feats, hidden_dim)
        self.conv3_dip = PILNet_Conv(num_node_feats, num_edge_feats, hidden_dim)
        self.conv4_dip = PILNet_Conv(num_node_feats, num_edge_feats, hidden_dim)
        self.conv5_dip = PILNet_Conv(num_node_feats, num_edge_feats, hidden_dim)

        self.conv1_quad = PILNet_Conv(num_node_feats, num_edge_feats, hidden_dim)
        self.conv2_quad = PILNet_Conv(num_node_feats, num_edge_feats, hidden_dim)
        self.conv3_quad = PILNet_Conv(num_node_feats, num_edge_feats, hidden_dim)
        self.conv4_quad = PILNet_Conv(num_node_feats, num_edge_feats, hidden_dim)
        self.conv5_quad = PILNet_Conv(num_node_feats, num_edge_feats, hidden_dim)

        self.conv1_oct = PILNet_Conv(num_node_feats, num_edge_feats, hidden_dim)
        self.conv2_oct = PILNet_Conv(num_node_feats, num_edge_feats, hidden_dim)
        self.conv3_oct = PILNet_Conv(num_node_feats, num_edge_feats, hidden_dim)
        self.conv4_oct = PILNet_Conv(num_node_feats, num_edge_feats, hidden_dim)
        self.conv5_oct = PILNet_Conv(num_node_feats, num_edge_feats, hidden_dim)

        # Separate output layer for each multipole type
        self.out_restriction_mon = nn.Sequential(nn.Linear(num_node_feats, 1))

        self.out_restriction_dip = nn.Sequential(nn.Linear(num_node_feats, 3))

        self.out_restriction_quad = nn.Sequential(nn.Linear(num_node_feats, 6))

        self.out_restriction_oct = nn.Sequential(nn.Linear(num_node_feats, 10))

    def forward(self, bgs: dgl.DGLGraph) -> torch.Tensor:
        """PILNet function for forward pass through network."""

        """ (1) Monopoles """

        hfeats_mon = bgs.ndata["nfeats"]
        cfeats_mon = bgs.ndata["coordinates"]
        efeats_mon = bgs.edata["efeats"]

        hfeats_mon, cfeats_mon, efeats_mon = self.conv1_mon(
            bgs, hfeats_mon, cfeats_mon, efeats_mon
        )
        hfeats_mon, cfeats_mon, efeats_mon = self.conv2_mon(
            bgs, hfeats_mon, cfeats_mon, efeats_mon
        )
        hfeats_mon, cfeats_mon, efeats_mon = self.conv3_mon(
            bgs, hfeats_mon, cfeats_mon, efeats_mon
        )
        hfeats_mon, cfeats_mon, efeats_mon = self.conv4_mon(
            bgs, hfeats_mon, cfeats_mon, efeats_mon
        )
        hfeats_mon, cfeats_mon, efeats_mon = self.conv5_mon(
            bgs, hfeats_mon, cfeats_mon, efeats_mon
        )

        prediction_mon = self.out_restriction_mon(hfeats_mon)

        hfeats_mon = []
        cfeats_mon = []
        efeats_mon = []

        # Apply physics-informed constraints
        if self.model_type == "PINN":

            # Make all atomic monopole values for Hydrogen atoms positive
            H_inds = (bgs.ndata["nfeats"][:, 0] == 1).nonzero(as_tuple=True)[0]
            prediction_mon[H_inds] = abs(prediction_mon[H_inds].clone())
            H_inds = []

            net_charge_bound = 1e-2

            bgs.ndata["prediction_mon"] = prediction_mon
            unsqueezed_batch_num_nodes = torch.unsqueeze(bgs.batch_num_nodes(), 1)

            # If the absolute value of the sum exceeds the net charge bound,
            #    redistribute the values so that the total charge equals the target value
            sum_pred_total_charge = dgl.readout_nodes(
                graph=bgs, feat="prediction_mon", op="sum"
            )
            final_value = sum_pred_total_charge / unsqueezed_batch_num_nodes

            indices = torch.where(torch.abs(sum_pred_total_charge) < net_charge_bound)
            final_value[indices[0]] = 0

            prediction_mon -= dgl.broadcast_nodes(bgs, final_value)
            del sum_pred_total_charge, unsqueezed_batch_num_nodes, final_value, indices

        """ (2) Dipoles """

        hfeats_dip = bgs.ndata["nfeats"]
        cfeats_dip = bgs.ndata["coordinates"]
        efeats_dip = bgs.edata["efeats"]

        hfeats_dip, cfeats_dip, efeats_dip = self.conv1_dip(
            bgs, hfeats_dip, cfeats_dip, efeats_dip
        )
        hfeats_dip, cfeats_dip, efeats_dip = self.conv2_dip(
            bgs, hfeats_dip, cfeats_dip, efeats_dip
        )
        hfeats_dip, cfeats_dip, efeats_dip = self.conv3_dip(
            bgs, hfeats_dip, cfeats_dip, efeats_dip
        )
        hfeats_dip, cfeats_dip, efeats_dip = self.conv4_dip(
            bgs, hfeats_dip, cfeats_dip, efeats_dip
        )
        hfeats_dip, cfeats_dip, efeats_dip = self.conv5_dip(
            bgs, hfeats_dip, cfeats_dip, efeats_dip
        )

        prediction_dip = self.out_restriction_dip(hfeats_dip)

        hfeats_dip = []
        cfeats_dip = []
        efeats_dip = []

        """ (3) Quadrupoles """

        hfeats_quad = bgs.ndata["nfeats"]
        cfeats_quad = bgs.ndata["coordinates"]
        efeats_quad = bgs.edata["efeats"]

        hfeats_quad, cfeats_quad, efeats_quad = self.conv1_quad(
            bgs, hfeats_quad, cfeats_quad, efeats_quad
        )
        hfeats_quad, cfeats_quad, efeats_quad = self.conv2_quad(
            bgs, hfeats_quad, cfeats_quad, efeats_quad
        )
        hfeats_quad, cfeats_quad, efeats_quad = self.conv3_quad(
            bgs, hfeats_quad, cfeats_quad, efeats_quad
        )
        hfeats_quad, cfeats_quad, efeats_quad = self.conv4_quad(
            bgs, hfeats_quad, cfeats_quad, efeats_quad
        )
        hfeats_quad, cfeats_quad, efeats_quad = self.conv5_quad(
            bgs, hfeats_quad, cfeats_quad, efeats_quad
        )

        prediction_quad = self.out_restriction_quad(hfeats_quad)

        hfeats_quad = []
        cfeats_quad = []
        efeats_quad = []

        # Apply physics-informed constraints
        if self.model_type == "PINN":

            # Centering the mean of the diagonal elements around zero
            mean_traces = (
                prediction_quad[:, 0] + prediction_quad[:, 3] + prediction_quad[:, 5]
            ) / 3
            prediction_quad[:, 0] -= mean_traces
            prediction_quad[:, 3] -= mean_traces
            prediction_quad[:, 5] -= mean_traces

            mean_traces = []

        """ (4) Octupoles """

        hfeats_oct = bgs.ndata["nfeats"]
        cfeats_oct = bgs.ndata["coordinates"]
        efeats_oct = bgs.edata["efeats"]

        hfeats_oct, cfeats_oct, efeats_oct = self.conv1_oct(
            bgs, hfeats_oct, cfeats_oct, efeats_oct
        )
        hfeats_oct, cfeats_oct, efeats_oct = self.conv2_oct(
            bgs, hfeats_oct, cfeats_oct, efeats_oct
        )
        hfeats_oct, cfeats_oct, efeats_oct = self.conv3_oct(
            bgs, hfeats_oct, cfeats_oct, efeats_oct
        )
        hfeats_oct, cfeats_oct, efeats_oct = self.conv4_oct(
            bgs, hfeats_oct, cfeats_oct, efeats_oct
        )
        hfeats_oct, cfeats_oct, efeats_oct = self.conv5_oct(
            bgs, hfeats_oct, cfeats_oct, efeats_oct
        )

        prediction_oct = self.out_restriction_oct(hfeats_oct)

        hfeats_oct = []
        cfeats_oct = []
        efeats_oct = []

        # Apply physics-informed constraints
        if self.model_type == "PINN":

            # Centering the mean of the elements that form each trace of the tensor around zero
            x = [0, 6, 9]
            y = [3, 1, 2]
            z = [5, 8, 7]

            for i in range(len(x)):
                mean_traces = (
                    prediction_oct[:, x[i]]
                    + prediction_oct[:, y[i]]
                    + prediction_oct[:, z[i]]
                ) / 3
                prediction_oct[:, x[i]] -= mean_traces
                prediction_oct[:, y[i]] -= mean_traces
                prediction_oct[:, z[i]] -= mean_traces

            mean_traces = []

        # Return predictions
        return torch.hstack(
            (prediction_mon, prediction_dip, prediction_quad, prediction_oct)
        )
