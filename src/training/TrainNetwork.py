"""Train a PILNet model using the training and validation dataset splits.
"""

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

import dgl
from dgl.data.utils import load_graphs

import random
import time
import datetime

from typing import Callable

from ..neuralnetwork.PILNet import PILNet


def load_format_datasets(
    device: torch.device,
    train_path: str,
    validation_path: str,
    model_precision: torch.dtype,
) -> tuple[list, list, float, float, float, float]:
    """Load, change precision, detrace, and obtain the loss function weights of the graphs."""

    traingraphs, validationgraphs = load_train_validation_datasets(
        train_path, validation_path
    )

    """ Training graphs """

    batch_traingraphs = dgl.batch(traingraphs)
    traingraphs = []

    # Convert graph data/labels to have specified precision
    batch_traingraphs = convert_to_model_precision(batch_traingraphs, model_precision)
    # Detrace the quadrupole and octupole labels
    batch_traingraphs = detrace_atomic_quadrupole_labels(batch_traingraphs)
    batch_traingraphs = detrace_atomic_octupole_labels(batch_traingraphs)

    # Compute the multipole loss function weights
    monopole_iqr = get_loss_func_weight(batch_traingraphs.ndata["label_monopoles"])  # type: ignore
    dipole_iqr = get_loss_func_weight(batch_traingraphs.ndata["label_dipoles"])  # type: ignore
    quadrupole_iqr = get_loss_func_weight(
        batch_traingraphs.ndata["label_quadrupoles"]  # type: ignore
    )  
    octupole_iqr = get_loss_func_weight(batch_traingraphs.ndata["label_octupoles"])  # type: ignore

    sum_iqr = monopole_iqr + dipole_iqr + quadrupole_iqr + octupole_iqr

    monopole_weight = monopole_iqr / sum_iqr
    dipole_weight = dipole_iqr / sum_iqr
    quadrupole_weight = quadrupole_iqr / sum_iqr
    octupole_weight = octupole_iqr / sum_iqr

    # Put graphs on GPU
    batch_traingraphs = batch_traingraphs.to(device)

    traingraphs = dgl.unbatch(batch_traingraphs)
    batch_traingraphs = []

    """ Validation graphs """

    batch_validationgraphs = dgl.batch(validationgraphs)
    validationgraphs = []

    # Convert graph data/labels to have specified precision
    batch_validationgraphs = convert_to_model_precision(
        batch_validationgraphs, model_precision
    )
    # Detrace the quadrupole and octupole labels
    batch_validationgraphs = detrace_atomic_quadrupole_labels(batch_validationgraphs)
    batch_validationgraphs = detrace_atomic_octupole_labels(batch_validationgraphs)

    batch_validationgraphs = batch_validationgraphs.to(device)
    validationgraphs = dgl.unbatch(batch_validationgraphs)
    batch_validationgraphs = []

    return (
        traingraphs,
        validationgraphs,
        monopole_weight,
        dipole_weight,
        quadrupole_weight,
        octupole_weight,
    )


def load_train_validation_datasets(
    train_path: str, validation_path: str
) -> tuple[list, list]:
    """Load graphs."""
    traingraphs = load_graphs(train_path)
    traingraphs = traingraphs[0]

    validationgraphs = load_graphs(validation_path)
    validationgraphs = validationgraphs[0]

    return traingraphs, validationgraphs


def convert_to_model_precision(
    batch_graphs: dgl.DGLGraph, model_precision: torch.dtype
) -> dgl.DGLGraph:
    """Convert data and labels to specified model precision."""

    batch_graphs.ndata["nfeats"] = batch_graphs.ndata["nfeats"].to(model_precision)  # type: ignore
    batch_graphs.edata["efeats"] = batch_graphs.edata["efeats"].to(model_precision)  # type: ignore

    batch_graphs.ndata["label_monopoles"] = batch_graphs.ndata[
        "label_monopoles"
    ].to(  # type: ignore
        model_precision
    )
    batch_graphs.ndata["label_dipoles"] = batch_graphs.ndata["label_dipoles"].to(  # type: ignore
        model_precision
    )
    batch_graphs.ndata["label_quadrupoles"] = batch_graphs.ndata[
        "label_quadrupoles"
    ].to(  # type: ignore
        model_precision
    )  
    batch_graphs.ndata["label_octupoles"] = batch_graphs.ndata[
        "label_octupoles"
    ].to(  # type: ignore
        model_precision
    )

    batch_graphs.ndata["coordinates"] = batch_graphs.ndata["coordinates"].to(  # type: ignore
        model_precision
    )
    batch_graphs.ndata["relative_coordinates"] = batch_graphs.ndata[
        "relative_coordinates"
    ].to(model_precision)  # type: ignore  

    return batch_graphs


def detrace_quadrupoles(quadrupoles: torch.Tensor) -> torch.Tensor:
    """Detrace the quadrupoles.
    Each row is organized as: xx xy xz yy yz zz."""

    inds = [0, 3, 5]
    trace = torch.sum(quadrupoles[:, inds], dim=1)

    quadrupoles[:, 0] -= trace / 3
    quadrupoles[:, 3] -= trace / 3
    quadrupoles[:, 5] -= trace / 3

    return quadrupoles


def detrace_atomic_quadrupole_labels(graphs: dgl.DGLGraph) -> dgl.DGLGraph:
    """Detrace atomic quadrupole labels from DGL graphs.
    Each vector is organized as: xx xy xz yy yz zz."""

    quadrupoles = detrace_quadrupoles(graphs.ndata["label_quadrupoles"])  # type: ignore
    graphs.ndata["label_quadrupoles"] = quadrupoles

    return graphs


def detrace_octupoles(octupoles: torch.Tensor) -> torch.Tensor:
    """Detrace the octupoles.
    Each row is organized as: xxx, xxy, xxz, xyy, xyz, xzz, yyy, yyz, yzz, zzz."""

    x = [0, 6, 9]
    y = [3, 1, 2]
    z = [5, 8, 7]

    for i in range(len(x)):
        inds = [x[i], y[i], z[i]]
        trace = torch.sum(octupoles[:, inds], dim=1)

        octupoles[:, x[i]] -= trace / 3
        octupoles[:, y[i]] -= trace / 3
        octupoles[:, z[i]] -= trace / 3

    return octupoles


def detrace_atomic_octupole_labels(graphs: dgl.DGLGraph) -> dgl.DGLGraph:
    """Detrace atomic octupole labels from DGL graphs.
    Each vector is organized as: xxx, xxy, xxz, xyy, xyz, xzz, yyy, yyz, yzz, zzz."""

    octupoles = detrace_octupoles(graphs.ndata["label_octupoles"])  # type: ignore
    graphs.ndata["label_octupoles"] = octupoles

    return graphs


def get_loss_func_weight(data: torch.Tensor) -> float:
    """Get multipole loss function weight based on label distribution."""

    iqr = 0

    for i in range(len(data[0])):
        # Quartiles
        q3, q1 = np.percentile(data[:, i], [75, 25], interpolation="midpoint")  # type: ignore
        iqr += q3 - q1

    avg_iqr = iqr / len(data[0])
    return avg_iqr


def set_network_parameters(
    device: torch.device,
    traingraphs: list,
    validationgraphs: list,
    num_node_feats: int,
    num_edge_feats: int,
    model_type: str,
    model_precision: torch.dtype,
) -> tuple[
    PILNet,
    list,
    dgl.dataloading.GraphDataLoader,
    dgl.dataloading.GraphDataLoader,
    Callable,
    list,
    list,
    list,
    list,
]:
    """Specify parameters used in PIL-Net."""

    # Specify general network parameters
    hidden_dim = 256
    train_bsz = 256
    validation_bsz = 256
    loss_function = nn.MSELoss()
    activation = nn.CELU()
    learnrate = 1e-3

    # Data loaders
    train_dataloader = dgl.dataloading.GraphDataLoader(
        dataset=traingraphs,
        use_ddp=False,
        batch_size=train_bsz,
        drop_last=True,
        shuffle=True,
    )

    validation_dataloader = dgl.dataloading.GraphDataLoader(
        dataset=validationgraphs,
        use_ddp=False,
        batch_size=validation_bsz,
        drop_last=False,
        shuffle=False,
    )

    traingraphs, validationgraphs = [], []

    # PIL-Net model
    model = (
        PILNet(num_node_feats, num_edge_feats, hidden_dim, model_type)
        .to(model_precision)
        .to(device)
    )

    # Create separate optimizers for each multipole type
    params_optimizer_mon = []
    params_optimizer_dip = []
    params_optimizer_quad = []
    params_optimizer_oct = []

    for name, param in model.named_parameters():
        if "mon" in name:
            params_optimizer_mon.append(param)
        if "dip" in name:
            params_optimizer_dip.append(param)
        if "quad" in name:
            params_optimizer_quad.append(param)
        if "oct" in name:
            params_optimizer_oct.append(param)

    optimizer_mon = optim.Adam(params_optimizer_mon, lr=learnrate)  # type: ignore
    optimizer_dip = optim.Adam(params_optimizer_dip, lr=learnrate)  # type: ignore
    optimizer_quad = optim.Adam(params_optimizer_quad, lr=learnrate)  # type: ignore
    optimizer_oct = optim.Adam(params_optimizer_oct, lr=learnrate)  # type: ignore

    # Specify scheduler parameters and create separate one for each multipole type
    lr_factor = 0.5
    lr_threshold = 1e-4
    lr_patience = 5
    min_lr = 1e-5

    scheduler_mon = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer_mon,
        factor=lr_factor,
        threshold=lr_threshold,
        patience=lr_patience,
        min_lr=min_lr,
        verbose=True,  # type: ignore
    )
    scheduler_dip = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer_dip,
        factor=lr_factor,
        threshold=lr_threshold,
        patience=lr_patience,
        min_lr=min_lr,
        verbose=True,  # type: ignore
    )
    scheduler_quad = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer_quad,
        factor=lr_factor,
        threshold=lr_threshold,
        patience=lr_patience,
        min_lr=min_lr,
        verbose=True,  # type: ignore
    )
    scheduler_oct = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer_oct,
        factor=lr_factor,
        threshold=lr_threshold,
        patience=lr_patience,
        min_lr=min_lr,
        verbose=True,  # type: ignore
    )

    # Store these values to return from function
    optimizer_list = [optimizer_mon, optimizer_dip, optimizer_quad, optimizer_oct]
    scheduler_list = [scheduler_mon, scheduler_dip, scheduler_quad, scheduler_oct]

    model_parameters = [num_node_feats, num_edge_feats, hidden_dim, activation]
    optimizer_parameters = [learnrate]
    scheduler_parameters = [lr_factor, lr_threshold, lr_patience]

    return (
        model,
        model_parameters,
        train_dataloader,
        validation_dataloader,
        loss_function,
        optimizer_list,
        optimizer_parameters,
        scheduler_list,
        scheduler_parameters,
    )


def train_network(
    device: torch.device,
    model: PILNet,
    model_parameters: list,
    train_dataloader: dgl.dataloading.GraphDataLoader,
    validation_dataloader: dgl.dataloading.GraphDataLoader,
    maxtraintime: float,
    loss_function: Callable,
    optimizer_list: list,
    optimizer_parameters: list,
    scheduler_list: list,
    scheduler_parameters: list,
    bestmodel_path: str,
    monopole_weight: float,
    dipole_weight: float,
    quadrupole_weight: float,
    octupole_weight: float,
    model_type: str,
    model_precision: torch.dtype,
) -> None:
    """Train neural network and save best model."""

    starttime = time.time()

    model.train()

    # Set network parameters
    num_epochs = 200
    smallest_valloss = float("inf")

    # Extract multipole-specific optimizers
    optimizer_mon = optimizer_list[0]
    optimizer_dip = optimizer_list[1]
    optimizer_quad = optimizer_list[2]
    optimizer_oct = optimizer_list[3]

    # Extract multipole-specific schedulers
    scheduler_mon = scheduler_list[0]
    scheduler_dip = scheduler_list[1]
    scheduler_quad = scheduler_list[2]
    scheduler_oct = scheduler_list[3]

    # Train neural network
    for epoch in range(num_epochs):

        totalnumlabels = 0.0
        count = 0.0

        epoch_loss = 0.0
        epoch_monopoles_loss = 0.0
        epoch_dipoles_loss = 0.0
        epoch_quadrupoles_loss = 0.0
        epoch_octupoles_loss = 0.0

        epoch_monopoles_penalty_total = 0.0
        epoch_monopoles_penalty_H = 0.0
        epoch_quadrupoles_penalty = 0.0
        epoch_octupoles_penalty = 0.0

        for iter, bgs in enumerate(train_dataloader):

            # Run graphs through model and obtain predictions
            predlabels = model(bgs)

            numlabels = len(predlabels)
            totalnumlabels += numlabels

            # Compute loss for each multipole type
            monopoles_loss = loss_function(
                bgs.ndata["label_monopoles"], predlabels[:, 0:1]
            )
            dipoles_loss = loss_function(bgs.ndata["label_dipoles"], predlabels[:, 1:4])
            quadrupoles_loss = loss_function(
                bgs.ndata["label_quadrupoles"], predlabels[:, 4:10]
            )
            octupoles_loss = loss_function(
                bgs.ndata["label_octupoles"], predlabels[:, 10:20]
            )

            # Compute scaled loss
            loss = (
                ((1 / monopole_weight) * monopoles_loss)
                + ((1 / dipole_weight) * dipoles_loss)
                + ((1 / quadrupole_weight) * quadrupoles_loss)
                + ((1 / octupole_weight) * octupoles_loss)
            )

            # Only computing PINN penalties for diagnostic purposes
            # (does not affect network training)
            # Choose one random graph per epoch as penalty representative
            random_graph_index = random.randint(0, bgs.batch_size - 1)
            (
                monopole_penalty_total,
                monopole_penalty_H,
                quadrupole_penalty,
                octupole_penalty,
            ) = PINN_penalties(
                bgs, predlabels, random_graph_index, model_precision, device
            )
            count += 1

            # Update running penalty total
            epoch_monopoles_penalty_total += monopole_penalty_total.detach().item()
            epoch_monopoles_penalty_H += monopole_penalty_H.detach().item()
            epoch_quadrupoles_penalty += quadrupole_penalty.detach().item()
            epoch_octupoles_penalty += octupole_penalty.detach().item()

            del (
                monopole_penalty_total,
                monopole_penalty_H,
                quadrupole_penalty,
                octupole_penalty,
            )
            del bgs, predlabels

            # Update network gradients
            optimizer_mon.zero_grad()
            optimizer_dip.zero_grad()
            optimizer_quad.zero_grad()
            optimizer_oct.zero_grad()

            loss.backward()

            optimizer_mon.step()
            optimizer_dip.step()
            optimizer_quad.step()
            optimizer_oct.step()

            # Update running loss total
            epoch_monopoles_loss += (monopoles_loss.detach().item()) * numlabels
            epoch_dipoles_loss += (dipoles_loss.detach().item()) * numlabels
            epoch_quadrupoles_loss += (quadrupoles_loss.detach().item()) * numlabels
            epoch_octupoles_loss += (octupoles_loss.detach().item()) * numlabels

            loss, monopoles_loss, dipoles_loss, quadrupoles_loss, octupoles_loss = (
                0,
                0,
                0,
                0,
                0,
            )

        # Evaluate validation set on model
        (
            validation_monopoles_loss,
            validation_dipoles_loss,
            validation_quadrupoles_loss,
            validation_octupoles_loss,
        ) = validation(model, validation_dataloader, loss_function, model_type)

        # Compute scaled validation loss
        validation_loss = (
            ((1 / monopole_weight) * validation_monopoles_loss)
            + ((1 / dipole_weight) * validation_dipoles_loss)
            + ((1 / quadrupole_weight) * validation_quadrupoles_loss)
            + ((1 / octupole_weight) * validation_octupoles_loss)
        )

        # Update network loss scheduler
        scheduler_mon.step((1 / monopole_weight) * validation_monopoles_loss)
        scheduler_dip.step((1 / dipole_weight) * validation_dipoles_loss)
        scheduler_quad.step((1 / quadrupole_weight) * validation_quadrupoles_loss)
        scheduler_oct.step((1 / octupole_weight) * validation_octupoles_loss)

        # Compute unsweighted validation loss
        validation_loss_unweighted = (
            validation_monopoles_loss
            + validation_dipoles_loss
            + validation_quadrupoles_loss
            + validation_octupoles_loss
        )

        # Average epoch loss across epoch molecules
        epoch_monopoles_loss /= totalnumlabels
        epoch_dipoles_loss /= totalnumlabels
        epoch_quadrupoles_loss /= totalnumlabels
        epoch_octupoles_loss /= totalnumlabels

        # Compute the unweighted and weighted epoch loss
        epoch_loss = (
            epoch_monopoles_loss
            + epoch_dipoles_loss
            + epoch_quadrupoles_loss
            + epoch_octupoles_loss
        )
        epoch_loss_weighted = (
            ((1 / monopole_weight) * epoch_monopoles_loss)
            + ((1 / dipole_weight) * epoch_dipoles_loss)
            + ((1 / quadrupole_weight) * epoch_quadrupoles_loss)
            + ((1 / octupole_weight) * epoch_octupoles_loss)
        )

        # Average pinn penalties across epoch molecules
        epoch_monopoles_penalty_total /= count
        epoch_monopoles_penalty_H /= count
        epoch_quadrupoles_penalty /= count
        epoch_octupoles_penalty /= count

        # Printing unweighted and weighted stats to users
        print("EPOCH {}\n".format(epoch))

        print("Unweighted:")
        print(
            "Training | epoch loss {:.8f}, monopole loss {:.8f}, dipole loss {:.8f},"
            " quadrupole loss {:.8f}, octupoles loss {:.8f}".format(
                epoch_loss,
                epoch_monopoles_loss,
                epoch_dipoles_loss,
                epoch_quadrupoles_loss,
                epoch_octupoles_loss,
            ),
            flush=True,
        )
        print(
            "Training | mon penalty total {:.8f}, mon penalty H {:.8f},"
            " quad penalty {:.8f}, oct penalty {:.8f}".format(
                epoch_monopoles_penalty_total,
                epoch_monopoles_penalty_H,
                epoch_quadrupoles_penalty,
                epoch_octupoles_penalty,
            ),
            flush=True,
        )
        print(
            "Validation | epoch loss {:.8f}, monopole loss {:.8f}, dipole loss {:.8f},"
            " quadrupole loss {:.8f}, octupole loss {:.8f}\n".format(
                validation_loss_unweighted,
                validation_monopoles_loss,
                validation_dipoles_loss,
                validation_quadrupoles_loss,
                validation_octupoles_loss,
            ),
            flush=True,
        )

        print("Weighted:")
        print(
            "Training | epoch loss {:.8f}, monopole loss {:.8f}, dipole loss {:.8f},"
            " quadrupole loss {:.8f}, octupoles loss {:.8f}".format(
                epoch_loss_weighted,
                (1 / monopole_weight) * epoch_monopoles_loss,
                (1 / dipole_weight) * epoch_dipoles_loss,
                (1 / quadrupole_weight) * epoch_quadrupoles_loss,
                (1 / octupole_weight) * epoch_octupoles_loss,
            ),
            flush=True,
        )
        print(
            "Training | mon penalty total {:.8f}, mon penalty H {:.8f},"
            " quad penalty {:.8f}, oct penalty {:.8f}".format(
                (1 / monopole_weight) * epoch_monopoles_penalty_total,
                (1 / monopole_weight) * epoch_monopoles_penalty_H,
                (1 / quadrupole_weight) * epoch_quadrupoles_penalty,
                (1 / octupole_weight) * epoch_octupoles_penalty,
            ),
            flush=True,
        )
        print(
            "Validation | epoch loss {:.8f}, monopole loss {:.8f}, dipole loss {:.8f},"
            " quadrupole loss {:.8f}, octupole loss {:.8f}\n".format(
                validation_loss,
                (1 / monopole_weight) * validation_monopoles_loss,
                (1 / dipole_weight) * validation_dipoles_loss,
                (1 / quadrupole_weight) * validation_quadrupoles_loss,
                (1 / octupole_weight) * validation_octupoles_loss,
            ),
            flush=True,
        )

        # Delete loss and penalty related values for the epoch
        del (
            epoch_loss,
            epoch_monopoles_loss,
            epoch_dipoles_loss,
            epoch_quadrupoles_loss,
            epoch_octupoles_loss,
        )
        del (
            epoch_monopoles_penalty_total,
            epoch_monopoles_penalty_H,
            epoch_quadrupoles_penalty,
            epoch_octupoles_penalty,
        )
        del (
            validation_monopoles_loss,
            validation_dipoles_loss,
            validation_quadrupoles_loss,
            validation_octupoles_loss,
        )

        # Save the current model if it has smallest weighted validation loss so far
        if validation_loss < smallest_valloss:
            smallest_valloss = validation_loss

            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "optimizer_mon_state_dict": optimizer_mon.state_dict(),
                    "optimizer_dip_state_dict": optimizer_dip.state_dict(),
                    "optimizer_quad_state_dict": optimizer_quad.state_dict(),
                    "optimizer_oct_state_dict": optimizer_oct.state_dict(),
                    "scheduler_mon_state_dict": scheduler_mon.state_dict(),
                    "scheduler_dip_state_dict": scheduler_dip.state_dict(),
                    "scheduler_quad_state_dict": scheduler_quad.state_dict(),
                    "scheduler_oct_state_dict": scheduler_oct.state_dict(),
                    "smallest_valloss": smallest_valloss,
                    "model_parameters": model_parameters,
                    "optimizer_parameters": optimizer_parameters,
                    "scheduler_parameters": scheduler_parameters,
                },
                bestmodel_path,
            )

            print("**Best model updated**\n", flush=True)

        # End training if over time limit
        endtime = time.time()
        if (endtime - starttime) >= maxtraintime:
            print("Maximum training time reached - stopping training.")
            break

    endtime = time.time()
    print("Time elapsed: {} seconds".format(endtime - starttime), flush=True)


def validation(
    model: PILNet,
    validation_data_loader: dgl.dataloading.GraphDataLoader,
    loss_func: Callable,
    model_type: str,
) -> tuple[float, float, float, float]:
    """Evaluate validation set on model."""

    model.eval()

    totalnumlabels = 0.0

    epoch_monopoles_loss = 0.0
    epoch_dipoles_loss = 0.0
    epoch_quadrupoles_loss = 0.0
    epoch_octupoles_loss = 0.0

    for iter, valbgs in enumerate(validation_data_loader):

        # Obtain prediction labels
        with torch.no_grad():
            predlabels = model(valbgs)

        numlabels = len(predlabels)
        totalnumlabels += numlabels

        # Compute multipole-type specific loss
        monopoles_loss = loss_func(valbgs.ndata["label_monopoles"], predlabels[:, 0:1])
        dipoles_loss = loss_func(valbgs.ndata["label_dipoles"], predlabels[:, 1:4])
        quadrupoles_loss = loss_func(
            valbgs.ndata["label_quadrupoles"], predlabels[:, 4:10]
        )
        octupoles_loss = loss_func(
            valbgs.ndata["label_octupoles"], predlabels[:, 10:20]
        )

        valbgs, predlabels = [], []

        # Update running epoch loss tally
        epoch_monopoles_loss += (monopoles_loss.detach().item()) * numlabels
        epoch_dipoles_loss += (dipoles_loss.detach().item()) * numlabels
        epoch_quadrupoles_loss += (quadrupoles_loss.detach().item()) * numlabels
        epoch_octupoles_loss += (octupoles_loss.detach().item()) * numlabels

        _, monopoles_loss, dipoles_loss, quadrupoles_loss, octupoles_loss = (
            0,
            0,
            0,
            0,
            0,
        )

    # Average epoch loss across epoch molecules
    epoch_monopoles_loss /= totalnumlabels
    epoch_dipoles_loss /= totalnumlabels
    epoch_quadrupoles_loss /= totalnumlabels
    epoch_octupoles_loss /= totalnumlabels

    model.train()

    return (
        epoch_monopoles_loss,
        epoch_dipoles_loss,
        epoch_quadrupoles_loss,
        epoch_octupoles_loss,
    )


def PINN_penalties(
    bgs: dgl.DGLGraph,
    predlabels: torch.Tensor,
    random_graph_index: int,
    model_precision: torch.dtype,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Obtain pinn penalty error for diagnostic purposes (does not impact network training)."""

    # Obtain graph specified by random_graph_index
    batch_num_nodes = bgs.batch_num_nodes()
    graph_node_index_start = sum(batch_num_nodes[0:random_graph_index])
    num_nodes = batch_num_nodes[random_graph_index]
    graph_node_index_end = graph_node_index_start + num_nodes

    # Initialize penalty values
    monopole_penalty_H = torch.zeros(1, dtype=model_precision).to(device)
    quadrupole_penalty = torch.zeros(1, dtype=model_precision).to(device)
    octupole_penalty = torch.zeros(1, dtype=model_precision).to(device)

    """ Monopole penalties """

    # Total charge
    pred_monopole_vectors = predlabels[graph_node_index_start:graph_node_index_end, 0]
    pred_monopole_sum = torch.sum(pred_monopole_vectors)

    true_monopole_vectors = bgs.ndata["label_monopoles"][
        graph_node_index_start:graph_node_index_end, :
    ]
    true_monopole_sum = torch.sum(true_monopole_vectors)
    monopole_penalty_total = abs(true_monopole_sum - pred_monopole_sum)
    monopole_penalty_total /= num_nodes

    # Hydrogen
    nfeats = bgs.ndata["nfeats"][graph_node_index_start:graph_node_index_end, :]

    H_inds = (nfeats[:, 0] == 1).nonzero(as_tuple=True)[0]
    for ind in H_inds:
        if pred_monopole_vectors[ind] < 0:
            monopole_penalty_H += abs(pred_monopole_vectors[ind])

    if len(H_inds) > 0:
        monopole_penalty_H /= len(H_inds)

    del nfeats, true_monopole_vectors, pred_monopole_vectors

    """ Quadrupole and octupole penalties """
    pred_quad_vectors = predlabels[graph_node_index_start:graph_node_index_end, 4:10]
    pred_oct_vectors = predlabels[graph_node_index_start:graph_node_index_end, 10:20]

    for i in range(num_nodes):
        pred_quad = pred_quad_vectors[i]
        quadrupole_penalty += abs(pred_quad[0] + pred_quad[3] + pred_quad[5])

        pred_oct = pred_oct_vectors[i]
        octupole_penalty += abs(pred_oct[0] + pred_oct[6] + pred_oct[9])

    quadrupole_penalty /= num_nodes
    octupole_penalty /= num_nodes

    del pred_quad_vectors, pred_oct_vectors, pred_quad, pred_oct

    return (
        monopole_penalty_total,
        monopole_penalty_H,
        quadrupole_penalty,
        octupole_penalty,
    )


def main(read_filepath: str, save_filepath: str):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # train_path = read_filepath + "traindata.bin"
    # validation_path = read_filepath + "validationdata.bin"

    train_path = read_filepath + "testdata.bin"
    validation_path = read_filepath + "testdata.bin"

    # Save the best model to specified folder, including current timestamp
    bestmodel_path = (
        save_filepath
        + "pilnet_model_"
        + str(datetime.datetime.now().timestamp()).replace(" ", "_")
        + ".bin"
    )

    # Details related to our QMDFAM graphs (desired number of features to use)
    num_node_feats = 16
    num_edge_feats = 26

    # Details related to PIL-Net experimental set-up
    maxtraintime = 24  # in hours
    maxtraintime *= 3600  # in seconds

    model_type = "PINN"
    # model_type = "Non-PINN"

    model_precision = torch.float32

    # Load datasets
    (
        traingraphs,
        validationgraphs,
        monopole_weight,
        dipole_weight,
        quadrupole_weight,
        octupole_weight,
    ) = load_format_datasets(device, train_path, validation_path, model_precision)

    # Set neural network parameters
    (
        model,
        model_parameters,
        train_dataloader,
        validation_dataloader,
        loss_function,
        optimizer_list,
        optimizer_parameters,
        scheduler_list,
        scheduler_parameters,
    ) = set_network_parameters(
        device,
        traingraphs,
        validationgraphs,
        num_node_feats,
        num_edge_feats,
        model_type,
        model_precision,
    )

    # Train neural network
    train_network(
        device,
        model,
        model_parameters,
        train_dataloader,
        validation_dataloader,
        maxtraintime,
        loss_function,
        optimizer_list,
        optimizer_parameters,
        scheduler_list,
        scheduler_parameters,
        bestmodel_path,
        monopole_weight,
        dipole_weight,
        quadrupole_weight,
        octupole_weight,
        model_type,
        model_precision,
    )
