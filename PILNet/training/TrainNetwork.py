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

from ..model.PILNet import PILNet


def load_format_datasets(
    device: torch.device,
    train_path: str,
    validation_path: str,
    model_type: str,
    model_precision: torch.dtype,
) -> tuple[list[dgl.DGLGraph], list[dgl.DGLGraph], float, float, float, float]:
    """
    Function to load, change precision, detrace, and obtain the loss function weights of the graphs.

    Parameters
    -------
    device: torch.device
        Either a cpu or cuda device.
    train_path: str
        Path to training set graphs.
    validation_path: str
        Path to validation set graphs.
    model_type: str
        Whether the model is a "PINN" or Non-PINN" model.
    model_precision: torch.dtype
        Precision to apply to graph features and labels.

    Returns
    ----------
    traingraphs: list[dgl.DGLGraph]
        List of training set graphs.
    validationgraphs: list[dgl.DGLGraph]
        List of validation graphs.
    monopole_weight: float
        Weight associated with the atomic monopoles for the neural network loss function.
    dipole_weight: float
        Weight associated with the atomic dipoles for the neural network loss function.
    quadrupole_weight: float
         Weight associated with the atomic quadrupoles for the neural network loss function.
    octupole_weight: float
         Weight associated with the atomic octupoles for the neural network loss function.

    """

    traingraphs, validationgraphs = load_train_validation_datasets(
        train_path, validation_path
    )

    print("Formatting dataset...", flush=True)

    """ Training graphs """

    batch_traingraphs = dgl.batch(traingraphs)
    traingraphs = []

    # Convert graph data/labels to have specified precision
    batch_traingraphs = convert_to_model_precision(batch_traingraphs, model_precision)
    # Detrace the quadrupole and octupole labels
    batch_traingraphs = detrace_atomic_quadrupole_labels(batch_traingraphs)
    batch_traingraphs = detrace_atomic_octupole_labels(batch_traingraphs)

    # Compute the multipole loss function weights
    if model_type == "PINN":
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

        print(f"Weights (mdqo): {monopole_weight}, {dipole_weight}, {quadrupole_weight}, {octupole_weight}", flush=True)

    # Else set all the weights to 1
    else:
        monopole_weight, dipole_weight, quadrupole_weight, octupole_weight = 1, 1, 1, 1


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
) -> tuple[list[dgl.DGLGraph], list[dgl.DGLGraph]]:
    """
    Function to load training set and validation set graphs from file.

    Parameters
    -------
    train_path: str
        Filepath to training set graphs.
    validation_path: str
        Filepath to validation set graphs.

    Returns
    ----------
    traingraphs: list[dgl.DGLGraph]
        List of training set graphs.
    validationgraphs: list[dgl.DGLGraph]
        List of validation set graphs.

    """

    print("Reading in dataset...", flush=True)

    traingraphs = load_graphs(train_path)
    traingraphs = traingraphs[0]

    validationgraphs = load_graphs(validation_path)
    validationgraphs = validationgraphs[0]

    return traingraphs, validationgraphs


def convert_to_model_precision(
    batch_graphs: dgl.DGLGraph, model_precision: torch.dtype
) -> dgl.DGLGraph:
    """
    Function to data and labels to the specified model precision.

    Parameters
    -------
    batch_graphs: dgl.DGLGraph
        Grouping of DGL graphs.
    model_precision: torch.dtype
        Precision to apply to graph features and labels.

    Returns
    ----------
    batch_graphs: dgl.DGLGraph
        Grouping of DGL graphs with precision applied.

    """

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

    return batch_graphs


def detrace_quadrupoles(quadrupoles: torch.Tensor) -> torch.Tensor:
    """
    Function to detrace the quadrupole moments.
        Each row is organized as: xx xy xz yy yz zz.

    Parameters
    -------
    quadrupoles: torch.Tensor
        Tensor of quadrupole moment values.

    Returns
    ----------
    quadrupoles: torch.Tensor
        Tensor of detraced quadrupole moment values.

    """

    inds = [0, 3, 5]
    trace = torch.sum(quadrupoles[:, inds], dim=1)

    quadrupoles[:, 0] -= trace / 3
    quadrupoles[:, 3] -= trace / 3
    quadrupoles[:, 5] -= trace / 3

    return quadrupoles


def detrace_atomic_quadrupole_labels(graphs: dgl.DGLGraph) -> dgl.DGLGraph:
    """
    Function to detrace the atomic quadrupole moment labels from DGL graphs.

    Parameters
    -------
    graphs: dgl.DGLGraph
        Graphs containing the quadrupole moment labels to detrace.

    Returns
    ----------
    graphs: dgl.DGLGraph
        Graphs containing the detraced quadrupole moment labels.

    """

    quadrupoles = detrace_quadrupoles(graphs.ndata["label_quadrupoles"])  # type: ignore
    graphs.ndata["label_quadrupoles"] = quadrupoles

    return graphs


def detrace_octupoles(octupoles: torch.Tensor) -> torch.Tensor:
    """
    Function to detrace the octupole moments.
        Each row is organized as: xxx, xxy, xxz, xyy, xyz, xzz, yyy, yyz, yzz, zzz.

    Parameters
    -------
    octupoles: torch.Tensor
        Tensor of octupole moment values.

    Returns
    ----------
    octupoles: torch.Tensor
        Tensor of detraced octupole moment values.

    """

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
    """
    Function to detrace the atomic octupole moment labels from DGL graphs.

    Parameters
    -------
    graphs: dgl.DGLGraph
        Graphs containing the octupole moment labels to detrace.

    Returns
    ----------
    graphs: dgl.DGLGraph
        Graphs containing the detraced octupole moment labels.

    """

    octupoles = detrace_octupoles(graphs.ndata["label_octupoles"])  # type: ignore
    graphs.ndata["label_octupoles"] = octupoles

    return graphs


def get_loss_func_weight(data: torch.Tensor) -> float:
    """
    Function for obtaining the loss function weights for each atomic multipole moment, 
        based on the label distribution.

    Parameters
    -------
    data: torch.Tensor
        Atomic multipole moment labels.

    Returns
    ----------
    avg_iqr: float
        The interquartile range for the input label, averaged across the dimension of the label.
            (e.g., atomic dipole moment labels have dimension 3)

    """

    iqr = 0

    for i in range(len(data[0])):
        # Quartiles
        q3, q1 = np.percentile(data[:, i], [75, 25], interpolation="midpoint")  # type: ignore
        iqr += q3 - q1

    avg_iqr = iqr / len(data[0])
    return avg_iqr


def set_network_parameters(
    device: torch.device,
    traingraphs: list[dgl.DGLGraph],
    validationgraphs: list[dgl.DGLGraph],
    num_node_feats: int,
    num_edge_feats: int,
    model_type: str,
    model_precision: torch.dtype,
) -> tuple[
    PILNet,
    list[int, int, int, torch.nn.Module],
    dgl.dataloading.GraphDataLoader,
    dgl.dataloading.GraphDataLoader,
    torch.nn.modules.loss._Loss,
    list[torch.optim.Optimizer, torch.optim.Optimizer, torch.optim.Optimizer, torch.optim.Optimizer],
    list[float],
    list[torch.optim.lr_scheduler._LRScheduler, torch.optim.lr_scheduler._LRScheduler, torch.optim.lr_scheduler._LRScheduler, torch.optim.lr_scheduler._LRScheduler],
    list[float, float, float],
]:
    """
    Function for specifying parameters and hyperparameters to be used in the PIL-Net pipeline.

    Parameters
    -------
    device: torch.device
        Either a cpu or cuda device.
    traingraphs: list[dgl.DGLGraph]
        List of training set graphs.
    validationgraphs: list[dgl.DGLGraph]
        List of validation graphs.
    num_node_feats: int
        Dimension of node feature vector for each node.
    num_edge_feats: int
        Dimension of edge feature vector for each edge.
    model_type: str
        Whether the model is a "PINN" or Non-PINN" model.
    model_precision: torch.dtype
        Precision to apply to graph features and labels,
            as well as the model weights.

    Returns
    ----------
    model: PILNet
        Initialized PILNet model.
    model_parameters: list[int, int, int, torch.nn.Module]
        num_node_feats (as above)
        num_edge_feats (as above)
        hidden_dim:
            Number of hidden neurons in layers (width of neural network)
        activation:
            Non-linear activation function.
    train_dataloader: dgl.dataloading.GraphDataLoader
        Object for facilitating the loading of
            the training set data during model training.
    validation_dataloader: dgl.dataloading.GraphDataLoader
        Object for facilitating the loading of
            the validation set data during model training.
    loss_function: torch.nn.modules.loss._Loss
        Loss function used during model training.
    optimizer_list: list[torch.optim.Optimizer, torch.optim.Optimizer, torch.optim.Optimizer, torch.optim.Optimizer]
        Optimizers used during model training, one for each atomic multipole type.
    optimizer_parameters: list[float]
        Starting learning rate used by optmizer during model training.
    scheduler_list: list[torch.optim.lr_scheduler._LRScheduler, torch.optim.lr_scheduler._LRScheduler, torch.optim.lr_scheduler._LRScheduler, torch.optim.lr_scheduler._LRScheduler]
        Learning rate scheduler used during model training, one for each multipole type.
    scheduler_parameters: list[float, float, float]
        lr_factor: Factor by which learning rate is reduced during model training.
        lr_threshold: Threshold used when determining newest optimal learning rate during model training.
        lr_patience: Number of epochs to weight prior to reducing the learning rate further during model training.

    """

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
    model_parameters: list[int, int, int, torch.nn.Module],
    model_id: str,
    train_dataloader: dgl.dataloading.GraphDataLoader,
    validation_dataloader: dgl.dataloading.GraphDataLoader,
    maxtraintime: float,
    loss_function: torch.nn.modules.loss._Loss,
    optimizer_list: list[torch.optim.Optimizer, torch.optim.Optimizer, torch.optim.Optimizer, torch.optim.Optimizer],
    optimizer_parameters: list[float],
    scheduler_list: list[torch.optim.lr_scheduler._LRScheduler, torch.optim.lr_scheduler._LRScheduler, torch.optim.lr_scheduler._LRScheduler, torch.optim.lr_scheduler._LRScheduler],
    scheduler_parameters: list[float, float, float],
    bestmodel_path: str,
    monopole_weight: float,
    dipole_weight: float,
    quadrupole_weight: float,
    octupole_weight: float,
    model_type: str,
    model_precision: torch.dtype,
) -> None:
    """
    Function for train the neural network and saving the best model.

    Parameters
    -------
    device: torch.device
        Either a cpu or cuda device.
    model: PILNet
        Initialized PILNet model.
    model_parameters: list[int, int, int, torch.nn.Module]
        num_node_feats: int
            Dimension of node feature vector for each node.
        num_edge_feats: int
            Dimension of edge feature vector for each edge.
        hidden_dim:
            Number of hidden neurons in layers (width of neural network)
        activation:
            Non-linear activation function.   
    model_id: str
        Unique identifier assigned to model for the purpose of checkpointing, 
            associated with recent timestamp.
    train_dataloader: dgl.dataloading.GraphDataLoader
        Object for facilitating the loading of
            the training set data during model training.
    validation_dataloader: dgl.dataloading.GraphDataLoader
        Object for facilitating the loading of
            the validation set data during model training.
    maxtraintime: float
        Max time elapsed (seconds) to run model training.
            If time expires, training will end and the best model will be saved.
    loss_function: torch.nn.modules.loss._Loss
        Loss function used during model training.
    optimizer_list: list[torch.optim.Optimizer, torch.optim.Optimizer, torch.optim.Optimizer, torch.optim.Optimizer]
        Optimizers used during model training, one for each atomic multipole type.
    optimizer_parameters: list[float]
        Starting learning rate used by optmizer during model training.
    scheduler_list: list[torch.optim.lr_scheduler._LRScheduler, torch.optim.lr_scheduler._LRScheduler, torch.optim.lr_scheduler._LRScheduler, torch.optim.lr_scheduler._LRScheduler]
        Learning rate scheduler used during model training, one for each multipole type.
    scheduler_parameters list[float, float, float]
        lr_factor: Factor by which learning rate is reduced during model training.
        lr_threshold: Threshold used when determining newest optimal learning rate during model training.
        lr_patience: Number of epochs to weight prior to reducing the learning rate further during model training.
    bestmodel_path: str
        Path to save the best trained model.
    monopole_weight: float
        Weight associated with the atomic monopoles for the neural network loss function.
    dipole_weight: float
        Weight associated with the atomic dipoles for the neural network loss function.
    quadrupole_weight: float
         Weight associated with the atomic quadrupoles for the neural network loss function.
    octupole_weight: float
         Weight associated with the atomic octupoles for the neural network loss function.
    model_type: str
        Whether the model is a "PINN" or Non-PINN" model.
    model_precision: torch.dtype
        Precision to apply to graph features and labels.


    Returns
    ----------
    None

    """

    print("Performing model training...\n", flush=True)

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
            if model_type == "PINN":
                loss = (
                    ((1 / monopole_weight) * monopoles_loss)
                    + ((1 / dipole_weight) * dipoles_loss)
                    + ((1 / quadrupole_weight) * quadrupoles_loss)
                    + ((1 / octupole_weight) * octupoles_loss)
                )
            # Compute unscaled loss
            else:
                loss = (monopoles_loss + dipoles_loss + quadrupoles_loss + octupoles_loss)

            # NOTE: Only computing PINN penalties for illustrative purposes
            # (does not affect model training)
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
        ) = validation(model, validation_dataloader, loss_function)

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

        if model_type == "PINN":
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
    loss_function: torch.nn.modules.loss._Loss,
) -> tuple[float, float, float, float]:
    """
    Function for evaluating the model on the validation set.

    Parameters
    -------
    model: PILNet
        PILNet model (partially trained).
    validation_data_loader: dgl.dataloading.GraphDataLoader
        Object for facilitating the loading of
            the validation set data during model training.
    loss_function: torch.nn.modules.loss._Loss
        Loss function used during model training.

    Returns
    ----------
    epoch_monopoles_loss: float
        Atomic monopole loss computed on the validation set.
    epoch_dipoles_loss: float
        Atomic dipole loss computed on the validation set.
    epoch_quadrupoles_loss: float
        Atomic quadrupole loss computed on the validation set.
    epoch_octupoles_loss: float
        Atomic octupole loss computed on the validation set.

    """

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
        monopoles_loss = loss_function(valbgs.ndata["label_monopoles"], predlabels[:, 0:1])
        dipoles_loss = loss_function(valbgs.ndata["label_dipoles"], predlabels[:, 1:4])
        quadrupoles_loss = loss_function(
            valbgs.ndata["label_quadrupoles"], predlabels[:, 4:10]
        )
        octupoles_loss = loss_function(
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
    """
    Function to obtain PINN penalty error, for illustrative purposes (does not impact network training)."

    Parameters
    -------
    bgs: dgl.DGLGraph
        Grouping of batched graphs.
    predlabels: torch.Tensor
        Labels predicted by model.
    random_graph_index: int
        Random integer corresponding to a training set graph.
    model_precision: torch.dtype
        Precision of the model weights.
    device: torch.device
        Either a cpu or cuda device.

    Returns
    ----------
    monopole_penalty_total: torch.Tensor[float]
        Error in the atomic monopole moment predictions 
            with respect to the monopole total charge penalty.
    monopole_penalty_H: torch.Tensor[float]
        Error in the atomic monopole moment predictions
            with respect to the monopole hydrogen penalty.
    quadrupole_penalty: torch.Tensor[float]
        Error in the atomic quadrupole moment predictions
            with respect to the quadrupole trace penalty.
    octupole_penalty: torch.Tensor[float]
        Error in the atomic octupole moment predictions
            with respect to the quadrupole trace penalty.

    """

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
        octupole_penalty += abs(pred_oct[0] + pred_oct[3] + pred_oct[5])
        octupole_penalty += abs(pred_oct[6] + pred_oct[1] + pred_oct[8])
        octupole_penalty += abs(pred_oct[9] + pred_oct[2] + pred_oct[7])

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
    """
    Main function for training a PILNet model.

    Parameters
    ----------
    read_filepath: str
        Path to training and validation set files.
    save_filepath: str
        Path to save trained model(s).

    Returns
    -------
    None

    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_path = read_filepath + "traindata.bin"
    validation_path = read_filepath + "validationdata.bin"

    # Save the best model to specified folder, including current timestamp
    model_id = "pilnet_model_" + str(datetime.datetime.now().timestamp()).replace(" ", "_")
    bestmodel_path = (
        save_filepath
        + model_id
        + ".bin"
    )
    print(f"Best model path: {bestmodel_path}")

    # Details related to our QMDFAM graphs (desired number of features to use)
    num_node_feats = 16
    num_edge_feats = 26

    # Details related to PIL-Net experimental set-up
    maxtraintime = 24  # in hours
    maxtraintime *= 3600  # in seconds

    model_type = "PINN"
    # model_type = "Non-PINN"
    print(f"Model type: {model_type}")

    model_precision = torch.float32

    # Load datasets
    (
        traingraphs,
        validationgraphs,
        monopole_weight,
        dipole_weight,
        quadrupole_weight,
        octupole_weight,
    ) = load_format_datasets(device, train_path, validation_path, model_type, model_precision)

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
        model_id,
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
