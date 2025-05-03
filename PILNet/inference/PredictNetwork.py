"""Use the trained PILNet model(s) to predict the test set labels.
"""

import numpy as np
import torch

import dgl
from dgl.data.utils import load_graphs

import os
import math
import copy

import h5py
import hashlib
from typing import Union
from scipy.spatial.distance import cdist


from ..training.TrainNetwork import (
    detrace_atomic_quadrupole_labels,
    detrace_atomic_octupole_labels,
    detrace_quadrupoles,
    detrace_octupoles,
)
from ..model.PILNet import PILNet

def load_format_dataset(
    device: torch.device, test_path: str, model_precision: torch.dtype
) -> list[dgl.DGLGraph]:
    """
    Function to load, change precision, and detrace the graphs.

    Parameters
    -------
    device: torch.device
        Either a cpu or cuda device.
    test_path: str
        Path to test set graphs.
    model_precision: torch.dtype
        Precision to apply to graph features and labels.

    Returns
    ----------
    testgraphs: list[dgl.DGLGraph]
        List of test set graphs.

    """

    testgraphs = load_test_dataset(test_path)

    print("Formatting dataset...", flush=True)

    batch_testgraphs = dgl.batch(testgraphs)

    # Convert graph data/labels to have specified precision
    batch_testgraphs.ndata["nfeats"] = batch_testgraphs.ndata["nfeats"].to(  # type: ignore
        model_precision
    )
    batch_testgraphs.edata["efeats"] = batch_testgraphs.edata["efeats"].to(  # type: ignore
        model_precision
    )

    batch_testgraphs.ndata["label_monopoles"] = batch_testgraphs.ndata[
        "label_monopoles"
    ].to(model_precision)  # type: ignore
    batch_testgraphs.ndata["label_dipoles"] = batch_testgraphs.ndata[
        "label_dipoles"
    ].to(model_precision)  # type: ignore
    batch_testgraphs.ndata["label_quadrupoles"] = batch_testgraphs.ndata[
        "label_quadrupoles"
    ].to(model_precision)  # type: ignore
    batch_testgraphs.ndata["label_octupoles"] = batch_testgraphs.ndata[
        "label_octupoles"
    ].to(model_precision)  # type: ignore

    batch_testgraphs.ndata["coordinates"] = batch_testgraphs.ndata[
        "coordinates"
    ].to(  # type: ignore
        model_precision
    )

    # Detrace the atomic quadrupole and octupole labels
    batch_testgraphs = detrace_atomic_quadrupole_labels(batch_testgraphs)
    batch_testgraphs = detrace_atomic_octupole_labels(batch_testgraphs)

    batch_testgraphs.ndata["molecular_dipole"] = batch_testgraphs.ndata[
        "molecular_dipole"
    ].to(model_precision)  # type: ignore

    batch_testgraphs.ndata["molecular_quadrupole"] = batch_testgraphs.ndata[
        "molecular_quadrupole"
    ].to(model_precision)  # type: ignore

    batch_testgraphs.ndata["molecular_octupole"] = batch_testgraphs.ndata[
        "molecular_octupole"
    ].to(model_precision)  # type: ignore

    batch_testgraphs = batch_testgraphs.to(device)
    testgraphs = dgl.unbatch(batch_testgraphs)

    return testgraphs


def load_test_dataset(test_path: str) -> list[dgl.DGLGraph]:
    """
    Function to load test set graphs from file.

    Parameters
    -------
    test_path: str
        Filepath to test set graphs.

    Returns
    ----------
    testgraphs: list[dgl.DGLGraph]
        List of test set graphs.

    """

    print("\nReading in dataset...", flush=True)

    testgraphs = load_graphs(test_path)
    testgraphs = testgraphs[0]
    return testgraphs


def compute_MAE(true: np.ndarray, pred: np.ndarray) -> np.ndarray:
    """
    Function to compute the mean absolute error (MAE)
        between the reference data and the model predictions.

    Parameters
    -------
    true: np.ndarray
        Reference multipole moment data.
    pred: np.ndarray
        Multipole moment model predictions.

    Returns
    ----------
    np.ndarray: Computed mean absolute error.

    """

    return np.mean((abs(true - pred)))  # type: ignore


def compute_R2(true: np.ndarray, pred: np.ndarray) -> np.ndarray:
    """
    Function to compute the coefficient of determination (R^2)
        between the reference data and the model predictions.

    Parameters
    -------
    true: np.ndarray
        Reference multipole moment data.
    pred: np.ndarray
        Multipole moment model predictions.

    Returns
    ----------
    np.ndarray: Computed coefficient of determination.

    """

    RSS = np.sum((true - pred) ** 2)
    mean_true = np.mean(true, axis=0)
    TSS = np.sum((true - mean_true) ** 2)

    return 1 - (RSS / TSS)


def compute_RMSD(true: np.ndarray, pred: np.ndarray) -> np.ndarray:
    """
    Function to compute the root mean squared deviation (RMSD)
        between the reference data and the model predictions.

    Parameters
    -------
    true: np.ndarray
        Reference multipole moment data.
    pred: np.ndarray
        Multipole moment model predictions.

    Returns
    ----------
    np.ndarray: Computed root mean squared deviation.

    """

    return np.sqrt(np.mean((true - pred) ** 2))


def print_testing_statistics(
    true: np.ndarray, pred: np.ndarray
) -> tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray
]:
    """
    Function to statistical information about the true and predicted labels.

    Parameters
    -------
    true: np.ndarray
        Reference multipole moment data.
    pred: np.ndarray
        Multipole moment model predictions.

    Returns
    ----------
    MAE: np.ndarray
        Computed mean absolute error between the reference data and the predicted values.
    R2: np.ndarray
        Computed coefficient of determination between the reference data and the predicted values.
    RMSD: np.ndarray
        Computed root mean squared deviation between the reference data and the predicted values.
    mean_true: np.ndarray
        Mean reference value, across each column dimension.
        E.g., Atomic dipole moments have dimension 3, 
            so this computed value will have three entries.
    mean_pred: np.ndarray
        Mean predicted value, across each column dimension.
        E.g., Atomic dipole moments have dimension 3, 
            so this computed value will have three entries.
    stdev_true: np.ndarray
        E.g., Atomic dipole moments have dimension 3, 
            so this computed value will have three entries.
    stdev_pred: np.ndarray
        E.g., Atomic dipole moments have dimension 3, 
            so this computed value will have three entries.

    """

    MAE = compute_MAE(true, pred)
    R2 = compute_R2(true, pred)
    RMSD = compute_RMSD(true, pred)

    mean_true = np.mean(pred, axis=0)
    mean_pred = np.mean(true, axis=0)
    stdev_true = np.std(pred, axis=0)
    stdev_pred = np.std(true, axis=0)

    return MAE, R2, RMSD, mean_true, mean_pred, stdev_true, stdev_pred


def conv_one_hot_to_bytes(onehot: torch.Tensor, element_types: list) -> bytes:
    """
    Function to undo a one-hot encoding 
        (convert one-hot encoding to element type).

    Parameters
    -------
    onehot: torch.Tensor
        Tensor of the one-hot encoding of an element.
    element_types: list
        Exhaustive list of elements present in the dataset.

    Returns
    ----------
    str: Atomic symbol corresponding to the input one-hot encoding.

    Raises
    ------
    ValueError:
        If the input one-hot encoding is unexpected.


    """
    
    for i in range(len(onehot)):
        if onehot[i] == 1:
            return element_types[i]

    raise ValueError(
        f"The one-hot pattern {onehot} does not exist"
        f" within the pre-defined element types list: {element_types}"
    )


def element_specific_statistics(
    true_labels: np.ndarray,
    pred_labels: np.ndarray,
    nfeats: np.ndarray,
    element_types: list,
    target_elem: bytes,
) -> tuple[Union[np.ndarray, float], Union[np.ndarray, float]]:
    """
    Function to obtain the element-specific MAE and R2 between reference and predicted labels.

    Parameters
    -------
    true_labels: np.ndarray
        Element-specific reference multipole moment data.
    pred_labels: np.ndarray
        Element-specific multipole moment model predictions.
    nfeats: np.ndarray
        Node features corresponding to a batch of DGL graphs.
    element_types: list
        Exhaustive list of elements present in the dataset.
    target_elem: bytes
        Target dataset element.

    Returns
    ----------
    elem_MAE: np.ndarray | float
        Element-specific mean absolute error, 
            or "nan" if the target element does not exist within the dataset.
    elem_R2: np.ndarray | float
        Element-specific coefficient of determination, 
            or "nan" if the target element does not exist within the dataset.

    """

    elem_true_labels = np.zeros(0)
    elem_pred_labels = np.zeros(0)

    for i in range(len(nfeats)):

        # If this feature corresponds to the target element:
        if conv_one_hot_to_bytes(nfeats[i], element_types) == target_elem:

            if len(elem_true_labels) == 0:
                elem_true_labels = true_labels[i]
                elem_pred_labels = pred_labels[i]
            else:
                elem_true_labels = np.vstack((elem_true_labels, true_labels[i]))
                elem_pred_labels = np.vstack((elem_pred_labels, pred_labels[i]))

    if len(elem_true_labels) > 0:
        elem_MAE, elem_R2, _, _, _, _, _ = print_testing_statistics(
            elem_true_labels, elem_pred_labels
        )

    else:  # Element does not exist within dataset
        elem_MAE = float("nan")
        elem_R2 = float("nan")

    return elem_MAE, elem_R2


def get_reference_molecular_moments(
    testbgs: dgl.DGLGraph, label_name: str
) -> torch.Tensor:
    """
    Function to extract reference molecular multipole moments from
        information stored in the graphs.

    Parameters
    -------
    testbgs: dgl.DGLGraph
        Batch of test set graphs.
    label_name: str
        Name of target molecular multipole moments.

    Returns
    ----------
    true_multipole_value: torch.Tensor
        Tensor of reference molecular multipole moment values corresponding 
            to the batched set of graphs.

    """

    batch_num_nodes = testbgs.batch_num_nodes()
    sum_batch_num_nodes = copy.deepcopy(batch_num_nodes)

    running_sum = 0
    for j in range(len(sum_batch_num_nodes)):
        sum_batch_num_nodes[j] = running_sum
        running_sum += batch_num_nodes[j]

    true_multipole_value = testbgs.ndata[label_name][sum_batch_num_nodes, :]

    return true_multipole_value


def sum_approximation_contributions(
    testbgs: dgl.DGLGraph, atomic_multipole_contribution: torch.Tensor
) -> torch.Tensor:
    """
    Function that serves as helper function for approximating
        molecular multipole moments through atomic multipole moment predictions.

    Parameters
    -------
    testbgs: dgl.DGLGraph
        Batch of test set graphs.
    atomic_multipole_contribution: torch.Tensor
        Predicted atomic multipole contributions
            to the molecular multipole moment approximation.

    Returns
    ----------
    pred_molecular_moment: torch.Tensor
        Approximated molecular multipole moments.

    """

    testbgs.ndata["approx_molecular_moment"] = atomic_multipole_contribution

    # Sum over all the atoms' multipole moment contributions
    # to obtain a single molecular moment vector per molecule
    pred_molecular_moment = dgl.readout_nodes(
        graph=testbgs, feat="approx_molecular_moment", op="sum"
    )

    return pred_molecular_moment


def approximate_molecular_dipole_moment(
    testbgs: dgl.DGLGraph,
    atomic_monopoles: torch.Tensor,
    atomic_dipoles: torch.Tensor,
    coordinates: torch.Tensor,
) -> torch.Tensor:
    """
    Function for approximating the molecular dipole moment of a group of molecules
        as a function of its corresponding atomic monopole moments, atomic dipole moments, and atomic coordinates.

    Parameters
    -------
    testbgs: dgl.DGLGraph
        Batch of test set graphs.
    atomic_monopoles: torch.Tensor
        Predicted atomic monopole moments.
    atomic_dipoles: torch.Tensor
        Predicted atomic dipole moments.
    coordinates: torch.Tensor
        Corresponding grouping of Cartesian coordinates.

    Returns
    ----------
    molecular_dipole_moment: torch.Tensor
        Approximated molecular dipole moments.

    """

    # Multiply atomic multipoles by atomic Cartesian coordinates
    monopole_contribution = (
        torch.reshape(atomic_monopoles, (-1, 1)) * coordinates
    )

    dipole_contribution = atomic_dipoles + monopole_contribution

    molecular_dipole_moment = sum_approximation_contributions(
        testbgs, dipole_contribution
    )
    return molecular_dipole_moment


def approximate_molecular_quadrupole_moment(
    testbgs: dgl.DGLGraph,
    atomic_monopoles: torch.Tensor,
    coordinates: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    """
    Function for approximating the molecular quadrupole moments of a group of molecules
        as a function of its corresponding atomic monopole moments and atomic coordinates.

    Parameters
    -------
    testbgs: dgl.DGLGraph
        Batch of test set graphs.
    atomic_monopoles: torch.Tensor
        Predicted atomic monopole moments.
    coordinates: torch.Tensor
        Corresponding grouping of Cartesian coordinates.
    device: torch.device
        Either a cpu or cuda device.

    Returns
    ----------
    molecular_quadrupole_moment: torch.Tensor
        Approximated molecular quadrupole moments.

    """

    # Final quadrupole vector is represented as xx, xy, xz, yy, yz, zz.
    monopole_contribution = torch.zeros((len(atomic_monopoles), 6)).to(device)

    q = torch.squeeze(atomic_monopoles)

    rx = coordinates[:, 0]
    ry = coordinates[:, 1]
    rz = coordinates[:, 2]

    dotprod = torch.einsum("ij,ij->i", coordinates, coordinates)

    monopole_contribution[:, 0] = q * ((1.5 * rx * rx) - (0.5 * dotprod))
    monopole_contribution[:, 1] = q * (1.5 * rx * ry)
    monopole_contribution[:, 2] = q * (1.5 * rx * rz)
    monopole_contribution[:, 3] = q * ((1.5 * ry * ry) - (0.5 * dotprod))
    monopole_contribution[:, 4] = q * (1.5 * ry * rz)
    monopole_contribution[:, 5] = q * ((1.5 * rz * rz) - (0.5 * dotprod))

    molecular_quadrupole_moment = sum_approximation_contributions(
        testbgs, monopole_contribution
    )

    return molecular_quadrupole_moment


def approximate_molecular_octupole_moment(
    testbgs: dgl.DGLGraph,
    atomic_monopoles: torch.Tensor,
    coordinates: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    """
    Function for approximating the molecular octupole moments of a group of molecules
        as a function of its corresponding atomic monopole moments and atomic coordinates.

    Parameters
    -------
    testbgs: dgl.DGLGraph
        Batch of test set graphs.
    atomic_monopoles: torch.Tensor
        Predicted atomic monopole moments.
    coordinates: torch.Tensor
        Corresponding grouping of Cartesian coordinates.
    device: torch.device
        Either a cpu or cuda device.

    Returns
    ----------
    molecular_octupole_moment: torch.Tensor
        Approximated molecular octupole moments.

    """

    # Final octupole vector is represented as xxx, xxy, xxz, xyy, xyz, xzz, yyy, yyz, yzz, zzz.
    monopole_contribution = torch.zeros((len(atomic_monopoles), 10)).to(device)

    q = torch.squeeze(atomic_monopoles)

    rx = coordinates[:, 0]
    ry = coordinates[:, 1]
    rz = coordinates[:, 2]

    dotprod = torch.einsum("ij,ij->i", coordinates, coordinates)

    monopole_contribution[:, 0] = q * (
        (2.5 * rx * rx * rx) - (0.5 * dotprod * (3 * rx))
    )
    monopole_contribution[:, 1] = q * ((2.5 * rx * rx * ry) - (0.5 * dotprod * ry))
    monopole_contribution[:, 2] = q * ((2.5 * rx * rx * rz) - (0.5 * dotprod * rz))
    monopole_contribution[:, 3] = q * ((2.5 * rx * ry * ry) - (0.5 * dotprod * rx))
    monopole_contribution[:, 4] = q * (2.5 * rx * ry * rz)
    monopole_contribution[:, 5] = q * ((2.5 * rx * rz * rz) - (0.5 * dotprod * rx))
    monopole_contribution[:, 6] = q * (
        (2.5 * ry * ry * ry) - (0.5 * dotprod * (3 * ry))
    )
    monopole_contribution[:, 7] = q * ((2.5 * ry * ry * rz) - (0.5 * dotprod * rz))
    monopole_contribution[:, 8] = q * ((2.5 * ry * rz * rz) - (0.5 * dotprod * ry))
    monopole_contribution[:, 9] = q * (
        (2.5 * rz * rz * rz) - (0.5 * dotprod * (3 * rz))
    )

    molecular_octupole_moment = sum_approximation_contributions(
        testbgs, monopole_contribution
    )

    return molecular_octupole_moment


def test_network(
    device: torch.device,
    bestmodel_paths: list[str],
    testgraphs: list[dgl.DGLGraph],
    esp_data: h5py.File,
    test_bsz: int,
    model_type: str,
    multipole_names: list[str],
    ESP_multipole_types: list[str],
    element_types: list[bytes],
    computed_multipole_indices: list[int]
) -> None:
    """
    Function for making predictions on test set using trained model(s).

    NOTE: This code assumes the feature and label orders determined in ExtractFeatures.py
        e.g., The one-hot encoded elment-type feature is accessible as:
            testbgs.ndata['nfeats'][:,0:7]
        e.g., The model returns label predictions in the order:
            atomic monopole, dipole, quadrupole, and then octupole

    Parameters
    -------
    device: torch.device
        Either a cpu or cuda device.
    bestmodel_paths: list[str]
        Path where best trained models are stored.
    testgraphs: list
        List of test set graphs.
    esp_data: h5py.File
        QMDFAM dataset file containing electrostatic potential (ESP) data.
    test_bsz: int
        Batch size to use when processing data when performing model inference.
    model_type: str
        Whether the model is a "PINN" or Non-PINN" model.
    multipole_names: list[str]
        List of names of multipole moments on which to perform evaluation
    ESP_multipole_types: list[str]
        List of names of atomic multipole moments to use in ESP reconstruction.
    element_types: list[bytes]
        Exhaustive list of elements present in the dataset.
    computed_multipole_indices: list[int]
        Integers corresponding to the indices of the test set graphs for which
            the reference molecular quadrupole and octupole moments were computed using the PSI4 library.

    Returns
    ----------
    None

    """

    print("Performing model inference...", flush=True)

    # Create lists to store MAE and R^2 computed information
    num_graphs = len(testgraphs)
    num_models = len(bestmodel_paths)
    num_multipoles = len(multipole_names)
    num_esp_multipole_types = len(ESP_multipole_types)
    num_atomic_multipoles = 4
    num_element_types = len(element_types)

    overall_MAE = np.zeros((num_models, num_multipoles))
    overall_R2 = np.zeros((num_models, num_multipoles))

    MAE_esp_reconstruction = np.zeros((num_models, num_esp_multipole_types))
    R2_esp_reconstruction = np.zeros((num_models, num_esp_multipole_types))

    if model_type == "PINN":
        # Do not include molecular moments here since they are a graph-level property
        elem_MAE = np.zeros((num_models, num_atomic_multipoles, num_element_types))
        elem_R2 = np.zeros((num_models, num_atomic_multipoles, num_element_types))
    else:
        elem_MAE = np.zeros((num_models, num_multipoles, num_element_types))
        elem_R2 = np.zeros((num_models, num_multipoles, num_element_types))

    # Iterate over each trained model
    print(
        "\n** Predictive errors averaged across {} model(s): **".format(num_models),
        flush=True,
    )

    for i in range(num_models):

        print("\nModel path: {}".format(bestmodel_paths[i]), flush=True)

        checkpoint = torch.load(bestmodel_paths[i])
        model_param = checkpoint["model_parameters"]
        model = PILNet(model_param[0], model_param[1], model_param[2], model_type).to(
            device
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()

        predlabels = torch.zeros(0).to(device)
        predlabels_mol_dipole = torch.zeros(0).to(device)
        num_batches = math.ceil(num_graphs / test_bsz)

        # Iterate over each batch of graphs
        start = 0
        end = test_bsz
        for j in range(num_batches):

            with torch.no_grad():

                # Obtain prediction labels
                testbgs = dgl.batch(testgraphs[start:end])
                preds = model(testbgs)

                # If model is of type PINN, approximate the molecular dipole moment
                if model_type == "PINN":

                    pred_monopoles = preds[:, 0:1]
                    pred_dipoles = preds[:, 1:4]
                    coordinates = testbgs.ndata["coordinates"]

                    # Approximate the molecular dipole moment
                    preds_mol_dipole = approximate_molecular_dipole_moment(
                        copy.deepcopy(testbgs),
                        pred_monopoles,
                        pred_dipoles,
                        coordinates,  # type: ignore
                    )

                    predlabels_mol_dipole = torch.cat(
                        (predlabels_mol_dipole, preds_mol_dipole)
                    )

                predlabels = torch.cat((predlabels, preds))

            start = end
            end += test_bsz

        # Convert format of dataset reference labels for atomic dipoles
        testbgs = dgl.batch(testgraphs)

        true_monopoles = testbgs.ndata["label_monopoles"].cpu().numpy()  # type: ignore
        true_dipoles = testbgs.ndata["label_dipoles"].cpu().numpy()  # type: ignore
        true_quadrupoles = testbgs.ndata["label_quadrupoles"].cpu().numpy()  # type: ignore
        true_octupoles = testbgs.ndata["label_octupoles"].cpu().numpy()  # type: ignore

        pred_monopoles = predlabels[:, 0:1]
        pred_dipoles = predlabels[:, 1:4]
        pred_quadrupoles = predlabels[:, 4:10]
        pred_octupoles = predlabels[:, 10:20]

        coordinates = testbgs.ndata["coordinates"]

        batch_num_nodes = testbgs.batch_num_nodes()
        sum_batch_num_nodes = copy.deepcopy(batch_num_nodes)

        running_sum = 0
        for j in range(len(sum_batch_num_nodes)):
            sum_batch_num_nodes[j] = running_sum
            running_sum += batch_num_nodes[j]

        identifiers = list(testbgs.ndata["unique_identifier_hashed"][sum_batch_num_nodes])
        identifiers = list([x.item() for x in identifiers])


        # If model is of type PINN, obtain molecular dipole, quadrupole,
        # and octupole moments for each molecule
        if model_type == "PINN":

            testbgs.ndata["pred_monopoles"] = pred_monopoles
            testbgs.ndata["pred_dipoles"] = pred_dipoles
            testbgs.ndata["pred_quadrupoles"] = pred_quadrupoles
            testbgs.ndata["pred_octupoles"] = pred_octupoles

            testgraphs_selected = dgl.unbatch(copy.deepcopy(testbgs))

            # Get values specific to the graphs randomly selected
            # for reference multipole calculation
            testbgs_selected = dgl.batch(
                [testgraphs_selected[i] for i in computed_multipole_indices]
            )

            coordinates_selected = testbgs_selected.ndata[
                "coordinates"
            ]
            pred_monopoles_selected = testbgs_selected.ndata["pred_monopoles"]

            # Approximate the molecular quadrupole moment
            predlabels_mol_quadrupole = approximate_molecular_quadrupole_moment(
                copy.deepcopy(testbgs_selected),
                pred_monopoles_selected,  # type: ignore
                coordinates_selected,  # type: ignore
                device,
            )

            # Approximate the molecular octupole moment
            predlabels_mol_octupole = approximate_molecular_octupole_moment(
                copy.deepcopy(testbgs_selected),
                pred_monopoles_selected,  # type: ignore
                coordinates_selected,  # type: ignore
                device,
            )

            predlabels_mol_dipole = predlabels_mol_dipole.cpu().numpy()
            predlabels_mol_quadrupole = predlabels_mol_quadrupole.cpu().numpy()
            predlabels_mol_octupole = predlabels_mol_octupole.cpu().numpy()

            true_mol_dipole = get_reference_molecular_moments(
                testbgs, "molecular_dipole"
            )

            true_mol_quadrupole = get_reference_molecular_moments(
                testbgs_selected, "molecular_quadrupole"
            )

            true_mol_octupole = get_reference_molecular_moments(
                testbgs_selected, "molecular_octupole"
            )

            true_mol_quadrupole = detrace_quadrupoles(true_mol_quadrupole)
            true_mol_octupole = detrace_octupoles(true_mol_octupole)

            true_mol_dipole = true_mol_dipole.cpu().numpy()
            true_mol_quadrupole = true_mol_quadrupole.cpu().numpy()
            true_mol_octupole = true_mol_octupole.cpu().numpy()

        # Organize reference and predicted labels in lists
        true_labels_by_multipole = [
            true_monopoles,
            true_dipoles,
            true_quadrupoles,
            true_octupoles,
        ]

        pred_monopoles = pred_monopoles.cpu().numpy()
        pred_dipoles = pred_dipoles.cpu().numpy()
        pred_quadrupoles = pred_quadrupoles.cpu().numpy()
        pred_octupoles = pred_octupoles.cpu().numpy()
        predlabels = predlabels.cpu().numpy()

        pred_labels_by_multipole = [
            pred_monopoles,
            pred_dipoles,
            pred_quadrupoles,
            pred_octupoles,
        ]

        if model_type == "PINN":
            true_labels_by_multipole.extend(
                [true_mol_dipole, true_mol_quadrupole, true_mol_octupole]
            )
            pred_labels_by_multipole.extend(
                [
                    predlabels_mol_dipole,
                    predlabels_mol_quadrupole,
                    predlabels_mol_octupole,
                ]
            )  # type: ignore

        # Record MAE and R^2 for each multipole type
        for j in range(len(multipole_names)):
            MAE, R2, _, _, _, _, _ = print_testing_statistics(
                true_labels_by_multipole[j], pred_labels_by_multipole[j]
            )
            overall_MAE[i, j] = MAE
            overall_R2[i, j] = R2

        # Record element-specific MAE and R^2 for each multipole type
        nfeats = testbgs.ndata["nfeats"][:, 0:7].cpu().numpy()
        for j in range(num_atomic_multipoles):
            for k in range(len(element_types)):
                elem_MAE[i, j, k], elem_R2[i, j, k] = element_specific_statistics(
                    true_labels_by_multipole[j],
                    pred_labels_by_multipole[j],
                    nfeats,
                    element_types,
                    element_types[k],
                )

        # ESP reconstruction
        MAE_kcal_mon, R2_mon, MAE_kcal_dip, R2_dip, MAE_kcal_quad, R2_quad, MAE_kcal_oct, R2_oct = reconstruct_esp(esp_data, testbgs.batch_num_nodes(), coordinates, identifiers, true_labels_by_multipole, pred_labels_by_multipole, device)
        
        MAE_esp_reconstruction[i, 0] += MAE_kcal_mon
        MAE_esp_reconstruction[i, 1] += MAE_kcal_dip
        MAE_esp_reconstruction[i, 2] += MAE_kcal_quad
        MAE_esp_reconstruction[i, 3] += MAE_kcal_oct
       
        R2_esp_reconstruction[i, 0] += R2_mon
        R2_esp_reconstruction[i, 1] += R2_dip
        R2_esp_reconstruction[i, 2] += R2_quad
        R2_esp_reconstruction[i, 3] += R2_oct


    print("\n\nOVERALL RESULTS", flush=True)
    for j in range(len(multipole_names)):
        print("\nMultipole type: {}".format(multipole_names[j]), flush=True)
        print("MAE: {}".format(np.mean(overall_MAE[:, j])), flush=True)
        print("R^2: {}".format(np.mean(overall_R2[:, j])), flush=True)

        print(
            "Stdev of MAE across all models: {}".format(np.std(overall_MAE[:, j])),
            flush=True,
        )
        print(
            "Stdev of R^2 across all models: {}".format(np.std(overall_R2[:, j])),
            flush=True,
        )

    print("\nELEMENT-SPECIFIC RESULTS")
    for j in range(num_atomic_multipoles):
        print("\nMultipole type: {}".format(multipole_names[j]))
        for k in range(len(element_types)):
            print("Element: {}".format(element_types[k]))
            print("MAE: {}".format(np.mean(elem_MAE[:, j, k])), flush=True)
            print("R^2: {}".format(np.mean(elem_R2[:, j, k])), flush=True)
            print(
                "Stdev of MAE across all models: {}".format(np.std(elem_MAE[:, j, k])),
                flush=True,
            )
            print(
                "Stdev of R^2 across all models: {}".format(np.std(elem_R2[:, j, k])),
                flush=True,
            )

    print("\n\nESP RECONSTRUCTION RESULTS", flush=True)
    for j in range(num_esp_multipole_types):
        print("\nMultipole type: {}".format(ESP_multipole_types[j]), flush=True)
        print("MAE: {} kcal/mol".format(np.mean(MAE_esp_reconstruction[:, j])), flush=True)
        print("R^2: {}".format(np.mean(R2_esp_reconstruction[:, j])), flush=True)

        print(
            "Stdev of MAE across all models: {}".format(np.std(MAE_esp_reconstruction[:, j])),
            flush=True,
        )
        print(
            "Stdev of R^2 across all models: {}".format(np.std(R2_esp_reconstruction[:, j])),
            flush=True,
        )


def compute_ESP_up_to_monopoles(
    vdw_surfaces: list,
    batch_num_nodes: torch.Tensor,
    coordinates_multipoles: torch.Tensor,
    pred_monopoles: np.ndarray
) -> np.ndarray:
    """
    Function for approximating the electrostatic potential (ESP)
        for a batch of molecules, using atomic monopole model predictions.

    Parameters
    -------
    vdw_surfaces: list
        Cartesian coordinates of the van der Waals surfaces on which the ESP was computed 
        (data originating from QMDFAM dataset).
    batch_num_nodes: torch.Tensor
        Ordered list of the number of atoms/nodes belonging to each molecule/graph in the batch of graphs.
    coordinates_multipoles: torch.Tensor
        Cartesian coordinates associated with the batch of molecules/graphs.
    pred_monopoles: np.ndarray
        Atomic monopole moment model predictions.

    Returns
    ----------
    ESP_up_to_monopoles: np.ndarray
        Electrostatic potential approximation contribution using atomic multipole predictions,
            up to atomic monopoles.

    """

    ESP_up_to_monopoles = []

    i_start = 0


    for mol_id, surface in enumerate(vdw_surfaces):  # Iterate over molecules
        
        if mol_id%1000 == 0:
            print("Index: ", mol_id, flush=True)

        i_end = i_start + batch_num_nodes[mol_id]

        coords = coordinates_multipoles[i_start:i_end] # [num_atoms, 3]
        pred_mons = pred_monopoles[i_start:i_end] # [num_atoms, 1]

        # Compute pairwise distances: [num_atoms, num_points]
        distances = cdist(coords, surface)  

        # Compute ESP using atomic monopoles: [num_points]
        monopoles_esp = np.sum(pred_mons / distances, axis=0) * 1389.35

        ESP_up_to_monopoles.append(monopoles_esp)

        i_start = i_end


    ESP_up_to_monopoles = np.concatenate(ESP_up_to_monopoles)

    return ESP_up_to_monopoles


def compute_ESP_up_to_dipoles(
    vdw_surfaces: list, 
    batch_num_nodes: torch.Tensor, 
    coordinates_multipoles: torch.Tensor, 
    pred_dipoles: np.ndarray
) -> np.ndarray:
    """
    Function for approximating the electrostatic potential (ESP)
        for a batch of molecules, using atomic multipole model predictions up to dipoles.

    Parameters
    -------
    vdw_surfaces: list
        Cartesian coordinates of the van der Waals surfaces on which the ESP was computed 
        (data originating from QMDFAM dataset).
    batch_num_nodes: torch.Tensor
        Ordered list of the number of atoms/nodes belonging to each molecule/graph in the batch of graphs.
    coordinates_multipoles: torch.Tensor
        Cartesian coordinates associated with the batch of molecules/graphs.
    pred_dipoles: np.ndarray
        Atomic dipole moment model predictions.

    Returns
    ----------
    ESP_up_to_dipoles: np.ndarray
        Electrostatic potential approximation contribution using atomic multipole predictions,
            up to atomic dipoles.

    """

    i_start = 0
    ESP_up_to_dipoles = []

    for mol_id, surface in enumerate(vdw_surfaces):  # Iterate over molecules
        
        if mol_id%1000 == 0:
            print("Index: ", mol_id, flush=True)

        i_end = i_start + batch_num_nodes[mol_id]

        coords = coordinates_multipoles[i_start:i_end]
        pred_dips = pred_dipoles[i_start:i_end]

        # Compute pairwise distances: shape [num_atoms, num_points]
        distances = cdist(coords, surface)  

        coords = np.array(coords)
        surface = np.array(surface)

        # surface is [num_points, 3] -> [1, num_points, 3]
        # coords is [num_atoms, 3] -> [num_atoms, 1, 3]
        # Rij is [num_atoms, num_points, 3]
            # for each atom, we compute the distance between it and each point 
        Rij = surface[None, :, :] - coords[:, None, :] 

        # pred_dips is [num_atoms, 3]
        # Einsum is [num_atoms, num_points, 3] by [num_atoms, 3] -> [num_atoms, num_points]
        # Divide by distances: [num_atoms, num_points]
        # Sum over the columns to get [num_points]
        dipoles_esp = np.sum(np.einsum("ijk,ik->ij", Rij, pred_dips) / distances**3, axis=0) * 1389.35

        ESP_up_to_dipoles.append(dipoles_esp)

        i_start = i_end


    ESP_up_to_dipoles = np.concatenate(ESP_up_to_dipoles)

    return ESP_up_to_dipoles



def compute_ESP_up_to_quadrupoles(
    vdw_surfaces: list, 
    batch_num_nodes: torch.Tensor,
    coordinates_multipoles: torch.Tensor,
    pred_quadrupoles: np.ndarray 
) -> np.ndarray:
    """
    Function for approximating the electrostatic potential (ESP)
        for a batch of molecules, using atomic multipole model predictions up to quadrupoles.

    Parameters
    -------
    vdw_surfaces: list
        Cartesian coordinates of the van der Waals surfaces on which the ESP was computed 
        (data originating from QMDFAM dataset).
    batch_num_nodes: torch.Tensor
        Ordered list of the number of atoms/nodes belonging to each molecule/graph in the batch of graphs.
    coordinates_multipoles: torch.Tensor
        Cartesian coordinates associated with the batch of molecules/graphs.
    pred_quadrupoles: np.ndarray
        Atomic quadrupole moment model predictions.

    Returns
    ----------
    ESP_up_to_quadrupoles: np.ndarray
        Electrostatic potential approximation contribution using atomic multipole predictions,
            up to atomic quadrupoles.

    """

    i_start = 0
    ESP_up_to_quadrupoles = []

    for mol_id, surface in enumerate(vdw_surfaces):  # Iterate over molecules
        
        if mol_id%1000 == 0:
            print("Index: ", mol_id, flush=True)

        i_end = i_start + batch_num_nodes[mol_id]

        coords = coordinates_multipoles[i_start:i_end]
        pred_quads = pred_quadrupoles[i_start:i_end]

        num_points = len(surface)

        # Compute pairwise distances: [num_atoms, num_points]
        distances = cdist(coords, surface)  

        coords = np.array(coords)
        surface = np.array(surface)

        # surface is [num_points, 3] -> [1, num_points, 3]
        # coords is [num_atoms, 3] -> [num_atoms, 1, 3]
        # Rij is [num_atoms, num_points, 3]
            # for each atom, we compute the distance between it and each point 
        Rij = surface[None, :, :] - coords[:, None, :] 

        R_squared = np.einsum("ijk,ijk->ij", Rij, Rij)  # [num_atoms, num_points]

        quadrupoles_esp = np.zeros(num_points)

        idx = 0 # mapping between 3x3 to 1x6 representation
        for alpha in range(3):
            for beta in range(alpha, 3):

                term =  3 * Rij[:, :, alpha] * Rij[:, :, beta] # [num_atoms, num_points]

                # This subtraction is to ensure tracelessness across the diagonal
                if alpha == beta:
                        term -= R_squared # [num_atoms, num_points] - [num_atoms, num_points] = [num_atoms, num_points]

                # term is [num_atoms, num_points]
                # pred_quads[:, idx] is [num_atoms]
                # denominator is [num_atoms, num_points]
                # quadrupoles_esp is [num_atoms, num_points]
                    # with sum over rows: [num_points]
                quadrupoles_esp += np.sum(np.einsum("ij,i->ij", term, pred_quads[:, idx]) / (2 * (distances**5)), axis=0)
                idx += 1


        quadrupoles_esp *= 1389.35
        ESP_up_to_quadrupoles.append(quadrupoles_esp)

        i_start = i_end


    ESP_up_to_quadrupoles = np.concatenate(ESP_up_to_quadrupoles)

    return ESP_up_to_quadrupoles


def compute_ESP_up_to_octupoles(
    vdw_surfaces: list, 
    batch_num_nodes: torch.Tensor, 
    coordinates_multipoles: torch.Tensor, 
    pred_octupoles: np.ndarray
):
    """
    Function for approximating the electrostatic potential (ESP)
        for a batch of molecules, using atomic multipole model predictions up to octupoles.

    Parameters
    -------
    vdw_surfaces: list
        Cartesian coordinates of the van der Waals surfaces on which the ESP was computed 
        (data originating from QMDFAM dataset).
    batch_num_nodes: torch.Tensor
        Ordered list of the number of atoms/nodes belonging to each molecule/graph in the batch of graphs.
    coordinates_multipoles: torch.Tensor
        Cartesian coordinates associated with the batch of molecules/graphs.
    pred_octupoles: np.ndarray
        Atomic octupole moment model predictions.

    Returns
    ----------
    ESP_up_to_octupoles: np.ndarray
        Electrostatic potential approximation contribution using atomic multipole predictions,
            up to atomic octupoles.

    """

    i_start = 0
    ESP_up_to_octupoles = []

    for mol_id, surface in enumerate(vdw_surfaces):  # Iterate over molecules
        
        if mol_id%1000 == 0:
            print("Index: ", mol_id, flush=True)

        i_end = i_start + batch_num_nodes[mol_id]

        coords = coordinates_multipoles[i_start:i_end]
        pred_octs = pred_octupoles[i_start:i_end]

        num_points = len(surface)

        # Compute pairwise distances: [num_atoms, num_points]
        distances = cdist(coords, surface)  

        coords = np.array(coords)
        surface = np.array(surface)

        # surface is [num_points, 3] -> [1, num_points, 3]
        # coords is [num_atoms, 3] -> [num_atoms, 1, 3]
        # Rij is [num_atoms, num_points, 3]
            # for each atom, we compute the distance between it and each point 
        Rij = surface[None, :, :] - coords[:, None, :] 

        R_squared = np.einsum("ijk,ijk->ij", Rij, Rij)  # [num_atoms, num_points]

        octupoles_esp = np.zeros(num_points)

        idx = 0 # mapping between 3x3x3 to 1x10 representation
        for alpha in range(3):
            for beta in range(alpha, 3):
                for gamma in range(beta, 3):

                    # 000, 001, 002, 011, 012, 022, 
                    # 111, 112, 122,
                    # 222

                    term =  15 * Rij[:, :, alpha] * Rij[:, :, beta] * Rij[:, :, gamma] # [num_atoms, num_points]

                    # These subtractions are to ensure tracelessness across the diagonals
                    # [num_atoms, num_points] - ([num_atoms, num_points] * [num_atoms, num_points]) = [num_atoms, num_points]
                    if beta == gamma:
                        term -= (R_squared * Rij[:, :, alpha])

                    if alpha == gamma:
                        term -= (R_squared * Rij[:, :, beta])

                    if alpha == beta:
                        term -= (R_squared * Rij[:, :, gamma])

                    # term is [num_atoms, num_points]
                    # pred_octs[:, idx] is [num_atoms]
                    # denominator is [num_atoms, num_points]
                    # octupoles_esp is [num_atoms, num_points]
                        # with sum over rows: [num_points]
                    octupoles_esp += np.sum(np.einsum("ij,i->ij", term, pred_octs[:, idx]) / (6 * (distances**7)), axis=0)
                    idx += 1


        octupoles_esp *= 1389.35
        ESP_up_to_octupoles.append(octupoles_esp)

        i_start = i_end


    ESP_up_to_octupoles = np.concatenate(ESP_up_to_octupoles)

    return ESP_up_to_octupoles

def reconstruct_esp(
    esp_data: h5py.File, 
    batch_num_nodes: torch.Tensor, 
    coordinates_multipoles: torch.Tensor, 
    unique_identifiers_multipoles: list,
    true_labels_by_multipole: list, 
    pred_labels_by_multipole: list, 
    device: torch.device
) -> tuple[float, float, float, float, float, float, float, float]:
    """
    Function for approximating the electrostatic potential (ESP)
        for a batch of molecules, using atomic multipole model predictions.

    Parameters
    -------
    esp_data: h5py.File
        QMDFAM dataset file containing electrostatic potential (ESP) data.
    batch_num_nodes: torch.Tensor
        Ordered list of the number of atoms/nodes belonging to each molecule/graph in the batch of graphs.
    coordinates_multipoles: torch.Tensor
        Cartesian coordinates associated with the batch of molecules/graphs.
    unique_identifiers_multipoles: list
        Unique identifier for matching the atomic multipole data associated with each molecule in the QMDFAM dataset
            with the electrostatic potential data associated with molecule.
    true_labels_by_multipole: list
        List of reference atomic multipole data.
    pred_labels_by_multipole: list
        List of predicted atomic multipole data.
    device: torch.device
        Either a cpu or cuda device.


    Returns
    ----------
    MAE_kcal_mon: float
        Mean absolute error (MAE) between the dataset reference ESP values 
            and the ESP approximation using the atomic monopole predictions (in kcal/mol).
    R2_mon: float
        Coefficient of determination (R^2) between the dataset reference ESP values 
            and the ESP approximation using the atomic monopole predictions.
    MAE_kcal_dip: float
        Mean absolute error (MAE) between the dataset reference ESP values 
            and the ESP approximation using the atomic multipole moment predictions
            up to atomic dipoles (in kcal/mol).
    R2_dip: float
        Coefficient of determination (R^2) between the dataset reference ESP values 
            and the ESP approximation using the atomic multipole moment predictions
            up to atomic dipoles.
    MAE_kcal_quad: float
        Mean absolute error (MAE) between the dataset reference ESP values 
            and the ESP approximation using the atomic multipole moment predictions
            up to atomic quadrupoles (in kcal/mol).
    R2_quad: float
        Coefficient of determination (R^2) between the dataset reference ESP values 
            and the ESP approximation using the atomic multipole moment predictions
            up to atomic quadrupoles.
    MAE_kcal_oct: float
        Mean absolute error (MAE) between the dataset reference ESP values 
            and the ESP approximation using the atomic multipole moment predictions
            up to atomic octupoles (in kcal/mol).
    R2_oct: float
        Coefficient of determination (R^2) between the dataset reference ESP values 
            and the ESP approximation using the atomic multipole moment predictions
            up to atomic octupoles.

    """
   
    """ Molecule coordinate data """

    coordinates_multipoles = list([x.tolist() for x in coordinates_multipoles])

    """ Predicted multipoles """

    pred_monopoles = pred_labels_by_multipole[0]
    pred_dipoles = pred_labels_by_multipole[1]
    pred_quadrupoles = pred_labels_by_multipole[2]
    pred_octupoles = pred_labels_by_multipole[3]


    """ Reference ESP data """

    unique_identifiers_esp = []
    vdw_surfaces = []
    esp_values = []

    for idx, key in enumerate(esp_data):

        hash_key = int(hashlib.md5(key.encode()).hexdigest(), 16) % (2**31)
        vdw = esp_data[key]["vdw_surface"][()]
        esp = esp_data[key]["esp"][()]

        unique_identifiers_esp.append(hash_key)
        vdw_surfaces.append(vdw)
        esp_values.append(esp)

    # Need to put the esp array in the order of the multipole array 
    # (extract the ones corresponding to the test set molecules and put them in the correct order)
    # Get the indices of elements in esp that are in multipole, in order
    esp_index_map = {hash_key : idx for idx, hash_key in enumerate(unique_identifiers_esp)}
    indices = [esp_index_map[hash_key] for hash_key in unique_identifiers_multipoles]

    # Extract the elements of esp that are in multipoles, maintaining the order in multipoles
    unique_identifiers_esp = [unique_identifiers_esp[i] for i in indices]
    vdw_surfaces = [vdw_surfaces[i] for i in indices]
    esp_values = [esp_values[i] for i in indices]

    esp_values = np.concatenate(esp_values) # dimension [total_num_molecules * total_num_points]
    esp_values = esp_values * 2625.5 # conversion from Hartree to kJ / mol

  
    """ Reconstruct ESP from multipole predictions"""

    print("\nUp to monopoles:")

    ESP_up_to_monopoles = compute_ESP_up_to_monopoles(vdw_surfaces, batch_num_nodes, coordinates_multipoles, pred_monopoles)

    MAE_mon = compute_MAE(esp_values, ESP_up_to_monopoles)
    MAE_kcal_mon = MAE_mon * 0.239001 # conversion to kcal/mol
    R2_mon = compute_R2(esp_values, ESP_up_to_monopoles)

    print("\nUp to dipoles:")

    ESP_up_to_dipoles = compute_ESP_up_to_dipoles(vdw_surfaces, batch_num_nodes, coordinates_multipoles, pred_dipoles)
    ESP_up_to_dipoles += ESP_up_to_monopoles

    MAE_dip = compute_MAE(esp_values, ESP_up_to_dipoles)
    MAE_kcal_dip = MAE_dip * 0.239001 # conversion to kcal/mol
    R2_dip = compute_R2(esp_values, ESP_up_to_dipoles)

    print("\nUp to quadrupoles:")

    ESP_up_to_quadrupoles = compute_ESP_up_to_quadrupoles(vdw_surfaces, batch_num_nodes, coordinates_multipoles, pred_quadrupoles)
    ESP_up_to_quadrupoles += ESP_up_to_dipoles

    MAE_quad = compute_MAE(esp_values, ESP_up_to_quadrupoles)
    MAE_kcal_quad = MAE_quad * 0.239001 # conversion to kcal/mol
    R2_quad = compute_R2(esp_values, ESP_up_to_quadrupoles)

    print("\nUp to octupoles:")

    ESP_up_to_octupoles = compute_ESP_up_to_octupoles(vdw_surfaces, batch_num_nodes, coordinates_multipoles, pred_octupoles)
    ESP_up_to_octupoles += ESP_up_to_quadrupoles

    MAE_oct = compute_MAE(esp_values, ESP_up_to_octupoles)
    MAE_kcal_oct = MAE_oct * 0.239001 # conversion to kcal/mol
    R2_oct = compute_R2(esp_values, ESP_up_to_octupoles)

    return MAE_kcal_mon, R2_mon, MAE_kcal_dip, R2_dip, MAE_kcal_quad, R2_quad, MAE_kcal_oct, R2_oct


def main(read_filepath_splits: str, read_filepath_esp: str, read_filepath_model: str):
    """
    Main function for performing inference on a trained PIL-Net model.

    Parameters
    ----------
    read_filepath_splits: str
        Path to test set file and 
            file containing the indices of the molecules for which the reference molecular quadrupole and octupole moments 
            were generated in ComputeReferenceMultipoles.py.
    read_filepath_esp: str
        Path to QMDFAM electrostatic potential hdf5 files.
    read_filepath_model: str
        Path to saved model(s).

    Returns
    -------
    None

    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    test_path = read_filepath_splits + "testdata.bin"
    rand_nums_path = read_filepath_splits + "molecular_multipole_indices.npy"
    esp_path = read_filepath_esp + "data_esp.hdf5"

    # Include more paths here to average result over multiple models
    bestmodel_paths = []
    for filename in os.listdir(read_filepath_model):
        if filename.endswith(".bin"):
            bestmodel_paths.append(read_filepath_model + filename)

    # Details related to QMDFAM
    element_types = [b"H", b"C", b"N", b"O", b"F", b"S", b"CL"]

    # Details related to neural network
    test_bsz = 2**10

    # PINN vs. Non-PINN has different predictive properties
    model_type = "PINN"
    # model_type = "Non-PINN"
    print(f"Model type: {model_type}")

    if model_type == "PINN":
        multipole_names = [
            "Monopoles",
            "Dipoles",
            "Quadrupoles",
            "Octupoles",
            "Molecular Dipole",
            "Molecular Quadrupole",
            "Molecular Octupole"
        ]
    elif model_type == "Non-PINN":
        multipole_names = ["Monopoles", "Dipoles", "Quadrupoles", "Octupoles"]


    ESP_multipole_types = ["Up to Monopoles", "Up to Dipoles", "Up to Quadrupoles", "Up to Octupoles"]

    # Load test dataset
    model_precision = torch.float32
    testgraphs = load_format_dataset(device, test_path, model_precision)

    # Load the esp dataset
    esp_data = h5py.File(esp_path, "r")

    # These are the indices of the reference molecular quadrupoles and octupoles
    # computed using psi4
    if model_type == "PINN":
        computed_multipole_indices = np.load(rand_nums_path)
        print(
            f"\nTest indices for molecular quadrupole and octupole moments: {computed_multipole_indices}"
        )
        print(f"Number of indices {len(computed_multipole_indices)}\n")
    else:
        computed_multipole_indices = [None]

    # Evaluate trained model on test set graphs
    test_network(
        device,
        bestmodel_paths,
        testgraphs,
        esp_data,
        test_bsz,
        model_type,
        multipole_names,
        ESP_multipole_types,
        element_types,
        computed_multipole_indices,
    )
