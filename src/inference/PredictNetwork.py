import numpy as np
import torch

import dgl
from dgl.data.utils import load_graphs

import os
import math
import copy
import argparse
# import psi4

from ..training.TrainNetwork import detrace_quadrupole_vector, detrace_octupole_vector
from ..neuralnetwork.PILNet import PILNet


def load_format_dataset(
    device: torch.device, test_path: str, model_precision: torch.dtype
) -> list:
    """Load, change precision, and detrace the graphs."""

    testgraphs = load_test_dataset(test_path)
    batch_testgraphs = dgl.batch(testgraphs)

    # Convert graph data/labels to have specified precision
    batch_testgraphs.ndata["nfeats"] = batch_testgraphs.ndata["nfeats"].to(
        model_precision
    )
    batch_testgraphs.edata["efeats"] = batch_testgraphs.edata["efeats"].to(
        model_precision
    )

    batch_testgraphs.ndata["label_monopoles"] = batch_testgraphs.ndata[
        "label_monopoles"
    ].to(model_precision)
    batch_testgraphs.ndata["label_dipoles"] = batch_testgraphs.ndata[
        "label_dipoles"
    ].to(model_precision)
    batch_testgraphs.ndata["label_quadrupoles"] = batch_testgraphs.ndata[
        "label_quadrupoles"
    ].to(model_precision)
    batch_testgraphs.ndata["label_octupoles"] = batch_testgraphs.ndata[
        "label_octupoles"
    ].to(model_precision)

    batch_testgraphs.ndata["coordinates"] = batch_testgraphs.ndata["coordinates"].to(
        model_precision
    )
    batch_testgraphs.ndata["relative_coordinates"] = batch_testgraphs.ndata[
        "relative_coordinates"
    ].to(model_precision)
    batch_testgraphs.ndata["molecular_dipole"] = batch_testgraphs.ndata[
        "molecular_dipole"
    ].to(model_precision)

    # Detrace the quadrupole and octupole labels
    batch_testgraphs = detrace_quadrupole_vector(batch_testgraphs)
    batch_testgraphs = detrace_octupole_vector(batch_testgraphs)

    batch_testgraphs = batch_testgraphs.to(device)
    testgraphs = dgl.unbatch(batch_testgraphs)

    return testgraphs


def load_test_dataset(test_path: str) -> list:
    """Load graphs."""
    testgraphs = load_graphs(test_path)
    testgraphs = testgraphs[0]
    return testgraphs


def compute_MAE(true: np.ndarray, pred: np.ndarray) -> float:
    """Compute mean absolute error."""
    return np.mean((abs(true - pred)))


def compute_R2(true: np.ndarray, pred: np.ndarray) -> float:
    """Compute coefficient of determination."""
    RSS = np.sum((true - pred) ** 2)
    mean_true = np.mean(true, axis=0)
    TSS = np.sum((true - mean_true) ** 2)

    return 1 - (RSS / TSS)


def compute_RMSD(true: np.ndarray, pred: np.ndarray) -> float:
    """Compute root mean squared deviation."""
    return np.sqrt(np.mean((true - pred)**2))


def print_testing_statistics(
    true: np.ndarray, pred: np.ndarray
) -> tuple[float, float, float, float, float, float, float]:
    """Return statistical information about the true and predicted labels."""

    MAE = compute_MAE(true, pred)
    R2 = compute_R2(true, pred)
    RMSD = compute_RMSD(true, pred)

    mean_true = np.mean(pred, axis=0)
    mean_pred = np.mean(true, axis=0)
    stdev_true = np.std(pred, axis=0)
    stdev_pred = np.std(true, axis=0)

    return MAE, R2, RMSD, mean_true, mean_pred, stdev_true, stdev_pred


def conv_one_hot(onehot: torch.Tensor, element_types: list) -> bytes:
    """Convert one hot encoding to element type."""
    for i in range(len(onehot)):
        if onehot[i] == 1:
            return element_types[i]

    raise ValueError(
        f"The one-hot pattern {onehot} does not exist within the pre-defined element types list: {element_types}"
    )


def element_specific_statistics(
    true_labels: np.ndarray,
    pred_labels: np.ndarray,
    nfeats: np.ndarray,
    element_types: list,
    target_elem: bytes,
) -> tuple[float, float]:
    """Obtain element-specific MAE and R2 between reference and predicted labels."""

    elem_true_labels = np.zeros(0)
    elem_pred_labels = np.zeros(0)

    for i in range(len(nfeats)):

        # If this feature corresponds to the target element:
        if conv_one_hot(nfeats[i], element_types) == target_elem:

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


def get_reference_molecular_dipole_moments(
    testbgs: dgl.DGLGraph, 
) -> torch.tensor:
    '''Obtain the reference molecular dipole moment from
        stored information from the dataset through reformatting.'''

    batch_num_nodes = testbgs.batch_num_nodes()
    sum_batch_num_nodes = copy.deepcopy(batch_num_nodes)

    running_sum = 0
    for j in range(len(sum_batch_num_nodes)):
        sum_batch_num_nodes[j] = running_sum
        running_sum += batch_num_nodes[j]

    true_mol_dipole = (
        testbgs.ndata["molecular_dipole"][sum_batch_num_nodes, :]
    )

    return true_mol_dipole
    


# def compute_reference_molecular_moment(elements: list[str], coordinates: list[float]) -> list:

#     # Set-up psi4
#     psi4.set_options({'basis': 'def2-TZVP'})

#     # Get wave function from molecule
#     psi4_mol = psi4.core.Molecule.from_arrays(coordinates, elements)
#     _, wfn = psi4.energy('PBE0', molecule=psi4_mol, return_wfn=True)

#     # Extract multipole moment properties from wave function
#     psi4.oeprop(wfn, 'GRID_ESP', 'MULTIPOLE(3)', title='MBIS Multipole Moments')
#     wfn_variables = wfn.variables()

#     # MBIS quadrupole moment
#     ref_quadrupole_moments = wfn_variables['MBIS Multipole Moments QUADUPOLE']
#     ref_quadrupole_moments = detrace_quadrupole_vector(ref_quadrupole_moments * B_to_A ** 2).numpy()

#     # MBIS octupole moment
#     ref_octupole_moments = wfn_variables['MBIS Multipole Moments OCTUPOLE']
#     ref_octupole_moments = detrace_octupole_vector(ref_octupole_moments * B_to_A ** 3).numpy()

#     return ref_molecular_quadrupoles, ref_octupole_moments


def approximate_molecular_moment(
    testbgs: dgl.DGLGraph, 
    atomic_monopole: torch.tensor, 
    atomic_multipoles_lower_order: list[torch.tensor], 
    relative_coordinates: torch.tensor
) -> torch.tensor:
    '''Approximate molecular moments as a function of 
        their corresponding lower-order atomic multipole predictions
        and atomic relative coordinates'''

    # Multiply atomic multipoles by atomic relative coordinates
    monopole_contribution = (
        torch.reshape(atomic_monopole, (-1, 1)) * relative_coordinates
    )

    # Sum over the monopole contribution and lower-order atomic multipoles
    testbgs.ndata["approx_molecular_moment"] = monopole_contribution
    for atomic_multipole in atomic_multipoles_lower_order:
        testbgs.ndata["approx_molecular_moment"] += atomic_multipole
    
    # Take a sum over all the atoms' molecular moment contributions
        # to obtain a single vector per molecule
    preds_molecular_moment = dgl.readout_nodes(
        graph=testbgs, feat="approx_molecular_moment", op="sum"
    )

    return preds_molecular_moment


def test_network(
    device: torch.device,
    bestmodel_paths: list,
    testgraphs: list,
    test_bsz: int,
    model_type: str,
    multipole_names: list,
    element_types: list,
) -> None:
    """Make predictions on test set from trained model
    NOTE: This code assumes the feature and label orders determined in ExtractFeatures.py
        e.g., The one-hot encoded elment-type feature is accessible as testbgs.ndata['nfeats'][:,0:7]
        e.g., The model returns label predictions in the order atomic monopole, dipole, quadrupole, and then octupole
    """

    # Create lists to store MAE and R^2 computed information
    num_graphs = len(testgraphs)

    num_models = len(bestmodel_paths)
    num_multipoles = len(multipole_names)
    num_element_types = len(element_types)

    overall_MAE = np.zeros((num_models, num_multipoles))
    overall_R2 = np.zeros((num_models, num_multipoles))

    if model_type == "PINN":
        # Do not include Molecular Dipole here since it is a graph-level property
        elem_MAE = np.zeros((num_models, num_multipoles - 1, num_element_types))
        elem_R2 = np.zeros((num_models, num_multipoles - 1, num_element_types))
    else:
        elem_MAE = np.zeros((num_models, num_multipoles, num_element_types))
        elem_R2 = np.zeros((num_models, num_multipoles, num_element_types))

    # Iterate over each trained model
    print("\n** Predictive errors averaged across {} model(s): **".format(num_models), flush=True)
    for i in range(num_models):

        print("\nModel path: {}".format(bestmodel_paths[i]), flush=True)

        # Load saved model
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

                # If model is of type PINN, approximate the molecular dipole
                if model_type == "PINN":

                    atomic_monopole = preds[:, 0:1]
                    atomic_dipole = preds[:, 1:4]
                    relative_coordinates = testbgs.ndata["relative_coordinates"]
    
                    # Approximate the molecular dipole moment
                    preds_mol_dipole = approximate_molecular_moment(
                        testbgs, atomic_monopole, [atomic_dipole], relative_coordinates
                    )

                    predlabels_mol_dipole = torch.cat(
                        (predlabels_mol_dipole, preds_mol_dipole)
                    )

                    # Approximate the molecular quadrupole moment
                    # Approximate the molecular octupole moment

                    # Compute higher-order molecular moments using psi4
                    # true_mol_quadrupole, true_mol_octupole = compute_reference_molecular_moments(elements: list[str], coordinates: list[float])


                predlabels = torch.cat((predlabels, preds))

            start = end
            end += test_bsz

        # Convert format of dataset reference labels for atomic dipoles
        testbgs = dgl.batch(testgraphs)

        true_monopoles = testbgs.ndata["label_monopoles"].cpu().numpy()
        true_dipoles = testbgs.ndata["label_dipoles"].cpu().numpy()
        true_quadrupoles = testbgs.ndata["label_quadrupoles"].cpu().numpy()
        true_octupoles = testbgs.ndata["label_octupoles"].cpu().numpy()

        pred_monopoles = predlabels[:, 0:1].cpu().numpy()
        pred_dipoles = predlabels[:, 1:4].cpu().numpy()
        pred_quadrupoles = predlabels[:, 4:10].cpu().numpy()
        pred_octupoles = predlabels[:, 10:20].cpu().numpy()

        predlabels = predlabels.cpu().numpy()

        # If model is of type PINN, obtain molecular dipole moment for each molecule
        if model_type == "PINN":

            true_mol_dipole = get_reference_molecular_dipole_moments(testbgs)
            true_mol_dipole = true_mol_dipole.cpu().numpy()

            predlabels_mol_dipole = predlabels_mol_dipole.cpu().numpy()

        # Organize reference and predicted labels in lists
        true_labels_by_multipole = [
            true_monopoles,
            true_dipoles,
            true_quadrupoles,
            true_octupoles,
        ]
        pred_labels_by_multipole = [
            pred_monopoles,
            pred_dipoles,
            pred_quadrupoles,
            pred_octupoles,
        ]

        if model_type == "PINN":
            true_labels_by_multipole.append(true_mol_dipole)
            pred_labels_by_multipole.append(predlabels_mol_dipole)

        # Keep running average of predictions
        if i == 0:
            average_predlabels = predlabels
            average_predlabels_moldip = predlabels_mol_dipole
        else:
            average_predlabels += predlabels
            average_predlabels_moldip += predlabels_mol_dipole

        # Record MAE and R^2 for each multipole type
        for j in range(len(multipole_names)):
            MAE, R2, _, _, _, _, _ = (
                print_testing_statistics(
                    true_labels_by_multipole[j], pred_labels_by_multipole[j]
                )
            )
            overall_MAE[i, j] = MAE
            overall_R2[i, j] = R2

        # Record element-specific MAE and R^2 for each multipole type
        nfeats = testbgs.ndata["nfeats"][:, 0:7].cpu().numpy()
        num_atomic_multipoles = 4
        for j in range(num_atomic_multipoles):
            for k in range(len(element_types)):
                elem_MAE[i, j, k], elem_R2[i, j, k] = element_specific_statistics(
                    true_labels_by_multipole[j],
                    pred_labels_by_multipole[j],
                    nfeats,
                    element_types,
                    element_types[k],
                )

    # Average predictive results and display prediction statistics information to user
    average_predlabels /= num_models
    average_pred_labels_by_multipole = [
        average_predlabels[:, 0:1],
        average_predlabels[:, 1:4],
        average_predlabels[:, 4:10],
        average_predlabels[:, 10:20],
    ]
    if model_type == "PINN":
        average_predlabels_moldip /= num_models
        average_pred_labels_by_multipole = [
            average_predlabels[:, 0:1],
            average_predlabels[:, 1:4],
            average_predlabels[:, 4:10],
            average_predlabels[:, 10:20],
            average_predlabels_moldip,
        ]

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


def main(read_filepath_splits: str, read_filepath_model: str):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    test_path = read_filepath_splits + "testdata.bin"

    # Include more paths here to average result over multiple models
    bestmodel_paths = []
    for filename in os.listdir(read_filepath_model):
        if filename.endswith('.bin'):
            bestmodel_paths.append(read_filepath_model + filename)

    # Details related to QMDFAM
    element_types = [b"H", b"C", b"N", b"O", b"F", b"S", b"CL"]

    # Details related to neural network
    test_bsz = 2**10

    # PINN vs. Non-PINN has different predictive properties
    model_type = "PINN"
    # model_type = "Non-PINN"

    if model_type == "PINN":
        multipole_names = [
            "Monopoles",
            "Dipoles",
            "Quadrupoles",
            "Octupoles",
            "Molecular Dipole",
        ]
    elif model_type == "Non-PINN":
        multipole_names = ["Monopoles", "Dipoles", "Quadrupoles", "Octupoles"]

    # Load test dataset
    model_precision = torch.float32
    testgraphs = load_format_dataset(device, test_path, model_precision)

    # Evaluate trained model on test set graphs
    test_network(
        device,
        bestmodel_paths,
        testgraphs,
        test_bsz,
        model_type,
        multipole_names,
        element_types,
    )
