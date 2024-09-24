"""Use PSI4 to compute reference molecular quadrupole and octupole multipole moments
(unavailable in QMDFAM) for some of the test set graphs."""

import psi4

import numpy as np
from scipy.spatial.distance import cdist
import torch

import dgl
from dgl.data.utils import load_graphs, save_graphs

import time
import random


def load_format_dataset(test_path: str, model_precision: torch.dtype) -> list:
    """Load and change precision of the graphs."""

    testgraphs = load_test_dataset(test_path)
    batch_testgraphs = dgl.batch(testgraphs)

    # Convert graph data/labels to have specified precision
    batch_testgraphs.ndata["nfeats"] = batch_testgraphs.ndata["nfeats"].to(  # type: ignore
        model_precision
    )

    batch_testgraphs.ndata["coordinates"] = batch_testgraphs.ndata[
        "coordinates"
    ].to(  # type: ignore
        model_precision
    )

    testgraphs = dgl.unbatch(batch_testgraphs)

    return testgraphs


def load_test_dataset(test_path: str) -> list:
    """Load graphs."""
    testgraphs = load_graphs(test_path)
    testgraphs = testgraphs[0]
    return testgraphs


def compute_reference_molecular_moments(
    elements: np.ndarray, coordinates: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Obtain reference molecular quadrupole and octupole moments from psi4.
    Code adapted from EquivariantMultipoleGNN/how_to_use.ipynb script available on GitHub.
    """

    # Generate grid.dat file so that 'GRID_ESP' method can integrate
    # the ESP over a grid surrounding the molecule
    N_points, CUTOFF = 400, 4.0
    surface_points = np.random.normal(size=[N_points, 3])

    surface_points = (
        surface_points / np.linalg.norm(surface_points, axis=-1, keepdims=True)
    ) * CUTOFF

    surface_points = np.reshape(surface_points[None] + coordinates[:, None], [-1, 3])

    surface_points = surface_points[
        np.where(
            np.all(cdist(surface_points, coordinates) >= (CUTOFF - 1e-1), axis=-1)
        )[0]
    ]

    with open("grid.dat", "w") as file:
        for xyz in surface_points:
            for c in xyz:
                file.write(str(c) + " ")
            file.write("\n")

    # Set-up psi4
    psi4.set_options({"basis": "def2-TZVP"})

    # Get wave function from molecule
    psi4_mol = psi4.core.Molecule.from_arrays(coordinates, elem=elements)
    _, wfn = psi4.energy("PBE0", molecule=psi4_mol, return_wfn=True)  # type: ignore

    # Extract multipole moment properties from wave function
    psi4.oeprop(wfn, "GRID_ESP", "MULTIPOLE(3)", title="MBIS Multipole Moments")  # type: ignore
    wfn_variables = wfn.variables()

    # Convert from Bohr radius to Angstrom
    B_to_A = 0.529177249

    # MBIS quadrupole moment
    ref_quadrupole_moments = wfn_variables[
        "MBIS MULTIPOLE MOMENTS QUADRUPOLE"
    ].to_array()
    ref_quadrupole_moments = ref_quadrupole_moments * (B_to_A**2)

    # MBIS octupole moment
    ref_octupole_moments = wfn_variables["MBIS MULTIPOLE MOMENTS OCTUPOLE"].to_array()
    ref_octupole_moments = ref_octupole_moments * (B_to_A**3)

    return ref_quadrupole_moments, ref_octupole_moments


def conv_one_hot_to_str(onehot: torch.Tensor, element_types: list) -> str:
    """Convert one hot encoding to element type."""
    for i in range(len(onehot)):
        if onehot[i] == 1:
            return element_types[i].decode("utf-8")

    raise ValueError(
        f"The one-hot pattern {onehot} does not exist"
        f"within the pre-defined element types list: {element_types}"
    )


def get_multipole_moments(
    testgraphs: list,
    num_reference_molecules: int, 
    seed_value: int,
    element_types: list, 
    save_filepath: str
) -> None:
    """Use psi4 library to obtain reference molecular quadrupole
    and molecular octupole moments for the test set.
    """

    elapsed_time = 0.0

    # Generate "num_reference_molecules" random numbers, with seeded value 
    # for consistency when using values in PredictNetwork.py
    random.seed(seed_value)
    rand_nums = random.sample(range(len(testgraphs)), num_reference_molecules)
    print(f"Random numbers: {rand_nums}", flush=True)

    # Save the chosen random numbers to external file
    with open("src/data/random_numbers.npy", "wb") as file:
        np.save(file, np.array(rand_nums))

    # Iterate over specified test graphs
    for i in range(len(testgraphs)):

        graph = testgraphs[i]

        # Only compute the reference molecular moments for graphs that correspond to random number
        if i in rand_nums:

            # Compute higher-order molecular moments using psi4
            elements = graph.ndata["nfeats"][:, 0:7]
            elements = [
                conv_one_hot_to_str(elements[i], element_types)
                for i in range(len(elements))
            ]

            elements = np.array(elements)
            coordinates = graph.ndata["coordinates"].numpy()

            print(f"Generating molecular moments for molecule {i}: ", flush=True)
            starttime = time.time()
            mol_quad, mol_oct = compute_reference_molecular_moments(
                elements, coordinates
            )
            elapsed_time += time.time() - starttime

            mol_quad = np.squeeze(mol_quad)
            mol_oct = np.squeeze(mol_oct)

        # If the graph is not among those randomly selected, assign arrays of all zeroes
        else:

            mol_quad = np.zeros(6)
            mol_oct = np.zeros(10)

        num_nodes = graph.number_of_nodes()

        molecular_quadrupole = [mol_quad] * num_nodes
        graph.ndata["molecular_quadrupole"] = torch.tensor(
            np.array(molecular_quadrupole)
        )

        molecular_octupole = [mol_oct] * num_nodes
        graph.ndata["molecular_octupole"] = torch.tensor(np.array(molecular_octupole))

        testgraphs[i] = graph

    save_graphs(save_filepath, testgraphs)
    print(f"Elapsed Time: {elapsed_time/60} minutes")


def main(read_filepath: str, save_filepath: str):

    test_read_filepath = read_filepath + "testdata.bin"
    test_save_filepath = save_filepath + "testdata.bin"

    # Details related to QMDFAM
    element_types = [b"H", b"C", b"N", b"O", b"F", b"S", b"CL"]

    # Load test dataset
    model_precision = torch.float32
    testgraphs = load_format_dataset(test_read_filepath, model_precision)

    # Number of molecules for which to compute reference multipole moments
    num_reference_molecules = 50
    # For consistent random number generation
    seed_value = 42

    # Obtain the reference multipole moments
    get_multipole_moments(testgraphs, num_reference_molecules, seed_value, element_types, test_save_filepath)
