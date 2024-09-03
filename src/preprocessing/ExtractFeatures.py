"""Read in dataset information, extract features,
    and save each molecule representation as a DGL graph.
"""

import h5py
import numpy as np
import torch

import dgl
from dgl.data.utils import save_graphs

import rdkit
from rdkit.Chem.rdchem import HybridizationType
from rdkit.Chem.rdchem import BondType


def getdataset(
    data: str,
    save_filepath: str,
    unique_element_list: list,
    unique_hybridization_list: list,
    unique_bondtype_list: list,
) -> None:
    """Read in dataset information, extract features,
    and save molecule representation as a DGL graph."""

    print(f"\nLoading {data}", flush=True)
    h = h5py.File(data, "r")

    missing_data_flag = False
    graphs = []
    graph_count = 0

    """ Form one graph per molecule """
    for key in h:

        """Check for missing data; Do not include molecules with missing data"""
        keywords = [
            "monopoles",
            "dipoles",
            "quadrupoles",
            "octupoles",
            "smiles",
            "elements",
            "coordinates",
        ]
        for val in keywords:
            if len(h[key][val][()]) == 0:  # type: ignore
                missing_data_flag = True
                break
        if missing_data_flag:
            missing_data_flag = False
            continue

        """ Start building graph of this molecule """
        g = dgl.DGLGraph()
        smiles = h[key]["smiles"][()]  # type: ignore
        mol = rdkit.Chem.MolFromSmiles(smiles)  # type: ignore
        mol = rdkit.Chem.AddHs(mol)  # type: ignore

        """ Add nodes and node features """

        # Node feature: One-hot encoded element type
        elements = h[key]["elements"][()]  # type: ignore
        nodefeats_elementtype = get_one_hot(elements, unique_element_list)  # type: ignore

        # Node feature: One-hot encoded hybridization state
        hybridization_state = []
        mol_atoms = mol.GetAtoms()

        for i in range(len(mol_atoms)):
            hybridization = mol_atoms[i].GetHybridization()
            hybridization_state.append(hybridization)

        nodefeats_hybridizationstate = get_one_hot(
            hybridization_state, unique_hybridization_list
        )

        # Note: These two features (from dataset and from rdkit) have the same atom ordering
        nodefeats = np.hstack((nodefeats_elementtype, nodefeats_hybridizationstate))
        num_nodes = len(elements)  # type: ignore
        g.add_nodes(num_nodes, data={"nfeats": torch.tensor(nodefeats)})

        """ Add coordinate feature """

        coordinates = h[key]["coordinates"][()]  # type: ignore

        molecule_mass = 0.0
        for i in range(len(elements)):  # type: ignore
            elem_mass = compute_mass(elements[i])  # type: ignore
            molecule_mass += elem_mass

            if i == 0:
                center_mass_position = elem_mass * coordinates[i, :]  # type: ignore
            else:
                center_mass_position += elem_mass * coordinates[i, :]  # type: ignore

        center_mass_position /= molecule_mass  # type: ignore
        relative_coordinates = coordinates - center_mass_position  # type: ignore

        g.ndata["coordinates"] = torch.tensor(coordinates)
        center_mass_position = [center_mass_position] * num_nodes
        g.ndata["com_coordinates"] = torch.tensor(np.array(center_mass_position))
        g.ndata["relative_coordinates"] = torch.tensor(relative_coordinates)

        """ Add edges and edge features """

        bonds = mol.GetBonds()
        for bond in bonds:

            atom1_idx = bond.GetBeginAtom().GetIdx()
            atom2_idx = bond.GetEndAtom().GetIdx()

            # Edge feature: Weighted interatomic distance
            # (to be normalized in separate file)
            distance = np.linalg.norm(
                coordinates[atom1_idx] - coordinates[atom2_idx]  # type: ignore
            )
            scaled_weight = np.exp(-np.power(distance, 2) / 2)

            atom1_tensor = torch.tensor([atom1_idx])
            atom2_tensor = torch.tensor([atom2_idx])

            # Edge feature: One-hot encoded bond type
            bond_type = str(bond.GetBondType())
            edgefeats_bondtype = get_one_hot([bond_type], unique_bondtype_list)[0]

            # Edge feature: One-hot encoded aromaticity
            aromaticity = bond.GetIsAromatic()
            if aromaticity is True:
                aromaticity = 1
            else:
                aromaticity = 0

            # Edge feature: One-hot encoded conjugacy
            conjugacy = bond.GetIsConjugated()
            if conjugacy is True:
                conjugacy = 1
            else:
                conjugacy = 0

            # Edge feature: One-hot incoded ring membership
            ring_membership = bond.IsInRing()
            if ring_membership is True:
                ring_membership = 1
            else:
                ring_membership = 0

            edgefeats = (
                [scaled_weight]
                + edgefeats_bondtype.tolist()
                + [aromaticity]
                + [conjugacy]
                + [ring_membership]
            )
            edgefeats = [edgefeats]

            g.add_edges(
                atom1_tensor, atom2_tensor, data={"efeats": torch.tensor(edgefeats)}
            )
            g.add_edges(
                atom2_tensor, atom1_tensor, data={"efeats": torch.tensor(edgefeats)}
            )

        """ Add label and training split information """

        # Labels: Save each multipole label as node data of the graph
        #  (not used until model evaluation)
        g.ndata["label_monopoles"] = torch.tensor(h[key]["monopoles"][()])  # type: ignore
        g.ndata["label_dipoles"] = torch.tensor(h[key]["dipoles"][()])  # type: ignore
        g.ndata["label_quadrupoles"] = torch.tensor(h[key]["quadrupoles"][()])  # type: ignore
        g.ndata["label_octupoles"] = torch.tensor(h[key]["octupoles"][()])  # type: ignore

        molecular_dipole = [h[key]["molecular_dipole"][()]] * num_nodes  # type: ignore
        g.ndata["molecular_dipole"] = torch.tensor(np.array(molecular_dipole))
        molecular_dipole_mbis = [h[key]["molecular_dipole_mbis"][()]] * num_nodes  # type: ignore
        g.ndata["molecular_dipole_mbis"] = torch.tensor(np.array(molecular_dipole_mbis))

        # Split information: Used to partition molecules into correct
        # train/val/test split as determined by dataset
        set = h[key]["set"][()]  # type: ignore
        if set == b"train" or set == b"train_gdb":
            set_list = [0] * num_nodes
        elif set == b"validation" or set == b"validation_gdb":
            set_list = [1] * num_nodes
        elif set == b"test" or set == b"test_gdb":
            set_list = [2] * num_nodes
        g.ndata["set"] = torch.tensor(set_list)

        # Printed reference for users
        if graph_count % 1000 == 0:
            print(f"Molecule: {graph_count}", flush=True)

        """ Append graph to graph list """
        graphs.append(g)
        graph_count += 1

    """ Save graphs to bin file """
    save_graphs(save_filepath, graphs)

    print("Number of graphs: {}".format(graph_count))


def get_one_hot(values: list, unique_values_list: list) -> np.ndarray:
    """Apply one-hot encoding of input values"""

    numatoms = len(values)
    numcategories = len(unique_values_list)
    onehot = np.zeros((numatoms, numcategories))
    flag = False

    for i in range(numatoms):
        for j in range(numcategories):
            if values[i] == unique_values_list[j]:
                onehot[i, j] = 1
                flag = True
                break
        if flag is False:
            print("**Invalid element found in list: ", values[i], flush=True)
            flag = True

    return onehot


def compute_mass(elem: bytes) -> float:
    """Return mass of input element"""

    if elem == b"H":
        return 1.008
    elif elem == b"C":
        return 12.011
    elif elem == b"N":
        return 14.007
    elif elem == b"O":
        return 15.999
    elif elem == b"F":
        return 18.998
    elif elem == b"S":
        return 32.066
    elif elem == b"CL":
        return 35.453
    raise ValueError(f"The element variable {elem!r} is an unexpected value.")


def main(read_filepath: str, save_filepath: str):

    # Details related to QMDFAM, some derived from the RDKit library
    unique_element_list = [b"H", b"C", b"N", b"O", b"F", b"S", b"CL"]
    unique_hybridization_list = list(HybridizationType.names.values())
    unique_bondtype_list = list(BondType.names)

    # QMDFAM File 1
    dataset1_hdf5_file = read_filepath + "data.hdf5"
    getdataset(
        dataset1_hdf5_file,
        save_filepath + "graph_data.bin",
        unique_element_list,
        unique_hybridization_list,
        unique_bondtype_list,
    )

    # QMDFAM File 2
    dataset2_hdf5_file = read_filepath + "data_gdb.hdf5"
    getdataset(
        dataset2_hdf5_file,
        save_filepath + "graph_gdb_data.bin",
        unique_element_list,
        unique_hybridization_list,
        unique_bondtype_list,
    )
