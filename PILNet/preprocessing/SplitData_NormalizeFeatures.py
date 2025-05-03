"""Separate data into train/validation/test set splits and normalize features.
"""

import torch

import dgl
from dgl.data.utils import save_graphs
from dgl.data.utils import load_graphs


def split_assignment_setID(
    graphs: list[dgl.DGLGraph], trainset: list, validationset: list, testset: list
) -> tuple[list[dgl.DGLGraph], list[dgl.DGLGraph], list[dgl.DGLGraph]]:
    """
    Function to assign each graph its training/val/test split based on the QMDFAM dataset specifications.

    Parameters
    -------
    graphs: list[dgl.DGLGraph]
        List of graph-structured data.
    trainset: list
        Empty list.
    validationset: list
        Empty list.
    testset: list
        Empty list.

    Returns
    ----------
    trainset: list[dgl.DGLGraph]
        List of DGL graphs assigned to the training set.
    validationset: list[dgl.DGLGraph]
        List of DGL graphs assigned to the validation set.
    testset: list[dgl.DGLGraph]
        List of DGL graphs assigned to the test set.

    """

    for i in range(len(graphs)):

        g = graphs[i]
        g_set = g.ndata["set"][0]

        if g_set == 0:
            trainset.append(g)
        elif g_set == 1:
            validationset.append(g)
        elif g_set == 2:
            testset.append(g)

    return trainset, validationset, testset


def get_edge_feature_distribution(
    trainset: list[dgl.DGLGraph], edgefeature_normalization_ids: list[int]
) -> tuple[float, float]:
    """
    Function to obtain the mean and standard deviation of the specified edge features.

    Parameters
    -------
    trainset: list[dgl.DGLGraph]
        List of graphs belonging to the training set.
    edgefeature_normalization_ids: list[int]
        List of identifiers associated with edge feature(s) to normalize.

    Returns
    ----------
    mean: float
        Mean of the specified edge feature(s)
    std: float
        Standard deviation of the specified edge feature(s)

    """

    batch_graphs = dgl.batch(trainset)
    edge_features = batch_graphs.edata["efeats"][:, edgefeature_normalization_ids]

    mean = torch.mean(edge_features)
    std = torch.std(edge_features)

    return mean.item(), std.item()


def normalize_edge_feature(
    graphs: list[dgl.DGLGraph], edgefeature_normalization_ids: list[int], mean: float, std: float
) -> list[dgl.DGLGraph]:
    """
    Function to normalize the specified edge features in the graph,
        based on the input mean and standard deviation.

    Parameters
    -------
    trainset: list[dgl.DGLGraph]
        List of graphs belonging to the training set.
    edgefeature_normalization_ids: list[int]
        List of identifiers associated with edge feature(s) to normalize.
    mean: float
        Mean of the specified edge feature(s)
    std: float
        Standard deviation of the specified edge feature(s)


    Returns
    ----------
    normalized_graphs: list[dgl.DGLGraph]
        List of DGL graphs with the specifed edge features normalized.

    """

    batch_graphs = dgl.batch(graphs)

    batch_graphs.edata["efeats"][:, edgefeature_normalization_ids] -= mean
    batch_graphs.edata["efeats"][:, edgefeature_normalization_ids] /= std

    normalized_graphs = dgl.unbatch(batch_graphs)

    return normalized_graphs


def main(read_filepath: str, save_filepath: str):
    """
    Main function for splitting dataset into training/validation/test sets
        and for normalizing the relevant feature.

    Parameters
    ----------
    read_filepath: str
        constructed QMDFAM graphs and formatted labels
    save_filepath: str
        Path to save dataset training/validation/test splits.

    Returns
    -------
    None

    """

    trainset = []  # type: list[dgl.DGLGraph]
    validationset = []  # type: list[dgl.DGLGraph]
    testset = []  # type: list[dgl.DGLGraph]

    """ Load graphs and assign to specified train/validation/test split """

    graphs_data = load_graphs(read_filepath + "graph_data.bin")[0]
    trainset, validationset, testset = split_assignment_setID(
        graphs_data, trainset, validationset, testset
    )
    graphs_data = []

    graphs_gdb_data = load_graphs(read_filepath + "graph_gdb_data.bin")[0]
    trainset, validationset, testset = split_assignment_setID(
        graphs_gdb_data, trainset, validationset, testset
    )
    graphs_gdb_data = []

    print("\nNumber of training graphs: {}".format(len(trainset)), flush=True)
    print("Number of validation graphs: {}".format(len(validationset)), flush=True)
    print("Number of test graphs {}".format(len(testset)), flush=True)

    """ Normalize edge features in all sets based on training set statistics """

    # Only normalize the first edge feature type
    edgefeature_normalization_ids = [0]

    # Normalize the edge features using distribution information from the training set
    mean, std = get_edge_feature_distribution(trainset, edgefeature_normalization_ids)

    trainset = normalize_edge_feature(
        trainset, edgefeature_normalization_ids, mean, std
    )
    validationset = normalize_edge_feature(
        validationset, edgefeature_normalization_ids, mean, std
    )
    testset = normalize_edge_feature(testset, edgefeature_normalization_ids, mean, std)

    """ Save the graph splits """

    save_graphs(save_filepath + "traindata.bin", trainset)
    save_graphs(save_filepath + "validationdata.bin", validationset)
    save_graphs(save_filepath + "testdata.bin", testset)
    
    print("Graph splits saved.")
