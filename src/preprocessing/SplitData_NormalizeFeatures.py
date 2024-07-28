import torch

import dgl
from dgl.data.utils import save_graphs
from dgl.data.utils import load_graphs

import argparse


def split_assignment_setID(
    graphs: dgl.DGLGraph, trainset: list, validationset: list, testset: list
) -> tuple[list, list, list]:
    """Assign each graph its train/val/test split based on QMDFAM dataset specifications."""

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
    trainset: list, edgefeature_normalization_ids: list
) -> tuple[float, float]:
    """Get mean and std of specified edge features."""

    batch_graphs = dgl.batch(trainset)
    edge_features = batch_graphs.edata["efeats"][:, edgefeature_normalization_ids]

    mean = torch.mean(edge_features)
    std = torch.std(edge_features)

    return mean.item(), std.item()


def normalize_edge_feature(
    graphs: list, edgefeature_normalization_ids: list, mean: float, std: float
) -> list:
    """Normalize specified edge features in the graph based on input mean and standard deviation."""

    batch_graphs = dgl.batch(graphs)

    batch_graphs.edata["efeats"][:, edgefeature_normalization_ids] -= mean
    batch_graphs.edata["efeats"][:, edgefeature_normalization_ids] /= std

    normalized_graphs = dgl.unbatch(batch_graphs)

    return normalized_graphs


def main(read_filepath: str, save_filepath: str):

    trainset = []
    validationset = []
    testset = []

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

    print("Number of training graphs: {}".format(len(trainset)), flush=True)
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
