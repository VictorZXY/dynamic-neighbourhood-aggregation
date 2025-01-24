import numpy as np
import torch
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.utils import sort_edge_index


def sort_graph(graph: Data, sort_y=False):
    x_arr = graph.x.numpy()
    sorted_idx = np.lexsort([x_arr[:, i] for i in range(x_arr.shape[1])])
    sorted_idx = torch.from_numpy(sorted_idx).long()
    inv_sorted_idx = torch.empty_like(sorted_idx)
    inv_sorted_idx[sorted_idx] = torch.arange(sorted_idx.shape[0])

    graph.new2old = sorted_idx
    graph.old2new = inv_sorted_idx

    # Sort the nodes
    graph.x = graph.x[graph.new2old]
    if sort_y:
        graph.y = graph.y[graph.new2old]
    graph.edge_index = graph.old2new[graph.edge_index]

    # Sort the edge indices and attributes
    if hasattr(graph, 'edge_attr'):
        graph.edge_index, graph.edge_attr = sort_edge_index(graph.edge_index, graph.edge_attr)
    else:
        graph.edge_index = sort_edge_index(graph.edge_index)

    return graph


def sort_graphs(dataset: InMemoryDataset, sort_y=False):
    all_graphs = [dataset.get(i) for i in range(len(dataset))]
    sorted_graphs = [sort_graph(g, sort_y) for g in all_graphs]

    dataset.data, dataset.slices = dataset.collate(sorted_graphs)
    return dataset
