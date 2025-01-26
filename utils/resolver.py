import ogb.graphproppred
import ogb.nodeproppred
import torch
from ogb.graphproppred import PygGraphPropPredDataset
from ogb.graphproppred.mol_encoder import AtomEncoder, BondEncoder
from torch import nn
from torch_geometric.loader import DataLoader
from torch_geometric.utils import degree

import models
from utils import sort_graphs


def model_and_data_resolver(model_query, dataset_query, **kwargs):
    model_kwargs = kwargs.get('model_args', {})
    dataset_kwargs = kwargs.get('data_args', {})
    batch_size = dataset_kwargs.pop('batch_size', 1)
    task_type = dataset_kwargs.pop('task_type', '')

    model_choices = ['DNA']
    dataset_choices = ['ogbg-molhiv', 'ogbg-molpcba']

    # Load the dataset
    if dataset_query in ['ogbg-molhiv', 'ogbg-molpcba']:
        dataset = PygGraphPropPredDataset(name=dataset_query, **dataset_kwargs)
        split_idx = dataset.get_idx_split()
    else:
        raise ValueError(f"Could not resolve dataset '{dataset_query}' among choices {dataset_choices}")

    # Sort the nodes and edge indices in the dataset
    if 'graph' in task_type:
        dataset = sort_graphs(dataset, sort_y=False)
    elif 'node' in task_type:
        dataset = sort_graphs(dataset, sort_y=True)
    else:
        raise ValueError(f"Could not resolve task type '{task_type}'. "
                         f"Please specify the task type in the format: (node|graph) (classification|regression)")

    # Split the dataset into train/val/test dataloaders, and update model kwargs
    if dataset_query in ['ogbg-molhiv', 'ogbg-molpcba']:
        emb_dim = model_kwargs['hidden_channels'] if model_query == 'DeeperGCN' else 128
        model_kwargs.update({
            'in_channels': emb_dim,
            'edge_dim': emb_dim,
            'node_encoder': AtomEncoder(emb_dim=emb_dim),
            'edge_encoder': BondEncoder(emb_dim=emb_dim),
            'num_pred_heads': dataset.num_tasks
        })
        train_loader = DataLoader(dataset[split_idx["train"]], batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(dataset[split_idx["valid"]], batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(dataset[split_idx["test"]], batch_size=batch_size, shuffle=False)

    # Load the model
    if model_query == 'DNA':
        model = models.DNA(**model_kwargs)
    elif model_query == 'DeeperGCN':
        model = models.DeeperGCN(**model_kwargs)
    elif model_query == 'EGC':
        model = models.EGC(**model_kwargs)
    elif model_query == 'GCN':
        model = models.GCN(**model_kwargs)
    elif model_query == 'GIN':
        model = models.GIN(**model_kwargs)
    elif model_query == 'GINE':
        model = models.GINE(**model_kwargs)
    elif model_query == 'PNA':
        train_dataset = dataset[split_idx["train"]]

        # Compute the maximum in-degree in the training data.
        max_degree = -1
        for data in train_dataset:
            d = degree(data.edge_index[1], num_nodes=data.num_nodes, dtype=torch.long)
            max_degree = max(max_degree, int(d.max()))

        # Compute the in-degree histogram tensor
        deg = torch.zeros(max_degree + 1, dtype=torch.long)
        for data in train_dataset:
            d = degree(data.edge_index[1], num_nodes=data.num_nodes, dtype=torch.long)
            deg += torch.bincount(d, minlength=deg.numel())

        model = models.PNA(deg=deg, **model_kwargs)
    else:
        raise ValueError(f"Could not resolve dataset '{model_query}' among choices {model_choices}")

    return model, train_loader, val_loader, test_loader


def loss_resolver(query, **kwargs):
    if hasattr(nn, query):
        cls = getattr(nn, query)
        if callable(cls):
            return cls(**kwargs)
        else:
            raise ValueError(f"Could not resolve loss '{query}'")
    else:
        raise ValueError(f"Could not resolve loss '{query}'")


def evaluator_resolver(query, **kwargs):
    choices = {
        'OGBNodePropPredEvaluator': ogb.nodeproppred.Evaluator,
        'OGBGraphPropPredEvaluator': ogb.graphproppred.Evaluator,
    }

    if query not in choices:
        raise ValueError(f"Could not resolve evaluator '{query}' among choices {choices.keys()}")

    return choices[query](**kwargs)
