import ogb.graphproppred
import ogb.nodeproppred
import torch
from ogb.graphproppred import PygGraphPropPredDataset
from ogb.graphproppred.mol_encoder import AtomEncoder, BondEncoder
from torch import nn
from torch_geometric.datasets import ZINC
from torch_geometric.loader import DataLoader
from torch_geometric.utils import degree

import models
from utils import sort_graphs
from utils.evaluator import ZINCEvaluator
from utils.transforms import ZINCTransform


def model_and_data_resolver(model_query, dataset_query, **kwargs):
    model_kwargs = kwargs.get('model_args', {})
    dataset_kwargs = kwargs.get('data_args', {})
    batch_size = dataset_kwargs.pop('batch_size', 1)

    model_choices = ['DNA', 'DeeperGCN', 'EGC', 'GCN', 'GIN', 'GINE', 'PNA']
    dataset_choices = ['ogbg-molhiv', 'ogbg-molpcba', 'ZINC']

    # Load the dataset
    if dataset_query in ['ogbg-molhiv', 'ogbg-molpcba']:
        dataset = PygGraphPropPredDataset(name=dataset_query, **dataset_kwargs)
        split_idx = dataset.get_idx_split()

        dataset = sort_graphs(dataset, sort_y=False)
        train_dataset = dataset[split_idx['train']]
        val_dataset = dataset[split_idx['valid']]
        test_dataset = dataset[split_idx['test']]
    elif dataset_query == 'ZINC':
        zinc_transform = ZINCTransform()
        train_dataset = ZINC('data/zinc', subset=False, split='train', pre_transform=zinc_transform)
        val_dataset = ZINC('data/zinc', subset=False, split='val', pre_transform=zinc_transform)
        test_dataset = ZINC('data/zinc', subset=False, split='test', pre_transform=zinc_transform)
        
        train_dataset = sort_graphs(train_dataset, sort_y=False)
        val_dataset = sort_graphs(val_dataset, sort_y=False)
        test_dataset = sort_graphs(test_dataset, sort_y=False)
    else:
        raise ValueError(f"Could not resolve dataset '{dataset_query}' among choices {dataset_choices}")

    # Split the dataset into train/val/test dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Update model kwargs
    if dataset_query in ['ogbg-molhiv', 'ogbg-molpcba']:
        embedding_dim = model_kwargs['hidden_channels'] if model_query == 'DeeperGCN' else 128
        model_kwargs.update({
            'in_channels': embedding_dim,
            'edge_dim': embedding_dim,
            'node_encoder': AtomEncoder(emb_dim=embedding_dim),
            'edge_encoder': BondEncoder(emb_dim=embedding_dim),
            'num_pred_heads': dataset.num_tasks
        })
    elif dataset_query == 'ZINC':
        embedding_dim = model_kwargs['hidden_channels'] if model_query == 'DeeperGCN' else 128
        model_kwargs.update({
            'in_channels': embedding_dim,
            'edge_dim': embedding_dim,
            'node_encoder': models.Encoder(28, embedding_dim=embedding_dim, num_features=1),
            'edge_encoder': nn.Embedding(4, embedding_dim=embedding_dim),
            'num_pred_heads': 1
        })

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
        'ZINCEvaluator': ZINCEvaluator,
    }

    if query not in choices:
        raise ValueError(f"Could not resolve evaluator '{query}' among choices {choices.keys()}")

    return choices[query](**kwargs)
