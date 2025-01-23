import ogb.graphproppred
import ogb.nodeproppred
from ogb.graphproppred import PygGraphPropPredDataset
from ogb.graphproppred.mol_encoder import AtomEncoder, BondEncoder
from torch import nn

import models


def dataset_resolver(query, **kwargs):
    choices = {
        'ogbg-molpcba': PygGraphPropPredDataset
    }

    if query not in choices:
        raise ValueError(f"Could not resolve dataset '{query}' among choices {choices.keys()}")

    if 'ogbg' in query:
        dataset = choices[query](name=query, **kwargs)
        split_idx = dataset.get_idx_split()
        train_data = dataset[split_idx['train']]
        val_data = dataset[split_idx['valid']]
        test_data = dataset[split_idx['test']]

    return train_data, val_data, test_data


def model_resolver(query, dataset_name, dataset, **kwargs):
    if hasattr(models, query):
        cls = getattr(models, query)
        if not callable(cls):
            raise ValueError(f"Could not resolve model '{query}'")
    else:
        raise ValueError(f"Could not resolve model '{query}'")

    dataset_choices = ['ogbg-molpcba']
    if dataset_name not in dataset_choices:
        raise ValueError(f"Could not resolve dataset '{dataset_name}' among choices {dataset_choices}")

    if 'ogbg-mol' in dataset_name:
        kwargs.update({
            'in_channels': 128,
            'edge_dim': 128,
            'node_encoder': AtomEncoder(emb_dim=128),
            'edge_encoder': BondEncoder(emb_dim=128),
            'num_pred_heads': dataset.num_tasks
        })

    return cls(**kwargs)


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
