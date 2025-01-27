from statistics import median

import optuna
import torch
from ogb.graphproppred import PygGraphPropPredDataset, Evaluator
from ogb.graphproppred.mol_encoder import AtomEncoder, BondEncoder
from torch import nn
from torch_geometric.loader import DataLoader

import models
from utils import sort_graphs


def train_DNA(
        trial: optuna.Trial,
        hidden_channels: int = 128,
        num_layers: int = 4,
        dropout: float = 0.3,
        batch_size: int = 256,
        lr: float = 1e-3,
        weight_decay: float = 1e-5,
        epochs: int = 50,
        device: str = 'cuda:0'):
    # ---- Load device ----
    device = torch.device(device if torch.cuda.is_available() else 'cpu')

    # ---- Load data ----
    dataset = PygGraphPropPredDataset(name='ogbg-molhiv', root='data/')
    split_idx = dataset.get_idx_split()
    dataset = sort_graphs(dataset, sort_y=False)
    train_loader = DataLoader(dataset[split_idx['train']], batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(dataset[split_idx['valid']], batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(dataset[split_idx['test']], batch_size=batch_size, shuffle=False)

    # ---- Instantiate model ----
    model = models.DNA(
        in_channels=128,
        hidden_channels=hidden_channels,
        out_channels=hidden_channels,
        num_layers=num_layers,
        edge_dim=128,
        node_encoder=AtomEncoder(emb_dim=128),
        edge_encoder=BondEncoder(emb_dim=128),
        num_pred_heads=dataset.num_tasks,
        dropout=dropout,
        readout='mean',
    ).to(device)

    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max')
    evaluator = Evaluator(name='ogbg-molhiv')

    all_results = []

    for epoch in range(epochs):
        # ---- Training ----
        model.train()

        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            out = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
            loss = loss_fn(out, batch.y.float())
            loss.backward()
            optimizer.step()

        # ---- Validation ----
        model.eval()
        y_true = []
        y_pred = []
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                out = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
                y_true.append(batch.y)
                y_pred.append(out)

        result = evaluator.eval({
            'y_true': torch.cat(y_true, dim=0),
            'y_pred': torch.cat(y_pred, dim=0)})[evaluator.eval_metric]
        all_results.append(result)
        scheduler.step(result)

        # # ---- Pruning ----
        # trial.report(result, step=epoch)
        # if trial.should_prune():
        #     raise optuna.TrialPruned()

    if epochs < 5:
        return median(all_results)
    else:
        return median(all_results[-5:])


def objective(trial: optuna.Trial) -> float:
    # --- Search spaces for the hyperparameters ---
    hidden_channels = 256
    num_layers = trial.suggest_categorical('num_layers', [4, 6, 8, 10])
    dropout = trial.suggest_float('dropout', 0.0, 0.7)
    batch_size = 512
    lr = trial.suggest_float('lr', 1e-5, 1e-2, log=True)
    weight_decay = trial.suggest_float('weight_decay', 1e-6, 1e-3, log=True)

    # --- Train model with these hyperparameters ---
    result = train_DNA(
        trial=trial,
        hidden_channels=hidden_channels,
        num_layers=num_layers,
        dropout=dropout,
        batch_size=batch_size,
        lr=lr,
        weight_decay=weight_decay,
        epochs=50,
        device='cuda:1'
    )

    # Since we want to maximise the validation ROC-AUC, we return it directly.
    return result


if __name__ == '__main__':
    # Create a study object
    # direction='maximize' because we want to maximize ROC-AUC
    study = optuna.create_study(
        direction='maximize',
        sampler=optuna.samplers.TPESampler(seed=42),
        # pruner=optuna.pruners.MedianPruner(
        #     n_warmup_steps=10,  # No pruning the first 10 epochs
        #     interval_steps=1)   # Check for pruning every epoch
    )

    # Optimize the objective function for N trials
    study.optimize(objective, n_trials=50)

    # Print out the best trial
    best_trial = study.best_trial
    print(f"Number of finished trials: {len(study.trials)}")
    print(f"Best trial:")
    print(f"  Value (Best Validation ROC-AUC): {best_trial.value}")
    print(f"  Params: ")
    for key, value in best_trial.params.items():
        print(f"    {key}: {value}")
