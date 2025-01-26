import optuna
import torch
from ogb.graphproppred import PygGraphPropPredDataset, Evaluator
from ogb.graphproppred.mol_encoder import AtomEncoder, BondEncoder
from torch import nn
from torch_geometric.loader import DataLoader

from models import DNA
from utils import sort_graphs


def train_DNA(
        trial,
        hidden_channels: int = 128,
        num_layers: int = 4,
        dropout: float = 0.3,
        lr: float = 1e-3,
        weight_decay: float = 1e-5,
        epochs: int = 50,
        device: int = 'cuda:0'):
    # Load device
    device = torch.device(device if torch.cuda.is_available() else 'cpu')

    # Load data
    dataset = PygGraphPropPredDataset(name='ogbg-molhiv', root='data/')
    split_idx = dataset.get_idx_split()
    dataset = sort_graphs(dataset, sort_y=False)
    train_loader = DataLoader(dataset[split_idx['train']], batch_size=256, shuffle=True)
    val_loader = DataLoader(dataset[split_idx['valid']], batch_size=256, shuffle=False)
    test_loader = DataLoader(dataset[split_idx['test']], batch_size=256, shuffle=False)

    # Instantiate model
    model = DNA(
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

        result_dict = evaluator.eval({
            'y_true': torch.cat(y_true, dim=0),
            'y_pred': torch.cat(y_pred, dim=0)})
        result = result_dict[evaluator.eval_metric]

        all_results.append(result)

        # # ---- Report to Optuna ----
        # # Step is the epoch index (0, 1, 2, ...)
        # trial.report(result, step=epoch)

        # # Check if we should prune (i.e., stop this trial early)
        # if trial.should_prune():
        #     raise optuna.TrialPruned()

    if epochs < 5:
        return sum(all_results) / len(all_results)
    else:
        return sum(all_results[-5:]) / 5


def objective(trial: optuna.Trial) -> float:
    # --- Search spaces for the hyperparameters ---
    hidden_channels = trial.suggest_categorical('hidden_channels', [64, 128, 256, 512])
    num_layers = trial.suggest_int('num_layers', 2, 6)
    dropout = trial.suggest_float('dropout', 0.0, 0.7)
    lr = trial.suggest_float('lr', 1e-4, 1e-2, log=True)
    weight_decay = trial.suggest_float('weight_decay', 1e-6, 1e-3, log=True)

    # --- Train model with these hyperparameters ---
    result = train_DNA(
        trial=trial,
        hidden_channels=hidden_channels,
        num_layers=num_layers,
        dropout=dropout,
        lr=lr,
        weight_decay=weight_decay,
        epochs=50,
        device='cuda:1'
    )

    # Since we want to maximise the validation ROC-AUC, we return it directly.
    return result


if __name__ == '__main__':
    # Create a study object
    # direction="maximize" because we want to maximize ROC-AUC
    study = optuna.create_study(
        direction='maximize',
        sampler=optuna.samplers.TPESampler(seed=42), 
        # pruner=optuna.pruners.MedianPruner(
        #     n_warmup_steps=10,  # no pruning the first 10 epochs
        #     interval_steps=1)    # check for pruning every epoch
    )

    # Optimize the objective function for N trials
    study.optimize(objective, n_trials=100)

    # Print out the best trial
    best_trial = study.best_trial
    print(f"Number of finished trials: {len(study.trials)}")
    print(f"Best trial:")
    print(f"  Value (Best Validation ROC-AUC): {best_trial.value}")
    print(f"  Params: ")
    for key, value in best_trial.params.items():
        print(f"    {key}: {value}")
