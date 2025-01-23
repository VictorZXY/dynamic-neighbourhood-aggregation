import os
import pickle
import time

import configargparse
import torch
import torch_geometric
import yaml
from configargparse import YAMLConfigFileParser
from torch_geometric.loader import DataLoader
from torch_geometric.nn.resolver import optimizer_resolver, lr_scheduler_resolver

from utils import Logger, dataset_resolver, evaluator_resolver, loss_resolver, model_resolver


@torch.no_grad()
def evaluate(model, loader, evaluator, loss_fn, device):
    model.eval()

    y_true = []
    y_pred = []
    total_loss = 0
    for data in loader:
        data = data.to(device)
        out = model(data.x, data.edge_index, data.edge_attr, data.batch)

        y_true.append(data.y)
        y_pred.append(out)
        if loss_fn is not None:
            loss = loss_fn(out, data.y)
            total_loss += loss.detach().item()

    result_dict = evaluator.eval({
        'y_true': torch.cat(y_true, dim=0),
        'y_pred': torch.cat(y_pred, dim=0)})
    if loss_fn is not None:
        result_dict['loss'] = total_loss / len(loader)

    return result_dict


def train(model, train_loader, val_loader, test_loader, train_args, device):
    model = model.to(device)
    print(f"# Params: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

    loss_kwargs = train_args['loss_kwargs'] if 'loss_kwargs' in train_args else {}
    loss_fn = loss_resolver(train_args['loss_fn'], **loss_kwargs)
    evaluator_kwargs = train_args['evaluator_kwargs'] if 'evaluator_kwargs' in train_args else {}
    evaluator = evaluator_resolver(train_args['evaluator'], **evaluator_kwargs)
    eval_metric = evaluator.eval_metric

    logger = Logger(train_args['runs'], eval_metric)
    run_times = []

    for run in range(train_args['runs']):
        start_time = time.time()
        model.reset_parameters()

        optimizer_kwargs = train_args['optimizer_kwargs'] if 'optimizer_kwargs' in train_args else {}
        optimizer = optimizer_resolver(train_args['optimizer'], model.parameters(), **optimizer_kwargs)
        scheduler_kwargs = train_args['scheduler_kwargs'] if 'scheduler_kwargs' in train_args else {}
        scheduler = lr_scheduler_resolver(train_args['scheduler'], optimizer, **scheduler_kwargs)

        for epoch in range(train_args['epochs']):
            model.train()
            optimizer.zero_grad()

            total_loss = 0
            for data in train_loader:
                data = data.to(device)
                out = model(data.x, data.edge_index, data.edge_attr, data.batch)
                loss = loss_fn(out, data.y)
                loss.backward()
                optimizer.step()
                total_loss += loss.detach().item()

            if (epoch + 1) % train_args['eval_interval'] == 0:
                train_result_dict = evaluate(model, train_loader, evaluator, loss_fn, device)
                val_result_dict = evaluate(model, val_loader, evaluator, loss_fn, device)
                test_result_dict = evaluate(model, test_loader, evaluator, loss_fn, device)
                logger.add_result(run, train_result_dict[eval_metric],
                                  val_result_dict[eval_metric],
                                  test_result_dict[eval_metric])
                print(f"Run {(run + 1):02d}, "
                      f"Epoch {(epoch + 1):3d}/{train_args['epochs']:3d}, "
                      f"Train loss: {train_result_dict['loss']:.4f}, "
                      f"Val loss: {val_result_dict['loss']:.4f}, "
                      f"Train {eval_metric}: {train_result_dict[eval_metric]:.4f}, "
                      f"Val {eval_metric}: {val_result_dict[eval_metric]:.4f}")

            scheduler.step()

        end_time = time.time()
        run_times.append(end_time - start_time)
        logger.print_statistics(run)

    return model, logger, run_times


def main(args):
    if args.cuda != -1:
        device = torch.device('cuda:' + str(args.cuda) if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device('cpu')
    print(f"Device: {device}")

    train_data, val_data, test_data = dataset_resolver(args.dataset, root=args.root, **(args.data_args or {}))
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False)
    print(f"Dataset name: {args.dataset}")

    model = model_resolver(args.model, args.dataset, train_data, **(args.model_args or {}))
    print(f"Model name: {args.model}")

    model, logger, run_times = train(model, train_loader, val_loader, test_loader, args.train_args, device)
    print(f"Finished Training {args.model} on {args.dataset} dataset")
    logger.print_statistics()
    if args.checkpoint_dir is not None:
        torch.save(model.state_dict(), os.path.join(args.checkpoint_dir, f'{args.experiment_name}.pt'))
    if args.log_dir is not None:
        with open(os.path.join(args.log_dir, f'{args.experiment_name}_logger.pickle'), 'wb') as f:
            pickle.dump(logger, f)


if __name__ == '__main__':
    parser = configargparse.ArgParser(config_file_parser_class=YAMLConfigFileParser)
    parser.add_argument('--config', is_config_file=True)
    parser.add_argument('--experiment_name', type=str, required=True)

    # Model specific arguments/hyperparameters
    parser.add_argument('--model', default='DNA', type=str)
    parser.add_argument('--model_args', default=None, type=yaml.safe_load)

    # Dataset specific arguments/hyperparameters
    parser.add_argument('--dataset', default='ogbg-molpcba', type=str)
    parser.add_argument('--root', default='data/', type=str)
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--data_args', default=None, type=yaml.safe_load)

    # Training specific arguments/hyperparameters
    parser.add_argument('--cuda', default=-1, type=int)
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--train_args', default=None, type=yaml.safe_load)
    parser.add_argument('--checkpoint_dir', default='checkpoints/', type=str)
    parser.add_argument('--log_dir', default='logs/', type=str)

    args = parser.parse_args()
    torch_geometric.seed_everything(args.seed)

    main(args)
