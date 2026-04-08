# Project Context

## Project Overview
- This project studies LDNA, short for Learnable Dynamic Neighborhood Aggregation. LDNA works by sorting graph inputs before model execution and then training a learnable graph neural network over those sorted graphs.
- This repository contains code for LDNA and several baseline GNN architectures on graph and node learning tasks.
- This repository is centered on running experiments rather than packaging a reusable library.

## Project Scope
- The main training entry point is `train.py`.
- Experiments are configured through YAML files in `configs/`.
- `utils/resolver.py` loads datasets, applies preprocessing, and builds the requested model.
- Sorting is performed in `utils/_utils.py` and applied during dataset preparation from `utils/resolver.py`.
- The repository compares LDNA against several baseline GNN model families under the same general training pipeline.
- The current repository contents are centered on graph-level experiments such as binary classification on MolHIV, regression on ZINC, and multi-class classification on MNIST superpixels.
- LDNA is not limited to graph-level prediction, and future evaluation is expected to include node datasets as well.

## Current Datasets
- `ogbg-molhiv`: graph-level binary classification.
- `ZINC`: graph-level regression.
- `MNISTSuperpixels`: graph-level multi-class classification.

## Model Families
- LDNA model: `LDNA` in `models/ldna_net.py` with `LDNAConv` in `models/ldna_conv.py`.
- Baseline models in `models/`: `GCN`, `GIN`, `GINE`, `GraphSAGE`, `GAT`, `GATv2`, `PNA`, `EGC`, and `DeeperGCN`.

## Code Map
- LDNA model code: `models/ldna_conv.py` and `models/ldna_net.py`.
- Dataset and model resolution: `utils/resolver.py`.
- Graph sorting and dataset-side preprocessing support: `utils/_utils.py`.
- Small transforms: `utils/transforms.py`.
- Evaluators and training logs: `utils/evaluator.py` and `utils/logger.py`.

## Configuration
- Experiments are configured through YAML files in `configs/`.
- Common top-level config fields include `experiment_name`, `model`, `model_args`, `dataset`, `data_args`, `train_args`, `checkpoint_dir`, and `log_dir`.
