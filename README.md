# FedBeeMoE

Federated graph isomorphism network with a residual mixture-of-experts head for honey bee toxicity prediction on ApisTox.

## Layout

| File | Content |
| --- | --- |
| `config.py` | settings, seeds, paths |
| `data.py` | ApisTox loading, splits, fingerprints, molecular graphs, descriptors |
| `federated_prep.py` | federated imputation, filtering, correlation pruning and feature selection |
| `metrics.py` | test metrics and the histogram-based federated validation score |
| `models.py` | GCN, GIN, mixture-of-experts and dense heads, fingerprint MLP |
| `training.py` | FedAvg, FedProx, FedAdam, SCAFFOLD, federated validation |
| `baselines.py` | local-only, pooled neural and pooled classical baselines |
| `run.py` | experiment driver |
| `analysis/` | result tables, statistics and figures |

## Data

Download `dataset_final.csv` and the `splits/` folder from the ApisTox release (https://doi.org/10.5281/zenodo.13350981) and point `APIS_DATA` to them.

## Run

```
pip install -r requirements.txt
export APIS_DATA=/path/to/apistox APIS_OUT=/path/to/output
APIS_MODE=tune python run.py
python run.py
python analysis/analyze.py
```

Optional variables: `APIS_SPLITS` (random, maxmin, time), `APIS_SEEDS` (for example 0,1,2), `APIS_ROUNDS`, `APIS_ABL`.
Each split and seed is saved separately, so an interrupted run continues where it stopped.

## Figures

```
python analysis/fig_eda.py
python analysis/fig_tsne.py
python analysis/fig_discovery.py
```
