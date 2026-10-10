import os
import json
import random
import warnings

import numpy as np
import torch

warnings.filterwarnings("ignore")
torch.set_num_threads(1)

CFG = dict(
    data_dir="/content",
    out_dir="/content/apistox_v4",
    mode="main",
    splits=["random", "maxmin", "time"],
    seeds=list(range(10)),
    abl_splits=["random"],
    val_frac=0.10,
    test_frac=0.20,
    hidden=64,
    n_layers=3,
    dropout=0.2,
    n_experts=3,
    cen_epochs=80,
    cen_lr=1e-3,
    batch_size=32,
    patience=20,
    rounds=40,
    local_epochs=2,
    fl_lr=1e-3,
    sel_k=200,
    corr_thr=0.95,
    n_bins=100,
    hp=dict(mu=0.01, adam_lr=0.01, scaf_lr=0.05),
)

for key, var in (("data_dir", "APIS_DATA"), ("out_dir", "APIS_OUT"), ("mode", "APIS_MODE")):
    if os.environ.get(var):
        CFG[key] = os.environ[var]
if os.environ.get("APIS_SPLITS"):
    CFG["splits"] = os.environ["APIS_SPLITS"].split(",")
if os.environ.get("APIS_ABL"):
    CFG["abl_splits"] = os.environ["APIS_ABL"].split(",")
if os.environ.get("APIS_SEEDS"):
    CFG["seeds"] = [int(x) for x in os.environ["APIS_SEEDS"].split(",")]
if os.environ.get("APIS_ROUNDS"):
    CFG["rounds"] = int(os.environ["APIS_ROUNDS"])

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUT = CFG["out_dir"]
for sub in ("rows", "probs", "clients", "extra"):
    os.makedirs(os.path.join(OUT, sub), exist_ok=True)

tuning_file = os.path.join(OUT, "tuning.json")
if os.path.exists(tuning_file):
    CFG["hp"].update(json.load(open(tuning_file)))


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
