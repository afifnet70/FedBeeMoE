import json
import os
import pickle
import time

import numpy as np
import pandas as pd
import torch

from config import CFG, OUT, set_seed
from data import SRC, Y, make_split
from metrics import METRICS, compute_metrics
from federated_prep import fed_prepare
from training import finish, per_client_metrics, run_federated
from models import nparams, predict
from baselines import run_central, run_classical, run_local_only

FED_MAIN = [
    ("FedAvg-GCN", "fedavg", "gcn"),
    ("FedAvg-GIN", "fedavg", "gin"),
    ("FedProx-GIN", "fedprox", "gin"),
    ("FedAdam-GIN", "fedadam", "gin"),
    ("SCAFFOLD-GIN", "scaffold", "gin"),
    ("FedAvg-GIN-MoE", "fedavg", "gin_moe"),
    ("FedAvg-GIN+Desc", "fedavg", "gin_desc"),
    ("FedAvg-GIN+Desc (dense head, param-matched)", "fedavg", "gin_dense"),
    ("FedAvg-MLP (FP)", "fedavg", "mlp"),
    ("FedAvg-MLP (Sel)", "fedavg", "mlp_sel"),
    ("FedBeeMoE [FedAvg]", "fedavg", "bee"),
    ("FedBeeMoE [FedProx]", "fedprox", "bee"),
    ("FedBeeMoE [FedAdam]", "fedadam", "bee"),
]
FED_ABL = [
    ("FedBeeMoE K=1", "fedavg", "bee_k1"),
    ("FedBeeMoE K=2", "fedavg", "bee_k2"),
    ("FedBeeMoE K=5", "fedavg", "bee_k5"),
    ("FedBeeMoE top-1 gate", "fedavg", "bee_top1"),
    ("FedBeeMoE uniform gate", "fedavg", "bee_uni"),
]
GROUP = lambda n: (
    "Federated (graph)"
    if n.startswith(("Fed", "SCAFFOLD")) and "MLP" not in n
    else (
        "Federated (fingerprint MLP)"
        if "MLP" in n and n.startswith("Fed")
        else (
            "Local-only"
            if n.startswith("Local")
            else "Pooled (neural)" if n.startswith("Pooled") else "Pooled (classical ML)"
        )
    )
)


def record(rows, clrows, probs, split, seed, name, group, prob, thr, te, sel=None, hist=None, extra=None):
    r05 = compute_metrics(Y[te], prob, 0.5)
    r = compute_metrics(Y[te], prob, thr)
    row = dict(
        Split=split,
        Group=group,
        Model=name,
        Seed=seed,
        Thr=float(np.mean(thr)),
        Val_Score=(sel or {}).get("score", np.nan),
        Val_MCC=(sel or {}).get("mcc", np.nan),
        Val_AUC=(sel or {}).get("auc", np.nan),
        **r,
        **{"t05_" + k: r05[k] for k in METRICS},
    )
    rows.append(row)
    for c, v in per_client_metrics(te, prob, thr).items():
        clrows.append(dict(Split=split, Model=name, Seed=seed, Client=c, **v))
    probs[name] = (prob.astype(np.float32), np.asarray(thr, dtype=np.float32))
    print(
        f"   {name:46s} acc={r['Accuracy']:.3f} rec={r['Recall']:.3f} auc={r['AUC']:.3f} mcc={r['MCC']:.3f} "
        f"prauc={r['PR_AUC']:.3f} val={row['Val_Score']:.3f}",
        flush=True,
    )


def run_split_seed(split, seed):
    tag = f"{split}_{seed}"
    if os.path.exists(os.path.join(OUT, "rows", tag + ".csv")):
        print("skip", tag)
        return
    t0 = time.time()
    set_seed(seed)
    tr, va, te = make_split(split, seed)
    info = fed_prepare(tr)
    print(
        f"\n=== {split} seed {seed} | train {len(tr)} val {len(va)} test {len(te)} | test toxic {Y[te].mean():.3f} | {info['funnel']}",
        flush=True,
    )
    rows, clrows, probs, extra = [], [], {}, dict(idx=te, info=info, nparam={})
    jobs = FED_MAIN + (FED_ABL if split in CFG["abl_splits"] else [])
    for disp, algo, nm in jobs:
        set_seed(seed)
        gm, sel, hist = run_federated(algo, nm, tr, va, te)
        isbee = nm.startswith("bee")
        pt, gate = finish(gm, nm, sel, te, extra_gate=isbee and gm.head == "moe")
        record(rows, clrows, probs, split, seed, disp, GROUP(disp), pt, sel["thr"], te, sel, hist)
        extra["nparam"][disp] = nparams(gm)
        if disp in ("FedBeeMoE [FedAvg]", "FedAvg-GIN-MoE"):
            extra.setdefault("hist", {})[disp] = hist
        if isbee and gm.head == "moe" and gate is not None:
            extra.setdefault("gate", {})[disp] = gate
            if disp in ("FedBeeMoE [FedAvg]", "FedBeeMoE [FedProx]", "FedBeeMoE [FedAdam]") or nm == "bee":
                ko = {}
                for k in [-1] + list(range(gm.n_exp)):
                    mask = torch.ones(gm.n_exp) if k < 0 else torch.ones(gm.n_exp)
                    if k < 0:
                        mask = torch.zeros(gm.n_exp)
                    else:
                        mask[k] = 0
                    gm.expert_mask = mask
                    pk = predict(gm, nm, te)
                    ko[k] = compute_metrics(Y[te], pk, sel["thr"])
                gm.expert_mask = None
                extra.setdefault("knockout", {})[disp] = ko
    set_seed(seed)
    for disp, nm in (("Local-GIN", "gin"), ("Local-GIN-MoE", "gin_moe")):
        p, th = run_local_only(nm, tr, va, te)
        record(rows, clrows, probs, split, seed, disp, "Local-only", p, th, te)
    for disp, nm in (("Pooled-GIN", "gin"), ("Pooled-GIN+Desc", "gin_desc"), ("Pooled-GIN-MoE+Desc", "bee")):
        set_seed(seed)
        model, sel = run_central(nm, tr, va, te)
        record(
            rows,
            clrows,
            probs,
            split,
            seed,
            disp,
            "Pooled (neural)",
            predict(model, nm, te),
            sel["thr"],
            te,
            sel,
        )
    for disp, (p, sel) in run_classical(tr, va, te, seed).items():
        record(rows, clrows, probs, split, seed, disp, "Pooled (classical ML)", p, sel["thr"], te, sel)
    pd.DataFrame(clrows).to_csv(os.path.join(OUT, "clients", tag + ".csv"), index=False)
    pickle.dump(
        dict(probs=probs, y=Y[te], src=SRC[te], idx=te), open(os.path.join(OUT, "probs", tag + ".pkl"), "wb")
    )
    pickle.dump(extra, open(os.path.join(OUT, "extra", tag + ".pkl"), "wb"))
    pd.DataFrame(rows).to_csv(os.path.join(OUT, "rows", tag + ".csv"), index=False)
    print(f"   [{tag} finished in {time.time() - t0:.0f}s]", flush=True)


def tune():
    res = {}
    grid = [
        ("fedprox", "mu", [0.001, 0.01, 0.1]),
        ("fedadam", "adam_lr", [0.003, 0.01, 0.03]),
        ("scaffold", "scaf_lr", [0.02, 0.05, 0.1]),
    ]
    for algo, key, vals in grid:
        sc = {v: [] for v in vals}
        for seed in (0, 1, 2):
            tr, va, te = make_split("random", seed)
            fed_prepare(tr)
            for v in vals:
                set_seed(seed)
                hp = dict(CFG["hp"])
                hp[key] = v
                _, sel, _ = run_federated(algo, "gin", tr, va, te, hp)
                sc[v].append(sel["score"])
                print(f"  tune {algo} {key}={v} seed {seed}: val score {sel['score']:.3f}", flush=True)
        res[key] = max(vals, key=lambda v: np.mean(sc[v]))
        res[key + "_curve"] = {str(v): float(np.mean(s)) for v, s in sc.items()}
    json.dump(res, open(os.path.join(OUT, "tuning.json"), "w"), indent=1)
    print("TUNED:", res)


if __name__ == "__main__":
    if CFG["mode"] == "tune":
        tune()
    else:
        for split in CFG["splits"]:
            for seed in CFG["seeds"]:
                run_split_seed(split, seed)
        print("ALL DONE")
