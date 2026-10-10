import copy

from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
import numpy as np
import torch
import torch.nn.functional as F

from config import CFG
from models import build, class_weights, collate, needs_graph, predict, to_dev
from metrics import client_hist, select_from_hists
from training import pooled_val
from data import CLIENTS, FP, SRC, Y
import data


def run_local_only(name, tr, va, te):
    prob = np.zeros(len(te))
    thr = np.full(len(te), 0.5)
    for c in CLIENTS:
        ids, vid, tm = tr[SRC[tr] == c], va[SRC[va] == c], SRC[te] == c
        if len(ids) < 4 or tm.sum() == 0:
            continue
        model = build(name)
        opt = torch.optim.Adam(model.parameters(), lr=CFG["fl_lr"], weight_decay=1e-5)
        cw = class_weights(ids)
        best, bst, bsel = -9, None, None
        for ep in range(CFG["rounds"] * CFG["local_epochs"]):
            model.train()
            perm = np.random.permutation(ids)
            for i in range(0, len(perm), CFG["batch_size"]):
                ch = perm[i : i + CFG["batch_size"]]
                if len(ch) < 2:
                    continue
                b = to_dev(collate(ch, needs_graph(name)))
                loss = F.cross_entropy(model(b), b["y"], weight=cw)
                opt.zero_grad()
                loss.backward()
                opt.step()
            if len(vid) >= 4 and len(set(Y[vid])) == 2:
                s = pooled_val(model, name, vid)
                if s["score"] > best:
                    best, bst, bsel = s["score"], copy.deepcopy(model.state_dict()), s
        if bst is not None:
            model.load_state_dict(bst)
        prob[tm] = predict(model, name, te[tm])
        thr[tm] = bsel["thr"] if bsel else 0.5
    return prob, thr


def run_central(name, tr, va, te):
    model = build(name)
    opt = torch.optim.Adam(model.parameters(), lr=CFG["cen_lr"], weight_decay=1e-5)
    cw = class_weights(tr)
    best, bst, bsel, bad = -9, None, None, 0
    for ep in range(CFG["cen_epochs"]):
        model.train()
        perm = np.random.permutation(tr)
        for i in range(0, len(perm), CFG["batch_size"]):
            ch = perm[i : i + CFG["batch_size"]]
            if len(ch) < 2:
                continue
            b = to_dev(collate(ch, needs_graph(name)))
            loss = F.cross_entropy(model(b), b["y"], weight=cw)
            opt.zero_grad()
            loss.backward()
            opt.step()
        s = pooled_val(model, name, va)
        if s["score"] > best:
            best, bst, bsel, bad = s["score"], copy.deepcopy(model.state_dict()), s, 0
        else:
            bad += 1
            if bad >= CFG["patience"]:
                break
    model.load_state_dict(bst)
    return model, bsel


def run_classical(tr, va, te, seed):
    Xf, Xs = FP.astype(np.float32), data.XS
    out = {}

    def fit_eval(label, X, mk, grid):
        best = None
        for g in grid:
            clf = mk(g).fit(X[tr], Y[tr])
            pv = clf.predict_proba(X[va])[:, 1]
            s = select_from_hists(*[h.astype(float) for h in client_hist(Y[va], pv)])
            if best is None or s["score"] > best[0]["score"]:
                best = (s, clf)
        s, clf = best
        out[label] = (clf.predict_proba(X[te])[:, 1], s)

    sp = float((Y[tr] == 0).sum() / max((Y[tr] == 1).sum(), 1))
    fit_eval(
        "LogReg (FP)",
        Xf,
        lambda C: LogisticRegression(C=C, max_iter=2000, class_weight="balanced"),
        [0.01, 0.1, 1.0],
    )
    fit_eval(
        "SVM-RBF (FP)",
        Xf,
        lambda C: SVC(C=C, kernel="rbf", probability=True, class_weight="balanced", random_state=seed),
        [1.0, 3.0, 10.0],
    )
    fit_eval(
        "k-NN (FP)",
        Xf.astype(bool),
        lambda k: KNeighborsClassifier(k, metric="jaccard", algorithm="brute", weights="distance"),
        [3, 7, 15],
    )
    fit_eval(
        "Random Forest (FP)",
        Xf,
        lambda m: RandomForestClassifier(
            500, class_weight="balanced", min_samples_leaf=m, n_jobs=1, random_state=seed
        ),
        [1, 2],
    )
    fit_eval(
        "Extra Trees (FP)",
        Xf,
        lambda m: ExtraTreesClassifier(
            500, class_weight="balanced", min_samples_leaf=m, n_jobs=1, random_state=seed
        ),
        [1, 2],
    )
    fit_eval(
        "Gradient Boosting (FP)",
        Xf,
        lambda d: GradientBoostingClassifier(
            n_estimators=200, max_depth=d, learning_rate=0.05, subsample=0.8, random_state=seed
        ),
        [3],
    )
    fit_eval(
        "XGBoost (FP)",
        Xf,
        lambda d: XGBClassifier(
            n_estimators=300,
            max_depth=d,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.5,
            scale_pos_weight=sp,
            n_jobs=1,
            random_state=seed,
            verbosity=0,
            eval_metric="logloss",
        ),
        [3, 5],
    )
    fit_eval(
        "MLP-sklearn (FP)",
        Xf,
        lambda a: MLPClassifier((256, 64), alpha=a, max_iter=400, early_stopping=True, random_state=seed),
        [1e-3, 1e-2],
    )
    fit_eval(
        "Random Forest (Sel)",
        Xs,
        lambda m: RandomForestClassifier(
            500, class_weight="balanced", min_samples_leaf=m, n_jobs=1, random_state=seed
        ),
        [1, 2],
    )
    fit_eval(
        "XGBoost (Sel)",
        Xs,
        lambda d: XGBClassifier(
            n_estimators=300,
            max_depth=d,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.5,
            scale_pos_weight=sp,
            n_jobs=1,
            random_state=seed,
            verbosity=0,
            eval_metric="logloss",
        ),
        [3, 5],
    )
    fit_eval(
        "SVM-RBF (Sel)",
        Xs,
        lambda C: SVC(C=C, kernel="rbf", probability=True, class_weight="balanced", random_state=seed),
        [1.0, 3.0, 10.0],
    )
    fit_eval(
        "LogReg (Sel)",
        Xs,
        lambda C: LogisticRegression(C=C, max_iter=2000, class_weight="balanced"),
        [0.01, 0.1, 1.0],
    )
    return out
