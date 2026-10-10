from sklearn.ensemble import RandomForestClassifier
import numpy as np
import pandas as pd

from config import CFG
from data import ALL_RAW, CLIENTS, FEAT_NAMES, N_FP, SRC, Y
import data


def fed_prepare(tr, k=None):

    k = k or CFG["sel_k"]
    X = ALL_RAW.copy()
    X[~np.isfinite(X)] = np.nan
    X = np.clip(X, -1e9, 1e9)
    X[:, N_FP:] = np.sign(X[:, N_FP:]) * np.log1p(np.abs(X[:, N_FP:]))
    cl = {c: tr[SRC[tr] == c] for c in CLIENTS if (SRC[tr] == c).sum() > 0}

    cnt = sum((~np.isnan(X[g])).sum(0) for g in cl.values())
    s1 = sum(np.nansum(X[g], 0) for g in cl.values())
    mu0 = np.where(cnt > 0, s1 / np.maximum(cnt, 1), 0.0)
    Xi = np.where(np.isnan(X), mu0, X)

    n = sum(len(g) for g in cl.values())
    s1 = sum(Xi[g].sum(0) for g in cl.values())
    s2 = sum((Xi[g] ** 2).sum(0) for g in cl.values())
    mu = s1 / n
    var = np.maximum(s2 / n - mu**2, 0)
    sd = np.sqrt(var)
    binary = np.arange(ALL_RAW.shape[1]) < N_FP
    keep = np.where(binary, (mu > 0.01) & (mu < 0.99), sd > 1e-6)
    idx = np.where(keep)[0]

    G = sum(Xi[g][:, idx].T @ Xi[g][:, idx] for g in cl.values())
    cov = G / n - np.outer(mu[idx], mu[idx])
    d = np.sqrt(np.maximum(np.diag(cov), 1e-12))
    C = np.abs(cov / np.outer(d, d))
    np.fill_diagonal(C, 0)
    drop = set()
    for i in range(len(idx)):
        if i in drop:
            continue
        drop.update((np.where(C[i, i + 1 :] > CFG["corr_thr"])[0] + i + 1).tolist())
    idx = np.array([j for q, j in enumerate(idx) if q not in drop])
    Z = np.clip((Xi - mu) / np.where(sd < 1e-6, 1, sd), -5, 5).astype(np.float32)

    imp, tot = np.zeros(len(idx)), 0
    for g in cl.values():
        if len(g) < 20 or len(set(Y[g])) < 2:
            continue
        rf = RandomForestClassifier(
            150, class_weight="balanced", n_jobs=1, random_state=0, min_samples_leaf=2
        ).fit(Z[g][:, idx], Y[g])
        imp += len(g) * rf.feature_importances_
        tot += len(g)
    imp /= max(tot, 1)
    order = idx[np.argsort(-imp)]
    sel = order[:k]
    data.XS_FULL, data.XS, data.XS_DIM = Z[:, idx], Z[:, sel], len(sel)
    return dict(
        funnel=[ALL_RAW.shape[1], int(keep.sum()), len(idx), len(sel)],
        kinds=pd.Series([("Morgan" if j < 2048 else "MACCS" if j < N_FP else "Descriptor") for j in sel])
        .value_counts()
        .to_dict(),
        top=[(FEAT_NAMES[j], float(imp[np.where(idx == j)[0][0]])) for j in order[:15]],
    )
