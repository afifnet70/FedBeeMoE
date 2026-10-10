import copy

import numpy as np
import torch
import torch.nn.functional as F

from config import CFG
from data import CLIENTS, SRC, Y
from metrics import NB, client_hist, compute_metrics, select_from_hists
from models import build, class_weights, collate, needs_graph, predict, to_dev


def agg_val(model, name, va):
    pv = predict(model, name, va)
    hp = np.zeros(NB)
    hn = np.zeros(NB)
    for c in CLIENTS:
        m = SRC[va] == c
        if m.sum() == 0:
            continue
        a, b = client_hist(Y[va][m], pv[m])
        hp += a
        hn += b
    return select_from_hists(hp, hn)


def pooled_val(model, name, va):
    return select_from_hists(*[x.astype(float) for x in client_hist(Y[va], predict(model, name, va))])


def local_train(model, name, ids, epochs, lr, opt_kind="adam", mu=0.0, glob=None, c_glob=None, c_loc=None):
    model.train()
    ids = np.array(ids)
    cw = class_weights(ids)
    params = list(model.parameters())
    opt = (
        torch.optim.Adam(params, lr=lr, weight_decay=1e-5)
        if opt_kind == "adam"
        else torch.optim.SGD(params, lr=lr)
    )
    steps, tot, cnt = 0, 0.0, 0
    for _ in range(epochs):
        perm = np.random.permutation(ids)
        for i in range(0, len(perm), CFG["batch_size"]):
            ch = perm[i : i + CFG["batch_size"]]
            if len(ch) < 2:
                continue
            b = to_dev(collate(ch, needs_graph(name)))
            loss = F.cross_entropy(model(b), b["y"], weight=cw)
            tot += float(loss.detach())
            cnt += 1
            if mu > 0 and glob is not None:
                loss = loss + 0.5 * mu * sum(((p - g) ** 2).sum() for p, g in zip(params, glob))
            opt.zero_grad()
            loss.backward()
            if c_glob is not None:
                for p, cg, cl in zip(params, c_glob, c_loc):
                    p.grad.add_(cg - cl)
            torch.nn.utils.clip_grad_norm_(params, 5.0)
            opt.step()
            steps += 1
    return steps, tot / max(cnt, 1)


def get_vec(m):
    return [p.detach().clone() for p in m.parameters()]


def set_vec(m, v):
    with torch.no_grad():
        for p, x in zip(m.parameters(), v):
            p.copy_(x)


def per_client_metrics(te, prob, thr):
    out = {}
    for c in CLIENTS:
        m = SRC[te] == c
        if m.sum() >= 1:
            t = thr[m] if np.ndim(thr) else thr
            r = compute_metrics(Y[te][m], prob[m], t)
            r["n"] = int(m.sum())
            r["pos"] = int(Y[te][m].sum())
            out[c] = r
    return out


def run_federated(algo, name, tr, va, te, hp=None):
    hp = hp or CFG["hp"]
    cids = {c: tr[SRC[tr] == c] for c in CLIENTS if (SRC[tr] == c).sum() >= 2}
    clients = list(cids)
    n_k = np.array([len(cids[c]) for c in clients], float)
    wts = n_k / n_k.sum()
    gm, work = build(name), build(name)
    gvec = get_vec(gm)
    m_s = [torch.zeros_like(p) for p in gvec]
    v_s = [torch.zeros_like(p) for p in gvec]
    c_glob = [torch.zeros_like(p) for p in gvec]
    c_loc = {c: [torch.zeros_like(p) for p in gvec] for c in clients}
    best, best_vec, best_sel, hist = -9, copy.deepcopy(gvec), None, []
    for rnd in range(1, CFG["rounds"] + 1):
        deltas, dcs, losses = [], [], []
        for c in clients:
            set_vec(work, gvec)
            if algo == "scaffold":
                st, ls = local_train(
                    work,
                    name,
                    cids[c],
                    CFG["local_epochs"],
                    hp["scaf_lr"],
                    "sgd",
                    c_glob=c_glob,
                    c_loc=c_loc[c],
                )
                y = get_vec(work)
                K = max(st, 1)
                nc = [
                    cl - cg + (g - yy) / (K * hp["scaf_lr"])
                    for cl, cg, g, yy in zip(c_loc[c], c_glob, gvec, y)
                ]
                dcs.append([a - b for a, b in zip(nc, c_loc[c])])
                c_loc[c] = nc
            else:
                _, ls = local_train(
                    work,
                    name,
                    cids[c],
                    CFG["local_epochs"],
                    CFG["fl_lr"],
                    "adam",
                    mu=hp["mu"] if algo == "fedprox" else 0.0,
                    glob=[g.clone() for g in gvec] if algo == "fedprox" else None,
                )
                y = get_vec(work)
            losses.append(ls)
            deltas.append([a - g for a, g in zip(y, gvec)])
        avg = [sum(w * d[i] for w, d in zip(wts, deltas)) for i in range(len(gvec))]
        if algo == "fedadam":
            for i in range(len(gvec)):
                m_s[i] = 0.9 * m_s[i] + 0.1 * avg[i]
                v_s[i] = 0.99 * v_s[i] + 0.01 * avg[i] ** 2
                gvec[i] = gvec[i] + hp["adam_lr"] * m_s[i] / (v_s[i].sqrt() + 1e-3)
        else:
            gvec = [g + a for g, a in zip(gvec, avg)]
        if algo == "scaffold":
            c_glob = [cg + sum(dc[i] for dc in dcs) / len(clients) for i, cg in enumerate(c_glob)]
        set_vec(gm, gvec)
        sel = agg_val(gm, name, va)
        hist.append(
            dict(
                round=rnd,
                val_score=sel["score"],
                val_mcc=sel["mcc"],
                val_auc=sel["auc"],
                train_loss=float(np.dot(wts, losses)),
            )
        )
        if sel["score"] > best:
            best, best_vec, best_sel = sel["score"], copy.deepcopy(gvec), sel
    set_vec(gm, best_vec)
    return gm, best_sel, hist


def finish(gm, name, sel, te, extra_gate=False):
    pt = predict(gm, name, te, gates=extra_gate)
    gate = None
    if extra_gate:
        pt, gate = pt
    return pt, gate
