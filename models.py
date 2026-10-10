import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from config import CFG, DEVICE
from data import FP, FP_DIM, GR, NODE_DIM, Y
import data


def collate(ids, with_graph=True):
    ids = list(ids)
    b = {"fp": torch.from_numpy(FP[ids]), "y": torch.tensor(Y[ids], dtype=torch.long)}
    if data.XS is not None:
        b["xs"] = torch.from_numpy(data.XS[ids])
    if with_graph:
        xs, eis, bs, off = [], [], [], 0
        for k, i in enumerate(ids):
            x, ei = GR[i]
            xs.append(x)
            eis.append(ei + off)
            bs.append(torch.full((x.shape[0],), k, dtype=torch.long))
            off += x.shape[0]
        b["x"], b["ei"], b["b"], b["n"] = torch.cat(xs), torch.cat(eis, 1), torch.cat(bs), len(ids)
    return b


def to_dev(b):
    return {k: (v.to(DEVICE) if torch.is_tensor(v) else v) for k, v in b.items()}


def pool(h, batch, n):
    s = torch.zeros(n, h.shape[1], device=h.device).index_add_(0, batch, h)
    c = torch.zeros(n, device=h.device).index_add_(0, batch, torch.ones(h.shape[0], device=h.device))
    return torch.cat([s, s / c.clamp(min=1).unsqueeze(1)], 1)


class GINLayer(nn.Module):
    def __init__(self, i, o):
        super().__init__()
        self.eps = nn.Parameter(torch.zeros(1))
        self.mlp = nn.Sequential(nn.Linear(i, o), nn.ReLU(), nn.Linear(o, o))

    def forward(self, h, ei):
        return self.mlp((1 + self.eps) * h + torch.zeros_like(h).index_add_(0, ei[1], h[ei[0]]))


class GCNLayer(nn.Module):
    def __init__(self, i, o):
        super().__init__()
        self.lin = nn.Linear(i, o)

    def forward(self, h, ei):
        n = h.shape[0]
        loop = torch.arange(n, device=h.device)
        src, dst = torch.cat([ei[0], loop]), torch.cat([ei[1], loop])
        deg = torch.zeros(n, device=h.device).index_add_(0, dst, torch.ones_like(dst, dtype=torch.float))
        w = (deg[src] * deg[dst]).pow(-0.5).unsqueeze(1)
        return self.lin(torch.zeros(n, h.shape[1], device=h.device).index_add_(0, dst, h[src] * w))


class GNN(nn.Module):
    def __init__(
        self,
        kind="gin",
        head="linear",
        hidden=64,
        layers=3,
        dropout=0.2,
        n_exp=3,
        gdim=0,
        gate_mode="soft",
        wide=0,
    ):
        super().__init__()
        L = {"gin": GINLayer, "gcn": GCNLayer}[kind]
        self.inp = nn.Linear(NODE_DIM, hidden)
        self.convs = nn.ModuleList([L(hidden, hidden) for _ in range(layers)])
        self.drop, self.gdim, self.head, self.gate_mode, self.n_exp = dropout, gdim, head, gate_mode, n_exp
        d = 2 * hidden
        if gdim > 0:
            self.gproj = nn.Sequential(nn.Linear(gdim, hidden), nn.ReLU(), nn.Dropout(dropout))
            d += hidden
        self.emb = nn.Sequential(nn.Linear(d, hidden), nn.ReLU(), nn.Dropout(dropout))
        self.shared = nn.Linear(hidden, 2)
        self.expert_mask = None
        if head == "moe":
            self.gate = nn.Linear(hidden, n_exp)
            self.experts = nn.ModuleList(
                [
                    nn.Sequential(nn.Linear(hidden, hidden // 2), nn.ReLU(), nn.Linear(hidden // 2, 2))
                    for _ in range(n_exp)
                ]
            )
            for e in self.experts:
                nn.init.zeros_(e[-1].weight)
                nn.init.zeros_(e[-1].bias)
        if head == "dense":
            self.dense = nn.Sequential(nn.Linear(hidden, wide), nn.ReLU(), nn.Linear(wide, 2))
            nn.init.zeros_(self.dense[-1].weight)
            nn.init.zeros_(self.dense[-1].bias)

    def forward(self, b):
        h = F.relu(self.inp(b["x"]))
        for c in self.convs:
            h = F.dropout(F.relu(c(h, b["ei"])) + h, self.drop, self.training)
        p = pool(h, b["b"], b["n"])
        if self.gdim > 0:
            p = torch.cat([p, self.gproj(b["xs"])], 1)
        z = self.emb(p)
        self.last_z = z.detach()
        out = self.shared(z)
        if self.head == "moe":
            g = F.softmax(self.gate(z), -1)
            if self.gate_mode == "uniform":
                g = torch.full_like(g, 1.0 / self.n_exp)
            elif self.gate_mode == "top1":
                g = g * F.one_hot(g.argmax(-1), self.n_exp).float()
            self.last_gate = g.detach()
            if self.expert_mask is not None:
                g = g * self.expert_mask.to(g.device)
            out = out + (g.unsqueeze(-1) * torch.stack([e(z) for e in self.experts], 1)).sum(1)
        elif self.head == "dense":
            out = out + self.dense(z)
        return out


class FPMLP(nn.Module):
    def __init__(self, hidden=256, dropout=0.3, in_dim=None, key="fp"):
        super().__init__()
        self.key = key
        self.net = nn.Sequential(
            nn.Linear(in_dim or FP_DIM, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, 2),
        )

    def forward(self, b):
        return self.net(b[self.key])


def nparams(m):
    return sum(p.numel() for p in m.parameters())


def matched_width():
    H, L, D, K = CFG["hidden"], CFG["n_layers"], CFG["dropout"], CFG["n_experts"]
    tgt = nparams(GNN("gin", "moe", H, L, D, K, gdim=data.XS_DIM))
    best = min(
        range(4, 400),
        key=lambda w: abs(nparams(GNN("gin", "dense", H, L, D, gdim=data.XS_DIM, wide=w)) - tgt),
    )
    return best


SPEC = {
    "gcn": dict(kind="gcn"),
    "gin": dict(kind="gin"),
    "gin_moe": dict(head="moe"),
    "gin_desc": dict(desc=True),
    "gin_dense": dict(head="dense", desc=True),
    "bee": dict(head="moe", desc=True),
    "bee_k1": dict(head="moe", desc=True, n_exp=1),
    "bee_k2": dict(head="moe", desc=True, n_exp=2),
    "bee_k5": dict(head="moe", desc=True, n_exp=5),
    "bee_top1": dict(head="moe", desc=True, gate_mode="top1"),
    "bee_uni": dict(head="moe", desc=True, gate_mode="uniform"),
}


def build(name):
    H, L, D = CFG["hidden"], CFG["n_layers"], CFG["dropout"]
    if name == "mlp":
        return FPMLP().to(DEVICE)
    if name == "mlp_sel":
        return FPMLP(in_dim=data.XS_DIM, key="xs").to(DEVICE)
    sp = dict(SPEC[name])
    desc = sp.pop("desc", False)
    kw = dict(
        kind=sp.pop("kind", "gin"),
        head=sp.pop("head", "linear"),
        hidden=H,
        layers=L,
        dropout=D,
        n_exp=sp.pop("n_exp", CFG["n_experts"]),
        gdim=data.XS_DIM if desc else 0,
        gate_mode=sp.pop("gate_mode", "soft"),
    )
    if kw["head"] == "dense":
        kw["wide"] = matched_width()
    return GNN(**kw).to(DEVICE)


def needs_graph(name):
    return name not in ("mlp", "mlp_sel")


def class_weights(ids):
    c = np.bincount(Y[ids], minlength=2).astype(float) + 1.0
    return torch.tensor(c.sum() / (2 * c), dtype=torch.float32, device=DEVICE)


@torch.no_grad()
def predict(model, name, ids, bs=256, gates=False):
    model.eval()
    out, gs = [], []
    ids = np.asarray(ids)
    for i in range(0, len(ids), bs):
        b = to_dev(collate(ids[i : i + bs], needs_graph(name)))
        out.append(F.softmax(model(b), -1)[:, 1].cpu().numpy())
        if gates and getattr(model, "head", "") == "moe":
            gs.append(model.last_gate.cpu().numpy())
    p = np.concatenate(out)
    return (p, np.concatenate(gs)) if gates else p
