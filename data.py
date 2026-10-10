import glob
import os

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import Descriptors
from rdkit.Chem import MACCSkeys
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import torch

from rdkit import RDLogger

from config import CFG

RDLogger.DisableLog("rdApp.*")


def _find(name):
    hits = glob.glob(os.path.join(CFG["data_dir"], "**", name), recursive=True)
    hits = [h for h in hits if "apistox_v4" not in h]
    if not hits:
        raise FileNotFoundError(f"{name} not found under {CFG['data_dir']}")
    return hits[0]


df = pd.read_csv(_find("dataset_final.csv"))
df["mol"] = [Chem.MolFromSmiles(s) for s in df["SMILES"]]
assert df["mol"].notnull().all()
df["key"] = df["name"].astype(str) + "||" + df["SMILES"].astype(str)
KEY2IDX = {k: i for i, k in enumerate(df["key"])}
Y = df["label"].values.astype(int)
SRC = df["source"].values
CLIENTS = sorted(set(SRC))
GROUPS = df[["herbicide", "fungicide", "insecticide", "other_agrochemical"]].values


def official_split(kind):
    tr = pd.read_csv(_find(f"{kind}_train.csv"))
    te = pd.read_csv(_find(f"{kind}_test.csv"))
    f = lambda d: np.array(
        [KEY2IDX[a + "||" + b] for a, b in zip(d["name"].astype(str), d["SMILES"].astype(str))]
    )
    return f(tr), f(te)


def make_split(kind, seed):
    key = np.array([f"{s}_{y}" for s, y in zip(SRC, Y)])
    if kind == "random":
        idx = np.arange(len(df))
        tr, te = train_test_split(idx, test_size=CFG["test_frac"], stratify=key, random_state=seed)
        vf = CFG["val_frac"] / (1 - CFG["test_frac"])
    else:
        tr, te = official_split(kind)
        vf = CFG["val_frac"] * len(df) / len(tr)
    tr, va = train_test_split(tr, test_size=vf, stratify=key[tr], random_state=seed)
    return np.sort(tr), np.sort(va), np.sort(te)


def fp_features(m):
    a = np.zeros(2048, dtype=np.float32)
    for i in AllChem.GetMorganFingerprintAsBitVect(m, 2, nBits=2048).GetOnBits():
        a[i] = 1.0
    b = np.zeros(167, dtype=np.float32)
    for i in MACCSkeys.GenMACCSKeys(m).GetOnBits():
        b[i] = 1.0
    return np.concatenate([a, b])


ATOMS = [
    "C",
    "N",
    "O",
    "S",
    "P",
    "F",
    "Cl",
    "Br",
    "I",
    "B",
    "Si",
    "Se",
    "Na",
    "K",
    "Cu",
    "Zn",
    "Sn",
    "Hg",
    "Fe",
    "As",
]
HYB = [
    Chem.rdchem.HybridizationType.SP,
    Chem.rdchem.HybridizationType.SP2,
    Chem.rdchem.HybridizationType.SP3,
    Chem.rdchem.HybridizationType.SP3D,
    Chem.rdchem.HybridizationType.SP3D2,
]


def onehot(v, choices):
    o = [0.0] * (len(choices) + 1)
    o[choices.index(v) if v in choices else -1] = 1.0
    return o


def atom_feat(a):
    return (
        onehot(a.GetSymbol(), ATOMS)
        + onehot(a.GetDegree(), [0, 1, 2, 3, 4, 5])
        + onehot(a.GetFormalCharge(), [-1, 0, 1])
        + onehot(a.GetHybridization(), HYB)
        + onehot(a.GetTotalNumHs(), [0, 1, 2, 3, 4])
        + [float(a.GetIsAromatic()), float(a.IsInRing()), a.GetMass() / 100.0]
    )


def mol_graph(m):
    x = torch.tensor([atom_feat(a) for a in m.GetAtoms()], dtype=torch.float32)
    src, dst = [], []
    for b in m.GetBonds():
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        src += [i, j]
        dst += [j, i]
    ei = torch.tensor([src, dst], dtype=torch.long) if src else torch.zeros((2, 0), dtype=torch.long)
    return x, ei


def desc_features(m):
    out = []
    for _, f in Descriptors.descList:
        try:
            out.append(float(f(m)))
        except Exception:
            out.append(np.nan)
    return out


FP = np.stack([fp_features(m) for m in df["mol"]])
GR = [mol_graph(m) for m in df["mol"]]
NODE_DIM, FP_DIM, N_FP = GR[0][0].shape[1], FP.shape[1], FP.shape[1]
DESC = np.array([desc_features(m) for m in df["mol"]], dtype=np.float64)
ALL_RAW = np.hstack([FP.astype(np.float64), DESC])
DESC_NAMES = [n for n, _ in Descriptors.descList]
FEAT_NAMES = [f"morgan_{i}" for i in range(2048)] + [f"maccs_{i}" for i in range(167)] + DESC_NAMES
XS, XS_DIM, XS_FULL = None, 1, None
