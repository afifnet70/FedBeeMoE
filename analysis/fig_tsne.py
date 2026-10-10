from figstyle import *
from rdkit import Chem, DataStructs
from rdkit.Chem import Descriptors, rdFingerprintGenerator
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from matplotlib.lines import Line2D

mols = [Chem.MolFromSmiles(s) for s in df.SMILES]
g = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
fps = [g.GetFingerprint(m) for m in mols]
n = len(fps)
Dm = np.zeros((n, n))
for i, f in enumerate(fps):
    Dm[i] = 1 - np.array(DataStructs.BulkTanimotoSimilarity(f, fps))
np.fill_diagonal(Dm, 0)
import os

COL_TOX = {0: "#18C0E0", 1: "#7A12D6"}
COL_SRC = {"ECOTOX": "#0A6CFF", "PPDB": "#FF8A00", "BPDB": "#00BA6C"}
COL_CLS = {
    "Insecticide": "#FF2D55",
    "Herbicide": "#2EC24A",
    "Fungicide": "#C21FE8",
    "Other agrochemical": "#FFC400",
    "Not agrochemical": "#6E6E6E",
}
E1 = np.load("tsne_fp.npy")
E2 = np.load("tsne_desc.npy")
SHORTC = {
    "Insecticide": "Insecticide",
    "Herbicide": "Herbicide",
    "Fungicide": "Fungicide",
    "Other agrochemical": "Other agro.",
    "Not agrochemical": "Non-agro.",
}
fig = plt.figure(figsize=(17, 14.2))
gs = fig.add_gridspec(2, 3, hspace=0.24, wspace=0.10, left=0.055, right=0.995, top=0.925, bottom=0.19)
rows = [("Morgan fingerprint, Jaccard distance", E1), ("RDKit 2D descriptors, PCA-50", E2)]
cols = [
    ("Toxicity", lambda: df.label.map(COL_TOX)),
    ("Source database", lambda: df.source.map(COL_SRC)),
    ("Agrochemical class", lambda: df.cls.map(COL_CLS)),
]
H = lambda col, lab: Line2D(
    [0], [0], marker="o", ls="", markerfacecolor=col, markeredgecolor="white", markersize=22, label=lab
)
LEG = [
    ([H(COL_TOX[0], "Non-toxic (739)"), H(COL_TOX[1], "Toxic (296)")], 2),
    ([H(COL_SRC[s], f"{s} ({(df.source==s).sum()})") for s in COL_SRC], 2),
    ([H(COL_CLS[c], f"{SHORTC[c]} ({(df.cls==c).sum()})") for c in CLS], 2),
]
letters = "abcdef"
k = 0
for r, (rn, E) in enumerate(rows):
    for c, (cn, cf) in enumerate(cols):
        ax = fig.add_subplot(gs[r, c])
        ax.set_xticks([])
        ax.set_yticks([])
        for nm, sp in ax.spines.items():
            sp.set_visible(nm in ("left", "bottom"))
            sp.set_linewidth(2.0)
        ax.margins(0.04)
        colr = cf().values
        order = np.argsort(df.label.values) if c == 0 else np.arange(n)
        ax.scatter(
            E[order, 0], E[order, 1], c=colr[order], s=230, alpha=0.92, edgecolors="white", linewidths=0.9
        )
        ax.text(
            0.0,
            1.02,
            letters[k],
            transform=ax.transAxes,
            fontsize=30,
            fontweight="bold",
            va="bottom",
            ha="left",
        )
        k += 1
        if c == 0:
            ax.text(0.095, 1.025, rn, transform=ax.transAxes, fontsize=18, va="bottom", ha="left")
        if r == 0:
            ax.set_title(cn, fontsize=26, pad=52, fontweight="bold")
        if c == 0:
            ax.set_ylabel("t-SNE 2", fontsize=22)
        ax.set_xlabel("t-SNE 1", fontsize=22, labelpad=6)
        if r == 1:
            h, nc = LEG[c]
            ax.legend(
                handles=h,
                loc="upper center",
                bbox_to_anchor=((0.43 if c == 2 else 0.5), -0.12),
                ncol=nc,
                frameon=False,
                fontsize=19,
                handletextpad=0.1,
                columnspacing=0.7,
                labelspacing=0.5,
            )
fig.savefig("fig_tsne_grouped.pdf")
fig.savefig("fig_tsne_grouped.png", dpi=200)
