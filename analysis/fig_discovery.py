from figstyle import *
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors
from scipy.stats import mannwhitneyu

RDLogger.DisableLog("rdApp.*")
ms = [Chem.MolFromSmiles(s) for s in df.SMILES]
ok = [m is not None for m in ms]
d = df[ok].reset_index(drop=True)
ms = [m for m in ms if m is not None]
names = [n for n, _ in Descriptors._descList]
X = np.array([[f(m) for _, f in Descriptors._descList] for m in ms], float)
y = d.label.values
rows = []
for j, n in enumerate(names):
    v = X[:, j]
    if not np.isfinite(v).all() or np.std(v) == 0:
        continue
    a, b = v[y == 1], v[y == 0]
    U, p = mannwhitneyu(a, b, alternative="two-sided")
    rows.append((n, 2 * U / (len(a) * len(b)) - 1, p))
r = pd.DataFrame(rows, columns=["d", "delta", "p"])
o = np.argsort(r.p.values)
m = len(r)
q = np.empty(m)
pv = r.p.values[o] * m / np.arange(1, m + 1)
q[o] = np.minimum.accumulate(pv[::-1])[::-1]
r["q"] = np.minimum(q, 1)
r["nl"] = -np.log10(np.maximum(r.q, 1e-300))
r.to_csv("discovery_desc.csv", index=False)
print(len(r), (r.q < 0.05).sum())
print(r.sort_values("q").head(12))
PUR, TEA, GRY = "#6A1B9A", "#00A6A6", "#B0B0B0"
fig, ax = plt.subplots(2, 2, figsize=(21, 17))
fig.subplots_adjust(wspace=0.38, hspace=0.36, left=0.11, right=0.93, top=0.94, bottom=0.075)

A = ax[0, 0]
sig = (r.q < 0.05) & (r.delta.abs() >= 0.2)
A.scatter(r.delta[~sig], r.nl[~sig], s=120, c=GRY, edgecolor="white", lw=1)
A.scatter(r.delta[sig & (r.delta > 0)], r.nl[sig & (r.delta > 0)], s=190, c=PUR, edgecolor="white", lw=1.2)
A.scatter(r.delta[sig & (r.delta < 0)], r.nl[sig & (r.delta < 0)], s=190, c=TEA, edgecolor="white", lw=1.2)
A.axhline(-np.log10(0.05), ls="--", c="k", lw=1.3)
A.axvline(0.2, ls=":", c="k", lw=1.3)
A.axvline(-0.2, ls=":", c="k", lw=1.3)
top = r[sig].sort_values("nl", ascending=False).head(5)
offs = [(-150, -4), (-150, -16), (-130, -40), (-120, -72), (-40, -95)]
for (i, t), (dx, dy) in zip(top.iterrows(), offs):
    A.annotate(
        t.d,
        (t.delta, t.nl),
        xytext=(dx, dy),
        textcoords="offset points",
        fontsize=18,
        fontweight="bold",
        va="center",
        arrowprops=dict(arrowstyle="-", lw=1.3, color="k"),
    )
A.set_xlabel("Effect size (Cliff's $\\delta$, toxic vs non-toxic)")
A.set_ylabel("$-\\log_{10}$ FDR $q$-value")
A.set_title("Descriptor volcano plot", fontweight="bold", loc="left")
tag(A, "a", dx=-0.2)
A.text(
    0.02,
    0.96,
    f"{int(sig.sum())} of {len(r)} descriptors\nsignificant, $|\\delta|\\geq0.2$",
    transform=A.transAxes,
    ha="left",
    va="top",
    fontsize=17,
)

B = ax[0, 1]
t8 = r.reindex(r.delta.abs().sort_values(ascending=False).index).head(8).iloc[::-1]
B.barh(t8.d, t8.delta, color=[PUR if v > 0 else TEA for v in t8.delta], height=0.7)
B.axvline(0, c="k", lw=1.5)
B.set_xlabel("Cliff's $\\delta$")
B.set_xlim(-0.1, 0.45)
B.set_title("Largest class differences", fontweight="bold", loc="left")
tag(B, "b", dx=-0.28)

C = ax[1, 0]
sm = {
    "Phosphorus": "[#15]",
    "Carbamate": "[NX3]C(=O)O[#6]",
    "Sulfur": "[#16]",
    "Halogen": "[F,Cl,Br,I]",
    "Nitro": "[N+](=O)[O-]",
    "Aromatic N": "[n]",
    "Ester": "[#6]C(=O)O[#6]",
    "Carboxylic acid": "C(=O)[OH]",
}
res = []
for k, s in sm.items():
    p = Chem.MolFromSmarts(s)
    h = np.array([m.HasSubstructMatch(p) for m in ms])
    if h.sum() >= 15:
        res.append((k, y[h].mean(), h.sum(), y[~h].mean()))
res = sorted(res, key=lambda z: z[1])
C.barh(
    [z[0] for z in res], [z[1] for z in res], color=[PUR if z[1] > y.mean() else TEA for z in res], height=0.7
)
for i, z in enumerate(res):
    C.text(max(z[1], y.mean()) + 0.05, i, f"n={z[2]}", va="center", fontsize=16)
C.axvline(y.mean(), ls="--", c="k", lw=1.4)
C.text(y.mean() + 0.01, len(res) - 0.45, "all: 0.29", fontsize=16)
C.set_xlabel("Toxic fraction")
C.set_xlim(0, max(z[1] for z in res) + 0.2)
C.set_title("Toxic fraction by substructure", fontweight="bold", loc="left")
tag(C, "c", dx=-0.3)

Dd = ax[1, 1]
mw = d.SMILES.map(lambda s: Descriptors.MolWt(Chem.MolFromSmiles(s))).values
lp = X[:, names.index("MolLogP")]
bx = [0, 200, 300, 400, 2000]
by = [-20, 1, 3, 5, 30]
H = np.full((4, 4), np.nan)
N = np.zeros((4, 4), int)
for i in range(4):
    for j in range(4):
        k = (lp >= by[i]) & (lp < by[i + 1]) & (mw >= bx[j]) & (mw < bx[j + 1])
        N[i, j] = k.sum()
        if k.sum() >= 8:
            H[i, j] = y[k].mean()
im = Dd.imshow(H, origin="lower", cmap="Purples", vmin=0, vmax=0.7, aspect="auto")
for i in range(4):
    for j in range(4):
        if N[i, j]:
            Dd.text(
                j,
                i,
                f"{H[i,j]:.2f}\n(n={N[i,j]})" if not np.isnan(H[i, j]) else f"n={N[i,j]}",
                ha="center",
                va="center",
                fontsize=16,
                fontweight="bold",
                color="k",
            )
Dd.set_xticks(range(4))
Dd.set_xticklabels(["<200", "200-300", "300-400", ">400"])
Dd.set_yticks(range(4))
Dd.set_yticklabels(["<1", "1-3", "3-5", ">5"])
Dd.set_xlabel("Molecular weight (Da)")
Dd.set_ylabel("Crippen logP")
for s in Dd.spines.values():
    s.set_visible(False)
cb = fig.colorbar(im, ax=Dd, fraction=0.045, pad=0.04)
cb.set_label("Toxic fraction")
cb.outline.set_visible(False)
Dd.set_title("Toxic fraction by size and lipophilicity", fontweight="bold", loc="left")
tag(Dd, "d", dx=-0.2)
fig.savefig("fig_discovery.pdf")
fig.savefig("fig_discovery.png", dpi=110)
