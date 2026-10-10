from figstyle import *
from rdkit import Chem, DataStructs
from rdkit.Chem import Descriptors, Crippen, rdFingerprintGenerator

mols = [Chem.MolFromSmiles(s) for s in df.SMILES]
df["MW"] = [Descriptors.MolWt(m) for m in mols]
df["logP"] = [Crippen.MolLogP(m) for m in mols]
g = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
fps = [g.GetFingerprint(m) for m in mols]
nn = []
for i, f in enumerate(fps):
    s = np.array(DataStructs.BulkTanimotoSimilarity(f, fps))
    s[i] = -1
    nn.append(s.max())
df["nn"] = nn
fig = plt.figure(figsize=(17, 10.4))
gs = fig.add_gridspec(2, 3, hspace=0.52, wspace=0.52, left=0.07, right=0.99, top=0.94, bottom=0.085)

ax = fig.add_subplot(gs[0, 0])
tag(ax, "a")
srcs = ["ECOTOX", "PPDB", "BPDB"]
x = np.arange(3)
nt = [((df.source == s) & (df.label == 1)).sum() for s in srcs]
nn_ = [((df.source == s) & (df.label == 0)).sum() for s in srcs]
ax.bar(x, nn_, color=COL_TOX[0], width=0.62, label="Non-toxic")
ax.bar(x, nt, bottom=nn_, color=COL_TOX[1], width=0.62, label="Toxic")
for i, (a, b) in enumerate(zip(nn_, nt)):
    ax.text(i, a / 2, str(a), ha="center", va="center", fontsize=17, color="black")
    if b > 40:
        ax.text(i, a + b / 2, str(b), ha="center", va="center", fontsize=17, color="white", fontweight="bold")
    lab = f"{100*b/(a+b):.0f}% toxic" if b > 40 else f"{b}\n(19%)"
    ax.text(i, a + b + 12, lab, ha="center", va="bottom", fontsize=16)
ax.set_xticks(x)
ax.set_xticklabels(srcs)
ax.set_ylabel("Compounds")
ax.set_ylim(0, 610)
ax.legend(frameon=False, loc="upper right", bbox_to_anchor=(1.04, 1.02), handlelength=1.2)

ax = fig.add_subplot(gs[0, 1])
tag(ax, "b", dx=-0.2)
cnt = df.groupby("cls").label.agg(["size", "mean"]).reindex(CLS)
names = ["Insecticide", "Herbicide", "Fungicide", "Other agrochem.", "Non-agrochem."]
y = np.arange(5)[::-1]
ax.barh(y, 100 * cnt["mean"], color=[COL_CLS[c] for c in CLS], height=0.64)
for yi, (n, m) in zip(y, zip(cnt["size"], cnt["mean"])):
    ax.text(
        100 * m + 2,
        yi,
        f"{100*m:.0f}%  (n={n})",
        ha="left",
        va="center",
        fontsize=16,
        bbox=dict(fc="white", ec="none", pad=1.5),
        zorder=5,
    )
ax.axvline(100 * df.label.mean(), color="black", ls="--", lw=1.5)
ax.text(100 * df.label.mean() + 1, 4.62, "overall 29%", ha="left", va="bottom", fontsize=14.5)
ax.set_yticks(y)
ax.set_yticklabels(names)
ax.set_xlabel("Toxic compounds (%)")
ax.set_xlim(0, 128)
ax.set_xticks([0, 25, 50, 75, 100])
ax.set_ylim(-0.6, 5.0)

ax = fig.add_subplot(gs[0, 2])
tag(ax, "c")
bins = np.arange(1830, 2030, 10)
ax.hist(
    [df[df.label == 0].year, df[df.label == 1].year],
    bins=bins,
    stacked=True,
    color=[COL_TOX[0], COL_TOX[1]],
    edgecolor="white",
    lw=0.8,
)
ax.axvline(2000, color="#D1495B", lw=2.2)
ax.text(1948, 190, "time-split test\n(year ≥ 2000)", ha="right", va="top", fontsize=15, color="#D1495B")
ax.set_ylim(0, 200)
ax.set_xlabel("Year of first report")
ax.set_ylabel("Compounds")
ax.set_xlim(1830, 2025)


def box(ax, col, ylabel, clip=None):
    data = [df[df.label == k][col].clip(upper=clip) if clip else df[df.label == k][col] for k in (0, 1)]
    bp = ax.boxplot(
        data,
        widths=0.55,
        patch_artist=True,
        showfliers=False,
        medianprops=dict(color="black", lw=2.2),
        whiskerprops=dict(lw=1.5),
        capprops=dict(lw=1.5),
        boxprops=dict(lw=1.5),
    )
    for p, k in zip(bp["boxes"], (0, 1)):
        p.set_facecolor(COL_TOX[k])
    ax.set_xticks([1, 2])
    ax.set_xticklabels(["Non-toxic", "Toxic"])
    ax.set_ylabel(ylabel)


ax = fig.add_subplot(gs[1, 0])
tag(ax, "d")
box(ax, "MW", "Molecular weight (Da)", clip=800)
ax.set_ylim(0, 620)
from scipy.stats import mannwhitneyu


def sig(ax, col, y):
    p = mannwhitneyu(df[df.label == 0][col], df[df.label == 1][col]).pvalue
    t = "P < 0.001" if p < 0.001 else f"P = {p:.3f}"
    ax.text(1.5, y, t, ha="center", va="bottom", fontsize=15)


sig(ax, "MW", 575)
ax = fig.add_subplot(gs[1, 1])
tag(ax, "e")
box(ax, "logP", "Crippen logP")
ax.set_ylim(-5, 10.5)
sig(ax, "logP", 9.2)

ax = fig.add_subplot(gs[1, 2])
tag(ax, "f")
data = [df[df.source == s].nn for s in srcs]
bp = ax.boxplot(
    data,
    widths=0.55,
    patch_artist=True,
    showfliers=False,
    medianprops=dict(color="black", lw=2.2),
    whiskerprops=dict(lw=1.5),
    capprops=dict(lw=1.5),
    boxprops=dict(lw=1.5),
)
for p, s in zip(bp["boxes"], srcs):
    p.set_facecolor(COL_SRC[s])
for i, s in enumerate(srcs):
    ax.text(i + 1, 1.08, f"{df[df.source==s].nn.median():.2f}", ha="center", fontsize=16, fontweight="bold")
ax.set_xticks([1, 2, 3])
ax.set_xticklabels(srcs)
ax.set_ylabel("Nearest-neighbour\nTanimoto similarity")
ax.set_ylim(0, 1.2)
ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
fig.savefig("fig_data_analysis.pdf")
fig.savefig("fig_data_analysis.png", dpi=200)
print(df.groupby("label")[["MW", "logP"]].median(), df.groupby("source").nn.median())
