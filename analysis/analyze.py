import os, glob, json, pickle, itertools, warnings
import numpy as np, pandas as pd
from scipy import stats
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from sklearn.metrics import precision_recall_curve, roc_curve

warnings.filterwarnings("ignore")

OUT = os.environ.get("APIS_OUT", "/content/apistox_v4")
DATA = os.environ.get("APIS_DATA", "/content")
REP = os.path.join(OUT, "report")
TAB = os.path.join(REP, "tables")
FIG = os.path.join(REP, "figs")
for d in (TAB, FIG):
    os.makedirs(d, exist_ok=True)
SPLITS = ["random", "maxmin", "time"]
SPLIT_NAME = {"random": "Random", "maxmin": "MaxMin", "time": "Time"}
MET = ["Accuracy", "Precision", "Recall", "F1", "Weighted_F1", "AUC", "MCC", "CSI", "Kappa", "Specificity"]
MET_SHORT = dict(
    Accuracy="Acc",
    Precision="Prec",
    Recall="Rec",
    F1="F1",
    Weighted_F1="wF1",
    AUC="AUC",
    MCC="MCC",
    CSI="CSI",
    Kappa="Kappa",
    Specificity="Spec",
    PR_AUC="PR-AUC",
)
P = dict(
    ink="#0E0A14",
    p1="#2A1250",
    p2="#4B2A85",
    p3="#6B46B0",
    p4="#8E6FD0",
    p5="#B7A1E4",
    p6="#DACCF2",
    p7="#F1EBFA",
    gray="#8C8794",
)
CM = LinearSegmentedColormap.from_list("pp", ["#FBF9FE", "#D9CCF2", "#9C7DDA", "#5B34A0", "#0E0A14"])
plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 16,
        "axes.labelsize": 18,
        "xtick.labelsize": 15,
        "ytick.labelsize": 15,
        "legend.fontsize": 14,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 1.6,
        "pdf.fonttype": 42,
    }
)


def save(fig, n):
    for e in ("png", "pdf"):
        fig.savefig(os.path.join(FIG, f"{n}.{e}"), bbox_inches="tight", dpi=300)
    plt.close(fig)


rows = pd.concat(
    [pd.read_csv(f) for f in sorted(glob.glob(os.path.join(OUT, "rows", "*.csv")))], ignore_index=True
)
cl = pd.concat(
    [pd.read_csv(f) for f in sorted(glob.glob(os.path.join(OUT, "clients", "*.csv")))], ignore_index=True
)
PROB, EXTRA = {}, {}
for f in glob.glob(os.path.join(OUT, "probs", "*.pkl")):
    PROB[os.path.basename(f)[:-4]] = pickle.load(open(f, "rb"))
for f in glob.glob(os.path.join(OUT, "extra", "*.pkl")):
    EXTRA[os.path.basename(f)[:-4]] = pickle.load(open(f, "rb"))
df = pd.read_csv(glob.glob(os.path.join(DATA, "**", "dataset_final.csv"), recursive=True)[0])
SRC = df["source"].values
Y = df["label"].values
ABBR = {
    "Random Forest": "RF",
    "Extra Trees": "ET",
    "Gradient Boosting": "GB",
    "XGBoost": "XGB",
    "LogReg": "LR",
    "SVM-RBF": "SVM",
    "MLP-sklearn": "MLP-sk",
}


def abbr(n):
    for a, b in ABBR.items():
        n = n.replace(a, b)
    return n


rows["Model"] = rows["Model"].map(abbr)
cl["Model"] = cl["Model"].map(abbr)
for k in list(PROB):
    PROB[k]["probs"] = {abbr(m): v for m, v in PROB[k]["probs"].items()}
print("runs per split:", rows.groupby("Split")["Seed"].nunique().to_dict())


bee = rows[(rows.Split == "random") & rows.Model.str.startswith("FedBeeMoE [")]
val_pick = bee.groupby("Model")["Val_Score"].mean().sort_values(ascending=False)
PROPOSED_RAW = val_pick.index[0]
print("validation score of the aggregation variants:\n", val_pick.round(4).to_string())
print("-> proposed =", PROPOSED_RAW)
AGG_OF = {"FedBeeMoE [FedAvg]": "FedAvg", "FedBeeMoE [FedProx]": "FedProx", "FedBeeMoE [FedAdam]": "FedAdam"}
PROP = "FedBeeMoE"
SHORT = {"FedAvg-GIN+Desc (dense head, param-matched)": "FedAvg-GIN+Desc (dense)"}
rows["Model"] = rows["Model"].replace({PROPOSED_RAW: PROP})
cl["Model"] = cl["Model"].replace({PROPOSED_RAW: PROP})
for k in list(PROB):
    PROB[k]["probs"] = {(PROP if m == PROPOSED_RAW else m): v for m, v in PROB[k]["probs"].items()}
for k in list(EXTRA):
    for key in ("gate", "knockout", "hist", "nparam"):
        if key in EXTRA[k]:
            EXTRA[k][key] = {
                (PROP if m == PROPOSED_RAW else SHORT.get(m, m)): v for m, v in EXTRA[k][key].items()
            }

rows["Model"] = rows["Model"].replace(SHORT)
cl["Model"] = cl["Model"].replace(SHORT)
DENSE = "FedAvg-GIN+Desc (dense)"
ABL = ["FedBeeMoE K=1", "FedBeeMoE K=2", "FedBeeMoE K=5", "FedBeeMoE top-1 gate", "FedBeeMoE uniform gate"]
AGGV = [m for m in ("FedBeeMoE [FedAvg]", "FedBeeMoE [FedProx]", "FedBeeMoE [FedAdam]") if m != PROPOSED_RAW]
MAIN_ROWS = rows[~rows.Model.isin(ABL + AGGV)]
GROUP_ORDER = [
    "Federated (graph)",
    "Federated (fingerprint MLP)",
    "Local-only",
    "Pooled (neural)",
    "Pooled (classical ML)",
]


def fmt(m, s, d=3):
    return f"{m:.{d}f}$\\pm${s:.{d}f}"


def stat_table(data, mets, split, bold=PROP):
    d = data[data.Split == split]
    g = d.groupby(["Group", "Model"], sort=False)[mets]
    mu, sd = g.mean(), g.std().fillna(0)
    out = []
    for grp in GROUP_ORDER:
        for gg, mm in mu.index:
            if gg != grp:
                continue
            cells = [fmt(mu.loc[(gg, mm), m], sd.loc[(gg, mm), m]) for m in mets]
            name = mm.replace("&", "\\&").replace("_", "\\_")
            if mm == bold:
                name, cells = f"\\textbf{{{name} (proposed)}}", [f"\\textbf{{{c}}}" for c in cells]
            out.append((grp, name, cells))
    return out, mu


def write_tabular(path, header, body, align=None, group_col=True, first="Model"):
    ncol = len(header) + 1
    align = align or "l" + "c" * (len(header))
    L = [
        f"\\begin{{tabular}}{{{align}}}",
        "\\toprule",
        first + " & " + " & ".join(header) + " \\\\",
        "\\midrule",
    ]
    last = None
    for grp, name, cells in body:
        if group_col and grp != last:
            if last is not None:
                L.append("\\midrule")
            L.append(f"\\multicolumn{{{ncol}}}{{l}}{{\\textit{{{grp}}}}} \\\\")
            last = grp
        L.append(name + " & " + " & ".join(cells) + " \\\\")
    L += ["\\bottomrule", "\\end{tabular}"]
    open(path, "w").write("\n".join(L))


SUMMARY = {
    "proposed_raw": PROPOSED_RAW,
    "val_scores": val_pick.round(4).to_dict(),
    "n_runs": rows.groupby("Split")["Seed"].nunique().to_dict(),
}


for sp in SPLITS:
    if sp not in rows.Split.values:
        continue
    NONPOOL = MAIN_ROWS[~MAIN_ROWS.Group.str.startswith("Pooled")]
    body, _ = stat_table(NONPOOL, MET + ["PR_AUC"], sp)
    write_tabular(
        os.path.join(TAB, f"main_{sp}.tex"),
        [MET_SHORT[m] for m in MET + ["PR_AUC"]],
        body,
        align="l" + "c" * 11,
    )
    REF = MAIN_ROWS[MAIN_ROWS.Group.str.startswith("Pooled")]
    REF = pd.concat(
        [MAIN_ROWS[MAIN_ROWS.Model == PROP].assign(Group="Proposed, federated (for comparison)"), REF],
        ignore_index=True,
    )
    (
        GROUP_ORDER.insert(0, "Proposed, federated (for comparison)")
        if "Proposed, federated (for comparison)" not in GROUP_ORDER
        else None
    )
    body, _ = stat_table(REF, MET + ["PR_AUC"], sp)
    write_tabular(
        os.path.join(TAB, f"ref_{sp}.tex"),
        [MET_SHORT[m] for m in MET + ["PR_AUC"]],
        body,
        align="l" + "c" * 11,
    )
    body, mu = stat_table(MAIN_ROWS, MET + ["PR_AUC"], sp)

    mu = mu.droplevel(0)
    ref = mu.loc[PROP]
    cnt = {
        m: int(sum(mu.loc[x, m] < ref[m] - 1e-12 for x in mu.index if x != PROP)) for m in MET + ["PR_AUC"]
    }
    SUMMARY.setdefault("n_lower", {})[sp] = dict(total_other=len(mu) - 1, **cnt)
    SUMMARY.setdefault("proposed_means", {})[sp] = {m: float(ref[m]) for m in MET + ["PR_AUC"]}
    SUMMARY.setdefault("rank_mcc", {})[sp] = [
        (i, float(v)) for i, v in mu["MCC"].sort_values(ascending=False).round(4).items()
    ]


key_models = [
    PROP,
    "FedAvg-GIN-MoE",
    "FedAvg-GIN+Desc",
    DENSE,
    "FedAvg-GIN",
    "FedAvg-MLP (Sel)",
    "Local-GIN",
    "Pooled-GIN-MoE+Desc",
    "RF (FP)",
    "XGB (FP)",
    "SVM (FP)",
]
body = []
for m in key_models:
    cells = []
    for sp in SPLITS:
        d = MAIN_ROWS[(MAIN_ROWS.Split == sp) & (MAIN_ROWS.Model == m)]
        if len(d) == 0:
            cells += ["--"] * 3
            continue
        cells += [fmt(d[x].mean(), d[x].std(), 3) for x in ("Accuracy", "AUC", "MCC")]
    nm = m.replace("&", "\\&")
    if m == PROP:
        nm, cells = f"\\textbf{{{nm}}}", [f"\\textbf{{{c}}}" for c in cells]
    body.append(("", nm, cells))
hdr = [f"{SPLIT_NAME[s]} {MET_SHORT[x]}" for s in SPLITS for x in ("Accuracy", "AUC", "MCC")]
write_tabular(os.path.join(TAB, "split_ablation.tex"), hdr, body, align="l" + "c" * 9, group_col=False)


def holm(p):
    p = np.asarray(p)
    o = np.argsort(p)
    m = len(p)
    adj = np.empty(m)
    run = 0
    for r, i in enumerate(o):
        run = max(run, (m - r) * p[i])
        adj[i] = min(run, 1)
    return adj


def paired(split, metric, others):
    base = MAIN_ROWS[(MAIN_ROWS.Split == split) & (MAIN_ROWS.Model == PROP)].set_index("Seed")[metric]
    res = []
    for o in others:
        y = MAIN_ROWS[(MAIN_ROWS.Split == split) & (MAIN_ROWS.Model == o)].set_index("Seed")[metric]
        s = base.index.intersection(y.index)
        if len(s) < 3:
            continue
        d = (base.loc[s] - y.loc[s]).values
        n = len(d)
        se = d.std(ddof=1) / np.sqrt(n)
        t = stats.t.ppf(0.975, n - 1)
        try:
            pw = stats.wilcoxon(d).pvalue if np.any(d != 0) else 1.0
        except Exception:
            pw = 1.0
        pt = stats.ttest_1samp(d, 0).pvalue if d.std(ddof=1) > 0 else 1.0
        res.append(
            dict(
                Model=o,
                n=n,
                diff=d.mean(),
                lo=d.mean() - t * se,
                hi=d.mean() + t * se,
                p_wilcoxon=pw,
                p_t=pt,
                dz=d.mean() / d.std(ddof=1) if d.std(ddof=1) > 0 else np.nan,
            )
        )
    r = pd.DataFrame(res)
    if len(r):
        r["p_holm"] = holm(r["p_t"].values)
    return r


stat_models = [m for m in MAIN_ROWS.Model.unique() if m != PROP]
STATS = {}
for sp in SPLITS:
    if sp not in rows.Split.values:
        continue
    for met in ("MCC", "AUC", "Accuracy"):
        STATS[(sp, met)] = paired(sp, met, stat_models)
    s = STATS[(sp, "MCC")]
    if len(s) == 0:
        continue
    body = []
    for _, r in s.iterrows():
        sig = lambda p: ("{\\bf " + f"{p:.3f}" + "}") if p < 0.05 else f"{p:.3f}"
        body.append(
            (
                "",
                r.Model.replace("&", "\\&"),
                [
                    f"{r['diff']:+.3f}",
                    f"[{r['lo']:+.3f}, {r['hi']:+.3f}]",
                    sig(r.p_wilcoxon),
                    sig(r.p_holm),
                    f"{r['dz']:+.2f}",
                ],
            )
        )
    write_tabular(
        os.path.join(TAB, f"stats_{sp}.tex"),
        ["$\\Delta$MCC", "95\\% CI", "$p_{W}$", "$p_{t}^{\\mathrm{Holm}}$", "$d_z$"],
        body,
        align="lccccc",
        group_col=False,
    )
    SUMMARY.setdefault("stats_mcc", {})[sp] = s.round(4).to_dict("records")
    SUMMARY.setdefault("stats_acc", {})[sp] = STATS[(sp, "Accuracy")].round(4).to_dict("records")


def abl_rows(split):
    cfg = [
        ("No expert head (GIN+Desc)", "FedAvg-GIN+Desc"),
        ("Dense residual head, parameter-matched", DENSE),
        ("$K=1$ expert", "FedBeeMoE K=1"),
        ("$K=2$ experts", "FedBeeMoE K=2"),
        (
            "$K=3$ experts, soft gate (FedAvg)",
            PROP if PROPOSED_RAW == "FedBeeMoE [FedAvg]" else "FedBeeMoE [FedAvg]",
        ),
        ("$K=5$ experts", "FedBeeMoE K=5"),
        ("$K=3$, top-1 gate", "FedBeeMoE top-1 gate"),
        ("$K=3$, uniform gate (no routing)", "FedBeeMoE uniform gate"),
    ]
    out = []
    for lab, m in cfg:
        d = rows[(rows.Split == split) & (rows.Model == m)]
        if len(d) == 0:
            continue
        np_ = np.mean(
            [
                EXTRA[k]["nparam"].get(m, np.nan)
                for k in EXTRA
                if k.startswith(split + "_") and m in EXTRA[k].get("nparam", {})
            ]
        )
        out.append((lab, m, d, np_))
    return out


A = abl_rows("random")
if A:
    ref = rows[
        (rows.Split == "random")
        & (rows.Model == ("FedBeeMoE" if PROPOSED_RAW == "FedBeeMoE [FedAvg]" else "FedBeeMoE [FedAvg]"))
    ].set_index("Seed")
    body = []
    for lab, m, d, npar in A:
        cells = [f"{npar/1000:.1f}k"] + [
            fmt(d[x].mean(), d[x].std()) for x in ("Accuracy", "Recall", "AUC", "MCC", "PR_AUC")
        ]
        s = d.set_index("Seed")["MCC"]
        z = ref["MCC"].loc[s.index] - s
        cells.append(
            "--"
            if m == ref.Model.iloc[0]
            else f"{stats.wilcoxon(z).pvalue:.3f}" if np.any(z != 0) and len(z) > 5 else "--"
        )
        body.append(("", lab, cells))
    write_tabular(
        os.path.join(TAB, "ablation_moe.tex"),
        ["Params", "Acc", "Rec", "AUC", "MCC", "PR-AUC", "$p$ vs $K{=}3$"],
        body,
        align="lccccccc",
        group_col=False,
    )
    SUMMARY["ablation_moe"] = {
        lab: {x: float(d[x].mean()) for x in ("Accuracy", "Recall", "AUC", "MCC", "PR_AUC")}
        for lab, m, d, _ in A
    }


body = []
for lab, m in [
    ("FedAvg", "FedAvg-GIN"),
    ("FedProx", "FedProx-GIN"),
    ("FedAdam", "FedAdam-GIN"),
    ("SCAFFOLD", "SCAFFOLD-GIN"),
]:
    d = rows[(rows.Split == "random") & (rows.Model == m)]
    if len(d):
        body.append(
            (
                "GIN",
                lab,
                [fmt(d[x].mean(), d[x].std()) for x in ("Accuracy", "AUC", "MCC")]
                + [f"{d.Val_Score.mean():.3f}"],
            )
        )
for lab, m in [
    ("FedAvg", "FedBeeMoE [FedAvg]"),
    ("FedProx", "FedBeeMoE [FedProx]"),
    ("FedAdam", "FedBeeMoE [FedAdam]"),
]:
    mm = PROP if m == PROPOSED_RAW else m
    d = rows[(rows.Split == "random") & (rows.Model == mm)]
    if len(d):
        body.append(
            (
                "FedBeeMoE",
                lab + (" (selected)" if m == PROPOSED_RAW else ""),
                [fmt(d[x].mean(), d[x].std()) for x in ("Accuracy", "AUC", "MCC")]
                + [f"{d.Val_Score.mean():.3f}"],
            )
        )
write_tabular(os.path.join(TAB, "aggregators.tex"), ["Acc", "AUC", "MCC", "Val. score"], body, align="lcccc")
if os.path.exists(os.path.join(OUT, "tuning.json")):
    SUMMARY["tuning"] = json.load(open(os.path.join(OUT, "tuning.json")))


cmodels = [
    PROP,
    "FedAvg-GIN-MoE",
    "FedAvg-GIN",
    "FedAvg-GIN+Desc",
    DENSE,
    "Local-GIN",
    "Pooled-GIN-MoE+Desc",
    "RF (FP)",
    "XGB (FP)",
]
CL = ["ECOTOX", "PPDB", "BPDB"]
for sp in SPLITS:
    c = cl[cl.Split == sp]
    if len(c) == 0:
        continue
    body = []
    for m in cmodels:
        cells = []
        for cc in CL:
            d = c[(c.Model == m) & (c.Client == cc)]
            cells += [fmt(d[x].mean(), d[x].std(), 2) if len(d) else "--" for x in ("MCC", "AUC", "Recall")]
        nm = m.replace("&", "\\&")
        if m == PROP:
            nm, cells = f"\\textbf{{{nm}}}", [f"\\textbf{{{x}}}" for x in cells]
        body.append(("", nm, cells))
    write_tabular(
        os.path.join(TAB, f"clients_{sp}.tex"),
        [f"{cc} {x}" for cc in CL for x in ("MCC", "AUC", "Rec")],
        body,
        align="l" + "c" * 9,
        group_col=False,
    )
    n_c = c[c.Model == PROP].groupby("Client")[["n", "pos"]].mean().round(1)
    SUMMARY.setdefault("client_n", {})[sp] = n_c.to_dict()
    SUMMARY.setdefault("client_perf", {})[sp] = {
        m: {
            cc: {
                x: float(c[(c.Model == m) & (c.Client == cc)][x].mean())
                for x in ("MCC", "AUC", "Recall", "Accuracy")
            }
            for cc in CL
        }
        for m in cmodels
    }
ds = df.groupby("source")["label"].agg(["size", "mean"]).round(3)
SUMMARY["source_stats"] = ds.to_dict()


def collect(split, model):
    out = []
    for k, v in PROB.items():
        if k.startswith(split + "_") and model in v["probs"]:
            p, th = v["probs"][model]
            out.append((int(k.split("_")[1]), v["idx"], v["y"], v["src"], p, th))
    return sorted(out, key=lambda x: x[0])


THR = np.round(np.arange(0.1, 0.91, 0.05), 2)
TH_SUM = {}
for sp in SPLITS:
    R = collect(sp, PROP)
    if not R:
        continue
    tab = []
    for t in THR:
        r = []
        for _, idx, y, src, p, th in R:
            pr = (p >= t - 1e-9).astype(int)
            tp = ((pr == 1) & (y == 1)).sum()
            fp = ((pr == 1) & (y == 0)).sum()
            fn = ((pr == 0) & (y == 1)).sum()
            tn = ((pr == 0) & (y == 0)).sum()
            rec = tp / max(tp + fn, 1)
            prec = tp / max(tp + fp, 1)
            mcc = (tp * tn - fp * fn) / np.sqrt(max((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn), 1))
            r.append(
                [
                    rec,
                    prec,
                    2 * prec * rec / max(prec + rec, 1e-9),
                    mcc,
                    tn / max(tn + fp, 1),
                    (tp + tn) / len(y),
                    fn,
                ]
            )
        tab.append(np.mean(r, 0))
    TH_SUM[sp] = np.array(tab)
    thr_sel = np.mean([th for *_, th in R])
    SUMMARY.setdefault("threshold_sweep", {})[sp] = dict(
        thr=THR.tolist(),
        cols=["Recall", "Precision", "F1", "MCC", "Specificity", "Accuracy", "FN"],
        values=np.round(TH_SUM[sp], 4).tolist(),
        validation_thr=float(thr_sel),
    )

    d = rows[(rows.Split == sp) & (rows.Model == PROP)]
    SUMMARY.setdefault("thr_compare", {})[sp] = dict(
        val_thr={x: float(d[x].mean()) for x in ("Accuracy", "Recall", "Precision", "F1", "MCC", "FN")},
        thr05={x: float(d["t05_" + x].mean()) for x in ("Accuracy", "Recall", "Precision", "F1", "MCC")},
        mean_thr=float(d.Thr.mean()),
        sd_thr=float(d.Thr.std()),
    )
tr = []
if "random" in TH_SUM:
    body = []
    for t, r in zip(THR, TH_SUM["random"]):
        if round(t * 100) % 10 in (0,) or t in (0.15, 0.25, 0.35, 0.45):
            body.append(
                (
                    "",
                    f"{t:.2f}",
                    [
                        f"{r[0]:.3f}",
                        f"{r[1]:.3f}",
                        f"{r[2]:.3f}",
                        f"{r[3]:.3f}",
                        f"{r[4]:.3f}",
                        f"{r[5]:.3f}",
                        f"{r[6]:.1f}",
                    ],
                )
            )
    write_tabular(
        os.path.join(TAB, "threshold.tex"),
        ["Recall", "Precision", "F1", "MCC", "Spec.", "Acc.", "FN"],
        body,
        align="lccccccc",
        group_col=False,
        first="Threshold",
    )


for sp in SPLITS:
    R = collect(sp, PROP)
    if not R:
        continue
    grp = ["Herbicide", "Fungicide", "Insecticide", "Other agrochemical"]
    G = df[["herbicide", "fungicide", "insecticide", "other_agrochemical"]].values
    stat = {g: [0, 0] for g in grp}
    sc = {c: [0, 0] for c in CL}
    miss = {}
    for _, idx, y, src, p, th in R:
        pred = p >= th - 1e-9
        for j, g in enumerate(grp):
            m = (G[idx, j] == 1) & (y == 1)
            stat[g][0] += int(m.sum())
            stat[g][1] += int((m & ~pred).sum())
        for c in CL:
            m = (src == c) & (y == 1)
            sc[c][0] += int(m.sum())
            sc[c][1] += int((m & ~pred).sum())
        for i, (ii, yy, pp) in enumerate(zip(idx, y, pred)):
            if yy == 1:
                a = miss.setdefault(int(ii), [0, 0])
                a[0] += 1
                a[1] += int(not pp)
    SUMMARY.setdefault("fn_by_group", {})[sp] = {
        g: dict(toxic=v[0], missed=v[1], fn_rate=v[1] / max(v[0], 1)) for g, v in stat.items()
    }
    SUMMARY.setdefault("fn_by_client", {})[sp] = {
        g: dict(toxic=v[0], missed=v[1], fn_rate=v[1] / max(v[0], 1)) for g, v in sc.items()
    }
    frequent = sorted(
        [
            (df.loc[i, "name"], df.loc[i, "source"], a[1] / a[0], a[0])
            for i, a in miss.items()
            if a[0] >= 3 and a[1] / a[0] >= 0.8
        ],
        key=lambda x: -x[2],
    )
    SUMMARY.setdefault("fn_persistent", {})[sp] = dict(
        n=len(frequent), examples=[(a, b, round(c, 2), d) for a, b, c, d in frequent[:12]]
    )
    if sp == "random":
        body = [("", g, [str(v[0]), str(v[1]), f"{v[1] / max(v[0], 1):.2f}"]) for g, v in stat.items()] + [
            ("", c + " (client)", [str(v[0]), str(v[1]), f"{v[1] / max(v[0], 1):.2f}"]) for c, v in sc.items()
        ]
        write_tabular(
            os.path.join(TAB, "fn_groups.tex"),
            ["Toxic test cases", "Missed", "FN rate"],
            body,
            align="lccc",
            group_col=False,
            first="Group",
        )


def tv(a, b):
    return 0.5 * np.abs(a - b).sum()


def spec_stat(g, labels):
    labs = np.unique(labels)
    means = np.array([g[labels == l].mean(0) for l in labs])
    return np.mean([tv(means[i], g[labels != labs[i]].mean(0)) for i in range(len(labs))])


rng = np.random.default_rng(0)
ROUT = {}
for sp in SPLITS:
    recs = []
    for k, ex in sorted(EXTRA.items()):
        if not k.startswith(sp + "_") or PROP not in ex.get("gate", {}):
            continue
        g = ex["gate"][PROP]
        te = ex["idx"]
        src = SRC[te]
        y = Y[te]
        use = g.mean(0)
        order = np.argsort(-use)
        gs = g[:, order]
        ent = float((-(g * np.log(g + 1e-9)).sum(1)).mean() / np.log(g.shape[1]))
        dom = np.bincount(g.argmax(1), minlength=g.shape[1])[order] / len(g)
        s_c, s_y = spec_stat(gs, src), spec_stat(gs, y)
        nc = [spec_stat(gs, rng.permutation(src)) for _ in range(500)]
        ny = [spec_stat(gs, rng.permutation(y)) for _ in range(500)]
        recs.append(
            dict(
                seed=int(k.split("_")[1]),
                use=use[order],
                dom=dom,
                ent=ent,
                s_c=s_c,
                p_c=float((np.array(nc) >= s_c).mean()),
                s_y=s_y,
                p_y=float((np.array(ny) >= s_y).mean()),
                by_client={c: gs[src == c].mean(0) for c in CL if (src == c).any()},
                by_class={int(c): gs[y == c].mean(0) for c in (0, 1)},
            )
        )
    if not recs:
        continue
    ROUT[sp] = recs
    K = len(recs[0]["use"])
    SUMMARY.setdefault("routing", {})[sp] = dict(
        runs=len(recs),
        mean_use_ranked=np.mean([r["use"] for r in recs], 0).round(3).tolist(),
        sd_use_ranked=np.std([r["use"] for r in recs], 0).round(3).tolist(),
        norm_entropy=float(np.mean([r["ent"] for r in recs])),
        dominant_share=np.mean([r["dom"] for r in recs], 0).round(3).tolist(),
        client_spec_mean=float(np.mean([r["s_c"] for r in recs])),
        client_spec_sig_runs=int(sum(r["p_c"] < 0.05 for r in recs)),
        class_spec_mean=float(np.mean([r["s_y"] for r in recs])),
        class_spec_sig_runs=int(sum(r["p_y"] < 0.05 for r in recs)),
        by_client={
            c: np.mean([r["by_client"][c] for r in recs if c in r["by_client"]], 0).round(3).tolist()
            for c in CL
        },
        by_class={str(c): np.mean([r["by_class"][c] for r in recs], 0).round(3).tolist() for c in (0, 1)},
    )


KO = {}
for sp in SPLITS:
    acc = {}
    for k, ex in sorted(EXTRA.items()):
        if not k.startswith(sp + "_") or PROP not in ex.get("knockout", {}):
            continue
        full = rows[(rows.Split == sp) & (rows.Seed == int(k.split("_")[1])) & (rows.Model == PROP)]
        if len(full) == 0:
            continue
        full = full.iloc[0]
        for j, m in ex["knockout"][PROP].items():
            a = acc.setdefault(j, {x: [] for x in ("Accuracy", "Recall", "AUC", "MCC")})
            for x in a:
                a[x].append(m[x] - full[x])
    if acc:
        KO[sp] = acc
        body = []
        for j in sorted(acc):
            lab = "all experts removed (shared head)" if j == -1 else f"expert {j + 1} removed (unranked)"
            if j >= 0:
                lab = f"expert slot {j + 1} removed"
            body.append(
                (
                    "",
                    lab,
                    [
                        fmt(np.mean(acc[j][x]), np.std(acc[j][x], ddof=1) if len(acc[j][x]) > 1 else 0, 3)
                        for x in ("Accuracy", "Recall", "AUC", "MCC")
                    ],
                )
            )
        write_tabular(
            os.path.join(TAB, f"knockout_{sp}.tex"),
            ["$\\Delta$Acc", "$\\Delta$Rec", "$\\Delta$AUC", "$\\Delta$MCC"],
            body,
            align="lcccc",
            group_col=False,
        )
        SUMMARY.setdefault("knockout", {})[sp] = {
            str(j): {x: [float(np.mean(v)), float(np.std(v))] for x, v in a.items()} for j, a in acc.items()
        }


from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
from rdkit import RDLogger

RDLogger.DisableLog("rdApp.*")
fps = [AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(s), 2, 2048) for s in df["SMILES"]]
SIM = {}
for sp in ("random", "maxmin", "time"):
    R = collect(sp, PROP)
    if not R:
        continue
    recs = []
    for seed, idx, y, src, p, th in R[:5]:
        ref = np.setdiff1d(np.arange(len(df)), idx)
        for i, ii in enumerate(idx):
            sims = np.array(DataStructs.BulkTanimotoSimilarity(fps[ii], [fps[j] for j in ref]))
            own = sims[SRC[ref] == SRC[ii]]
            oth = sims[SRC[ref] != SRC[ii]]
            recs.append(
                dict(
                    split=sp,
                    seed=seed,
                    client=SRC[ii],
                    y=int(y[i]),
                    p=float(p[i]),
                    thr=float(th),
                    all=sims.max(),
                    own=own.max() if len(own) else np.nan,
                    oth=oth.max() if len(oth) else np.nan,
                )
            )
    SIM[sp] = pd.DataFrame(recs)
    s = SIM[sp]
    SUMMARY.setdefault("similarity", {})[sp] = {
        c: dict(
            median_all=float(s[s.client == c]["all"].median()),
            median_own=float(s[s.client == c].own.median()),
            median_other=float(s[s.client == c].oth.median()),
        )
        for c in CL
    }

    q = s[s.client == "PPDB"].copy()
    if len(q) > 30:
        q["bin"] = pd.qcut(q["all"], 3, labels=["low", "mid", "high"])
        from sklearn.metrics import roc_auc_score, matthews_corrcoef

        SUMMARY.setdefault("ppdb_by_similarity", {})[sp] = {
            b: dict(
                n=int(len(g)),
                pos=float(g.y.mean()),
                auc=float(roc_auc_score(g.y, g.p)) if g.y.nunique() > 1 else np.nan,
                mcc=float(matthews_corrcoef(g.y, (g.p >= g.thr).astype(int))),
                recall=float(((g.p >= g.thr) & (g.y == 1)).sum() / max((g.y == 1).sum(), 1)),
            )
            for b, g in q.groupby("bin")
        }


BAR_C = {PROP: P["ink"]}


def colour(m, grp):
    if m == PROP:
        return P["ink"]
    return {
        "Federated (graph)": P["p3"],
        "Federated (fingerprint MLP)": P["p4"],
        "Local-only": P["p6"],
        "Pooled (neural)": P["p5"],
        "Pooled (classical ML)": P["p2"],
    }[grp]


fig, axs = plt.subplots(1, 3, figsize=(22, 10.5))
for ax, sp in zip(axs, SPLITS):
    d = MAIN_ROWS[MAIN_ROWS.Split == sp]
    if len(d) == 0:
        continue
    g = d.groupby(["Model", "Group"])["MCC"].agg(["mean", "std"]).reset_index().sort_values("mean")
    ax.barh(
        range(len(g)),
        g["mean"],
        xerr=g["std"],
        color=[colour(m, gp) for m, gp in zip(g.Model, g.Group)],
        edgecolor=P["ink"],
        lw=1.2,
        error_kw=dict(lw=1.4, capsize=3, ecolor=P["gray"]),
    )
    ax.set_yticks(range(len(g)))
    ax.set_yticklabels([m.replace(" (dense)", " (dense)") for m in g.Model], fontsize=15)
    ax.set_xlabel("MCC")
    ax.text(
        -0.02,
        1.01,
        f"({'abc'[SPLITS.index(sp)]}) {SPLIT_NAME[sp]}",
        transform=ax.transAxes,
        fontsize=20,
        fontweight="bold",
    )
import matplotlib.patches as mp

fig.legend(
    handles=[
        mp.Patch(fc=colour("x", g), ec=P["ink"], label=g)
        for g in GROUP_ORDER
        if g
        in (
            "Federated (graph)",
            "Federated (fingerprint MLP)",
            "Local-only",
            "Pooled (neural)",
            "Pooled (classical ML)",
        )
    ]
    + [mp.Patch(fc=P["ink"], ec=P["ink"], label="FedBeeMoE (proposed)")],
    loc="lower center",
    ncol=3,
    frameon=False,
    fontsize=17,
    bbox_to_anchor=(0.5, -0.06),
)
fig.tight_layout()
save(fig, "fig_ranking")


fig, axs = plt.subplots(1, 3, figsize=(22, 6.2))
show = [PROP, "FedAvg-GIN-MoE", "FedAvg-GIN", "Local-GIN", "Pooled-GIN-MoE+Desc", "RF (FP)", "XGB (FP)"]
cols = [P["ink"], P["p3"], P["p5"], P["gray"], P["p2"], P["p4"], P["p6"]]
for ax, met in zip(axs, ("Accuracy", "AUC", "MCC")):
    for i, (m, c) in enumerate(zip(show, cols)):
        mu = [MAIN_ROWS[(MAIN_ROWS.Split == s) & (MAIN_ROWS.Model == m)][met].mean() for s in SPLITS]
        sd = [MAIN_ROWS[(MAIN_ROWS.Split == s) & (MAIN_ROWS.Model == m)][met].std() for s in SPLITS]
        ax.errorbar(
            range(3),
            mu,
            yerr=sd,
            color=c,
            marker="os^vDPX"[i],
            ms=11,
            lw=3 if m == PROP else 2,
            capsize=4,
            label=m,
            ls="-" if m == PROP else "--",
        )
    ax.set_xticks(range(3))
    ax.set_xticklabels([SPLIT_NAME[s] for s in SPLITS])
    ax.set_ylabel(met)
    ax.text(
        -0.18,
        1.03,
        f"({'abc'[('Accuracy','AUC','MCC').index(met)]})",
        transform=ax.transAxes,
        fontsize=22,
        fontweight="bold",
    )
axs[0].legend(loc="lower left", frameon=False, fontsize=12)
fig.tight_layout()
save(fig, "fig_splits")


grid = np.linspace(0, 1, 201)


def mean_curve(R, kind):
    ys = []
    for _, idx, y, src, p, th in R:
        if kind == "pr":
            pr, rc, _ = precision_recall_curve(y, p)
            pr, rc = pr[::-1], rc[::-1]
            ys.append(np.array([pr[rc >= g].max() if (rc >= g).any() else 0 for g in grid]))
        else:
            fpr, tpr, _ = roc_curve(y, p)
            ys.append(np.interp(grid, fpr, tpr))
    return np.mean(ys, 0), np.std(ys, 0)


cm = [PROP, "FedAvg-GIN-MoE", "FedAvg-GIN", "RF (FP)", "XGB (FP)", "Pooled-GIN-MoE+Desc"]
cc = [P["ink"], P["p3"], P["p5"], P["p4"], P["gray"], P["p2"]]
fig, axs = plt.subplots(1, 3, figsize=(24, 7))
for ax, kind in zip(axs[:2], ("pr", "roc")):
    for m, c in zip(cm, cc):
        R = collect("random", m)
        if not R:
            continue
        mu, sd = mean_curve(R, kind)
        ax.plot(grid, mu, color=c, lw=4 if m == PROP else 2.5, label=m, ls="-" if m == PROP else "--")
        if m == PROP:
            ax.fill_between(grid, mu - sd, mu + sd, color=P["p5"], alpha=0.35)
    ax.set_xlabel("Recall" if kind == "pr" else "False positive rate")
    ax.set_ylabel("Precision" if kind == "pr" else "True positive rate")
    if kind == "roc":
        ax.plot([0, 1], [0, 1], color=P["gray"], lw=1.5, ls=":")
    ax.text(-0.16, 1.03, f"({'ab'[kind == 'roc']})", transform=ax.transAxes, fontsize=22, fontweight="bold")
axs[0].legend(frameon=False, loc="lower left", fontsize=12)
if "random" in TH_SUM:
    ax = axs[2]
    for j, (lab, c) in enumerate(
        zip(["Recall", "Precision", "F1", "MCC"], [P["ink"], P["p4"], P["p2"], P["gray"]])
    ):
        ax.plot(THR, TH_SUM["random"][:, j], color=c, lw=3.5, marker="o", ms=7, label=lab)
    ax.axvline(SUMMARY["thr_compare"]["random"]["mean_thr"], color=P["p3"], lw=2.5, ls="--")
    ax.set_xlabel("Decision threshold")
    ax.set_ylabel("Score")
    ax.legend(frameon=False, fontsize=13)
    ax.text(-0.16, 1.03, "(c)", transform=ax.transAxes, fontsize=22, fontweight="bold")
fig.tight_layout()
save(fig, "fig_pr_threshold")


if "random" in ROUT:
    recs = ROUT["random"]
    K = len(recs[0]["use"])
    fig, axs = plt.subplots(1, 3, figsize=(23, 6.4))
    ax = axs[0]
    U = np.array([r["use"] for r in recs])
    ax.boxplot(
        [U[:, k] for k in range(K)],
        patch_artist=True,
        widths=0.55,
        boxprops=dict(facecolor=P["p6"], edgecolor=P["ink"], lw=1.6),
        medianprops=dict(color=P["ink"], lw=2.5),
        whiskerprops=dict(lw=1.5),
        capprops=dict(lw=1.5),
    )
    ax.axhline(1 / K, color=P["gray"], ls=":", lw=2)
    ax.set_xticklabels([f"E{k + 1}" for k in range(K)])
    ax.set_xlabel("Expert (ranked by use within each run)")
    ax.set_ylabel("Mean gate weight")
    ax.text(-0.2, 1.03, "(a)", transform=ax.transAxes, fontsize=22, fontweight="bold")
    for ax, key, labs, nm in ((axs[1], "by_client", CL, "Client"), (axs[2], "by_class", [0, 1], "Class")):
        M = np.array([np.mean([r[key][l] for r in recs if l in r[key]], 0) for l in labs])
        im = ax.imshow(M, cmap=CM, aspect="auto", vmin=0, vmax=max(M.max(), 0.5))
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                ax.text(
                    j,
                    i,
                    f"{M[i, j]:.2f}",
                    ha="center",
                    va="center",
                    fontsize=18,
                    fontweight="bold",
                    color="white" if M[i, j] > 0.4 * M.max() else P["ink"],
                )
        ax.set_xticks(range(K))
        ax.set_xticklabels([f"E{k + 1}" for k in range(K)])
        ax.set_yticks(range(len(labs)))
        ax.set_yticklabels(labs if nm == "Client" else ["non-toxic", "toxic"])
        for s_ in ax.spines.values():
            s_.set_visible(False)
        cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
        cb.set_label("Mean gate weight", fontsize=15)
        cb.outline.set_visible(False)
        ax.text(
            -0.25, 1.03, f"({'bc'[nm == 'Class']})", transform=ax.transAxes, fontsize=22, fontweight="bold"
        )
    fig.tight_layout()
    save(fig, "fig_routing")


if A:
    fig, axs = plt.subplots(1, 2, figsize=(19, 6.8))
    labs = [a[0].replace("$", "") for a in A]
    short = ["No head", "Dense\n(matched)", "K=1", "K=2", "K=3", "K=5", "Top-1", "Uniform"][: len(A)]
    for ax, met in zip(axs, ("MCC", "AUC")):
        mu = [a[2][met].mean() for a in A]
        sd = list(np.nan_to_num([a[2][met].std() for a in A]))
        ax.bar(
            range(len(A)),
            mu,
            yerr=sd,
            color=[P["ink"] if "soft" in a[0] else P["p4"] for a in A],
            edgecolor=P["ink"],
            lw=1.4,
            error_kw=dict(lw=1.6, capsize=4, ecolor=P["gray"]),
        )
        ax.set_xticks(range(len(A)))
        ax.set_xticklabels(short, fontsize=14)
        ax.set_ylabel(met)
        ax.set_ylim(min(mu) - 0.12, max(mu) + max(sd) + 0.04)
        ax.text(
            -0.15, 1.03, f"({'ab'[met == 'AUC']})", transform=ax.transAxes, fontsize=22, fontweight="bold"
        )
    fig.tight_layout()
    save(fig, "fig_ablation")


fig, axs = plt.subplots(1, 3, figsize=(24, 6.6))
for ax, sp in zip(axs[:1], ["random"]):
    c = cl[cl.Split == sp]
    w = 0.12
    for i, (m, col) in enumerate(
        zip(cmodels[:7], [P["ink"], P["p3"], P["p5"], P["p4"], P["p6"], P["gray"], P["p2"]])
    ):
        mu = [c[(c.Model == m) & (c.Client == k)]["MCC"].mean() for k in CL]
        sd = [c[(c.Model == m) & (c.Client == k)]["MCC"].std() for k in CL]
        ax.bar(
            np.arange(3) + (i - 3) * w,
            mu,
            w,
            yerr=sd,
            color=col,
            edgecolor=P["ink"],
            lw=1,
            label=m.replace(" (dense)", " (dense)"),
            error_kw=dict(lw=1.2, capsize=2, ecolor=P["gray"]),
        )
    ax.set_xticks(range(3))
    ax.set_xticklabels(CL)
    ax.set_ylabel("MCC")
    ax.legend(frameon=False, fontsize=10, ncol=2, loc="upper right")
    ax.text(-0.15, 1.03, "(a)", transform=ax.transAxes, fontsize=22, fontweight="bold")
if "random" in SIM:
    s = SIM["random"]
    ax = axs[1]
    data = [s[s.client == k]["all"] for k in CL]
    bp = ax.boxplot(
        data,
        patch_artist=True,
        widths=0.55,
        boxprops=dict(facecolor=P["p6"], edgecolor=P["ink"], lw=1.6),
        medianprops=dict(color=P["ink"], lw=2.5),
    )
    ax.set_xticklabels(CL)
    ax.set_ylabel("Max. Tanimoto to reference set")
    ax.text(-0.15, 1.03, "(b)", transform=ax.transAxes, fontsize=22, fontweight="bold")
    ax = axs[2]
    w = 0.25
    for i, sp in enumerate(SPLITS):
        if sp not in SIM:
            continue
        ss = SIM[sp]
        ax.bar(
            np.arange(3) + (i - 1) * w,
            [ss[ss.client == k]["all"].median() for k in CL],
            w,
            color=[P["p6"], P["p4"], P["ink"]][i],
            edgecolor=P["ink"],
            label=SPLIT_NAME[sp],
        )
    ax.set_xticks(range(3))
    ax.set_xticklabels(CL)
    ax.set_ylabel("Median max. Tanimoto")
    ax.legend(frameon=False)
    ax.text(-0.15, 1.03, "(c)", transform=ax.transAxes, fontsize=22, fontweight="bold")
fig.tight_layout()
save(fig, "fig_clients_similarity")

json.dump(
    SUMMARY,
    open(os.path.join(REP, "summary.json"), "w"),
    indent=1,
    default=lambda o: o.item() if hasattr(o, "item") else str(o),
)
rows.to_csv(os.path.join(REP, "all_runs.csv"), index=False)
print("report written to", REP)
