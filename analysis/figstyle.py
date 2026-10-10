import os
import numpy as np, pandas as pd, matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm

plt.rcParams.update(
    {
        "font.family": "Liberation Sans",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "font.size": 17,
        "axes.labelsize": 19,
        "axes.titlesize": 19,
        "xtick.labelsize": 16,
        "ytick.labelsize": 16,
        "legend.fontsize": 16,
        "axes.linewidth": 1.5,
        "xtick.major.width": 1.5,
        "ytick.major.width": 1.5,
        "xtick.major.size": 5,
        "ytick.major.size": 5,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)
D = os.path.join(os.environ.get("APIS_DATA", "."), "dataset_final.csv")
df = pd.read_csv(D)
COL_TOX = {0: "#9DB7C9", 1: "#5B2A8F"}
NAME_TOX = {0: "Non-toxic", 1: "Toxic"}
COL_SRC = {"ECOTOX": "#1F77B4", "PPDB": "#E08A1E", "BPDB": "#2A9D8F"}


def agro(r):
    for k, n in (
        ("insecticide", "Insecticide"),
        ("herbicide", "Herbicide"),
        ("fungicide", "Fungicide"),
        ("other_agrochemical", "Other agrochemical"),
    ):
        if r[k] == 1:
            return n
    return "Not agrochemical"


df["cls"] = df.apply(agro, axis=1)
CLS = ["Insecticide", "Herbicide", "Fungicide", "Other agrochemical", "Not agrochemical"]
COL_CLS = {
    "Insecticide": "#D1495B",
    "Herbicide": "#4C9F38",
    "Fungicide": "#B5379B",
    "Other agrochemical": "#E9B000",
    "Not agrochemical": "#8A8A8A",
}


def tag(ax, s, dx=-0.16, dy=1.06):
    ax.text(dx, dy, s, transform=ax.transAxes, fontsize=26, fontweight="bold", va="bottom", ha="left")
