import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

FIG_DIR = Path(__file__).resolve().parent / "figures"

logging.getLogger("fontTools").setLevel(logging.ERROR)


def f(x):
    return x ** 2


def h(x):
    return 1 - x


def lagrangian(mu, x):
    return f(x) + mu * h(x)


def x_of_mu(mu):
    return mu / 2


def dual(mu):
    return mu - mu ** 2 / 4


X_STAR = 1.0
MU_STAR = 2.0
P_STAR = 1.0


BLUE = "#2a78d6"
ORANGE = "#eb6834"
AQUA = "#1baf7a"
BLUE_RAMP = ["#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]

INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
WASH = "#f0efec"


def setup_style():
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["cmr10"],
        "mathtext.fontset": "cm",
        "axes.formatter.use_mathtext": True,
        "axes.unicode_minus": False,
        "font.size": 9,
        "axes.labelsize": 10,
        "legend.fontsize": 8.5,
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5,
        "text.color": INK,
        "axes.labelcolor": INK,
        "axes.edgecolor": AXIS,
        "axes.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "grid.linestyle": "-",
        "xtick.color": AXIS,
        "ytick.color": AXIS,
        "xtick.labelcolor": INK2,
        "ytick.labelcolor": INK2,
        "lines.linewidth": 1.7,
        "lines.solid_capstyle": "round",
        "lines.dash_capstyle": "round",
        "legend.frameon": False,
        "legend.handlelength": 1.8,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.03,
        "pdf.fonttype": 42,
    })


def mark_point(ax, x, y, color=ORANGE, size=7, zorder=6):
    ax.plot(x, y, "o", ms=size, mfc=color, mec="white", mew=1.5, zorder=zorder)


def save(fig, name):
    FIG_DIR.mkdir(exist_ok=True)
    fig.savefig(FIG_DIR / f"{name}.png", dpi=200)
    plt.close(fig)
    print(f"  saved figures/{name}.png")
