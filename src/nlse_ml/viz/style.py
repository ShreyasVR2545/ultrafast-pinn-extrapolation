"""Shared figure style. Fixed categorical colour per model (never re-assigned by rank), plus a
marker and line style per model so identity never relies on colour alone (CVD / print)."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# Validated reference categorical palette (dataviz skill, light mode), in fixed slot order.
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
SEQ_CMAP = "magma"  # perceptually uniform, one-directional, for intensity heatmaps

MODELS = {  # key: (label, colour, marker, linestyle)
    "pinn_strict": ("PINN-strict", PALETTE[0], "o", "-"),
    "pinn_extended": ("PINN-extended", PALETTE[1], "s", "-"),
    "pinn_plus": ("PINN+", PALETTE[2], "D", "-"),
    "dd_mlp": ("DD-MLP", PALETTE[3], "^", "--"),
    "cfno": ("cFNO", PALETTE[4], "v", "--"),
    "cfno_ar": ("cFNO-AR", PALETTE[5], "P", "--"),
    "pinn_finetune": ("PINN-finetune", PALETTE[6], "X", ":"),
    "ssfm": ("SSFM", INK2, "*", ":"),
}


def apply():
    plt.rcParams.update({
        "figure.dpi": 110, "savefig.dpi": 300, "font.size": 9, "axes.titlesize": 10,
        "axes.labelsize": 9, "legend.fontsize": 8, "axes.edgecolor": INK2, "axes.labelcolor": INK,
        "xtick.color": INK2, "ytick.color": INK2, "axes.grid": True, "grid.color": GRID,
        "grid.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False,
        "lines.linewidth": 2.0, "lines.markersize": 5, "figure.facecolor": "white",
        "axes.facecolor": "white", "legend.frameon": False,
    })


def save(fig, path):
    """Save PNG (300 dpi) and PDF next to each other."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path.with_suffix(".png"), bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight", metadata={"CreationDate": None})  # reproducible bytes
    plt.close(fig)
