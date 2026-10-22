"""Phase 8/9 figures: ablations (results/ablations_long.csv + main seed-matched baselines) and test-time
fine-tuning (results/finetune.csv + results/ssfm_cost.csv). Regenerates from results/ alone.

  python scripts/make_ablation_figures.py [--aggregate]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nlse_ml.data import splits  # noqa: E402
from nlse_ml.viz import style  # noqa: E402
from nlse_ml.viz.style import MODELS  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402

RES, FIG = ROOT / "results", ROOT / "figures"
SPL = ["interp", "near", "far"]


def _load():
    ab = pd.read_csv(RES / "ablations_long.csv")
    mn = pd.read_csv(RES / "main_long.csv")
    for d in (ab, mn):
        d["tag"] = d["tag"].fillna("")
    mn = mn[(mn.regime == "A") & (mn.exp == "n_extrap") & (mn.tag == "")].assign(tag="main")
    return pd.concat([ab[(ab.regime == "A") & (ab.exp == "n_extrap")], mn])


def _split_means(d, metric="rel_l2"):
    d = d[d.metric == metric]
    return d.groupby(["model", "tag", "seed", "split"])["value"].mean().reset_index()


def fig_range_width(d):
    ranges = {"rw_narrow": (1.4, 1.6), "rw_mid": (1.2, 1.8), "main": (1.0, 2.0)}
    labels = {"rw_narrow": "train N ∈ [1.4, 1.6]", "rw_mid": "train N ∈ [1.2, 1.8]", "main": "train N ∈ [1.0, 2.0]"}
    models = [m for m in ["pinn_strict", "dd_mlp", "cfno"] if m in set(d.model)]
    if not models or "rw_narrow" not in set(d.tag):
        return
    fig, axes = plt.subplots(1, len(models), figsize=(4.2 * len(models), 3.6), sharey=True, squeeze=False)
    for ax, m in zip(axes[0], models):
        for k, (tag, (lo, hi)) in enumerate(ranges.items()):
            dd = d[(d.model == m) & (d.tag == tag) & (d.metric == "rel_l2") & (d.seed.isin([0, 1]))]
            if dd.empty:
                continue
            dist = np.maximum(0, np.maximum(dd.N - hi, lo - dd.N))
            g = dd.assign(dist=dist.round(3)).groupby("dist")["value"]
            x = np.array(sorted(g.groups))
            ax.plot(x, g.mean().loc[x].values, color=style.PALETTE[k], marker="osD"[k], label=labels[tag])
        ax.set(yscale="log", xlabel="distance of test N outside the training range", title=MODELS[m][0])
    axes[0, 0].set_ylabel("relative L2 (mean over seeds 0-1)")
    axes[0, 0].legend(fontsize=7)
    fig.suptitle("Ablation 1: training-range width (Regime A). Is apparent generalization a narrow-range artifact?",
                 fontsize=9)
    fig.tight_layout()
    style.save(fig, FIG / "fig_ablation_range_width")


def _bars(ax, sm, model, tags, names, title):
    w = 0.8 / len(tags)
    for k, (tag, name) in enumerate(zip(tags, names)):
        vals = [sm[(sm.model == model) & (sm.tag == tag) & (sm.split == s) & (sm.seed == 0)]["value"].mean()
                for s in SPL]
        ax.bar(np.arange(len(SPL)) + (k - (len(tags) - 1) / 2) * w, vals, w * 0.92, color=style.PALETTE[k],
               label=name)
    ax.set(xticks=range(len(SPL)), xticklabels=SPL, yscale="log", ylabel="relative L2 (seed 0)", title=title)
    ax.legend(fontsize=7)


def fig_bars(d):
    sm = _split_means(d)
    panels = [
        ("pinn_plus", ["main", "pp_noff", "pp_nocausal", "pp_noadapt"],
         ["PINN+ (all on)", "no Fourier features", "no causal weights", "no adaptive weights"], "PINN+ components"),
        ("pinn_strict", ["col10k", "main", "col50k"], ["10k points", "20k (main)", "50k points"], "Collocation points"),
        ("pinn_strict", ["wic1", "main", "wic100"], ["w_ic = 1", "w_ic = 10 (main)", "w_ic = 100"], "IC loss weight"),
        ("dd_mlp", ["de2", "de4", "main"], ["2 train N", "4 train N", "6 train N (main)"], "Data efficiency: DD-MLP"),
        ("cfno", ["de2", "de4", "main"], ["2 train N", "4 train N", "6 train N (main)"], "Data efficiency: cFNO"),
    ]
    panels = [p for p in panels if any(t in set(sm.tag) for t in p[1] if t != "main")]
    if not panels:
        return
    fig, axes = plt.subplots(1, len(panels), figsize=(4 * len(panels), 3.6), squeeze=False)
    for ax, (m, tags, names, title) in zip(axes[0], panels):
        _bars(ax, sm, m, tags, names, title)
    fig.suptitle("Ablations 2-5 (Regime A, N-extrapolation, seed 0)", fontsize=9)
    fig.tight_layout()
    style.save(fig, FIG / "fig_ablation_components")


def fig_finetune():
    p = RES / "finetune.csv"
    if not p.exists():
        return
    ft = pd.read_csv(p)
    cost = pd.read_csv(RES / "ssfm_cost.csv")
    cases = ft.groupby(["regime", "N"]).size().index.tolist()
    fig, axes = plt.subplots(1, len(cases), figsize=(3.6 * len(cases), 3.4), squeeze=False, sharey=True)
    for ax, (reg, N) in zip(axes[0], cases):
        d = ft[(ft.regime == reg) & (ft.N == N) & (ft.seed == 0)].sort_values("step")
        ax.plot(d.wall_s.clip(lower=0.1), d.rel_l2, color=MODELS["pinn_finetune"][1], marker="X", markevery=4,
                label="PINN fine-tune (physics only)")
        tgt = d.target_interp.iloc[0]
        ax.axhline(tgt, color=style.INK2, ls="--", lw=1)
        ax.text(0.12, tgt * 1.1, "model's in-distribution error", fontsize=6.5, color=style.INK2)
        c = cost[cost.regime == reg]
        if not c.empty:
            n_near = c.N.iloc[(c.N - N).abs().argmin()]
            ok = c[(c.N == n_near) & (c.rel_l2 <= tgt)]
            if not ok.empty:
                t_ssfm = ok.wall_s.min()
                ax.axvline(t_ssfm, color=style.INK, lw=1.2, ls=":")
                ax.text(t_ssfm * 1.15, ax.get_ylim()[1] if False else tgt * 3, f"SSFM reaches it\nin {t_ssfm * 1e3:.0f} ms"
                        + ("" if n_near == N else f"\n(ladder N={n_near:g})"), fontsize=6.5)
        ax.set(xscale="log", yscale="log", xlabel="fine-tuning wall-clock (s)",
               title=f"{reg}, N={N:g}")
    axes[0, 0].set_ylabel("relative L2 at that N")
    axes[0, 0].legend(fontsize=7)
    fig.tight_layout()
    style.save(fig, FIG / "fig_finetune")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--aggregate", action="store_true")
    a = ap.parse_args()
    if a.aggregate:
        from nlse_ml.eval import aggregate

        aggregate.write("ablations")
    style.apply()
    if (RES / "ablations_long.csv").exists():
        d = _load()
        fig_range_width(d)
        fig_bars(d)
    fig_finetune()
    print("ablation figures done")


if __name__ == "__main__":
    main()
