"""Core figures from committed results/ CSVs (plus run predictions for heatmap/spectra figures).

  python scripts/make_figures.py [--sweep main]           # every figure whose inputs exist
  python scripts/make_figures.py --only extrap_N pareto

Figures needing predictions (heatmaps_compare, spectra_far) read runs/<sweep>/.../preds.npz and the
SSFM data in data/; when those are absent (fresh clone), they are skipped with a message. Every
other figure is regenerated from results/ alone.
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

FIG = ROOT / "figures"
REG_TITLE = {"A": "Regime A (normal, Gaussian)", "B_sech": "Regime B (anomalous, sech)",
             "B_gauss": "Regime B (anomalous, Gaussian)"}


def _load(sweep):
    long = pd.read_csv(ROOT / "results" / f"{sweep}_long.csv")
    runs = pd.read_csv(ROOT / "results" / f"{sweep}_runs.csv")
    sp = ROOT / "results" / f"{sweep}_slices.csv.gz"
    slices = pd.read_csv(sp) if sp.exists() else pd.DataFrame()
    for d in (long, runs, slices):
        if "tag" in d:
            d["tag"] = d["tag"].fillna("")
    return long, runs, slices


def _models(df):
    return [m for m in MODELS if m in set(df.model)]


def _band(ax, x, mean, std, key):
    lab, col, mk, ls = MODELS[key]
    ax.plot(x, mean, color=col, marker=mk, ls=ls, label=lab)
    ok = np.isfinite(std)
    if ok.any():
        lo = np.maximum(mean - std, mean * 0.05)
        ax.fill_between(x[ok], lo[ok], (mean + std)[ok], color=col, alpha=0.15, lw=0)


def fig_extrap_N(long, regime, sweep):
    base = long[(long.regime == regime) & (long.exp == "n_extrap") & (long.tag == "")]
    if base.empty:
        return
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    tr = splits.train_n(regime)
    nseed = base.groupby("model").seed.nunique().min()
    for ax, metric, ylab in [(axes[0], "rel_l2", "relative L2 error, complex field"),
                             (axes[1], "rel_l2_intensity", r"relative L2 error, intensity $|U|^2$")]:
        d = base[base.metric == metric]
        ax.axvspan(min(tr), max(tr), color=style.GRID, alpha=0.7, lw=0)
        ax.text(np.mean(tr), 0.98, "training range", transform=ax.get_xaxis_transform(), ha="center", va="top",
                color=style.INK2, fontsize=8)
        for m in _models(d):
            g = d[d.model == m].groupby("N")["value"]
            x = np.array(sorted(g.groups))
            _band(ax, x, g.mean().loc[x].values, g.std(ddof=1).loc[x].values, m)
        ax.set(yscale="log", xlabel="soliton order N", ylabel=ylab)
    axes[0].set_title("complex field (phase-sensitive)")
    axes[1].set_title("intensity only")
    axes[1].legend(ncol=2, fontsize=7.5)
    fig.suptitle(f"{REG_TITLE[regime]}: N-extrapolation (mean ± std over {nseed} seeds)", fontsize=10)
    fig.tight_layout()
    style.save(fig, FIG / f"fig_extrap_N_{'regime' + regime if regime == 'A' else regime}")


def fig_extrap_z(slices, sweep):
    d = slices[(slices.metric == "rel_l2") & (slices.exp == "z_extrap") & (slices.tag == "")]
    if d.empty:
        return
    regs = [r for r in ["A", "B_sech", "B_gauss"] if r in set(d.regime)]
    fig, axes = plt.subplots(1, len(regs), figsize=(4.3 * len(regs), 3.6), squeeze=False)
    for ax, reg in zip(axes[0], regs):
        dd = d[(d.regime == reg) & (d.split == "interp")]
        ax.axvspan(0, 2, color=style.GRID, alpha=0.7, lw=0)
        ax.text(1.0, 0.98, "trained ξ ≤ 2", transform=ax.get_xaxis_transform(), ha="center", va="top",
                color=style.INK2, fontsize=8)
        for m in _models(dd):
            g = dd[dd.model == m].groupby("xi")["value"]  # pooled over interp N and seeds
            x = np.array(sorted(g.groups))
            lab, col, mk, ls = MODELS[m]
            mean = g.mean().loc[x].values
            ax.plot(x, mean, color=col, ls=ls, label=lab)
            ax.plot(x[::20], mean[::20], color=col, marker=mk, ls="none")
        ax.set(yscale="log", xlabel=r"$\xi$", ylabel="relative L2 per slice", title=REG_TITLE[reg] + ", interp N")
    axes[0, 0].legend(fontsize=7)
    fig.tight_layout()
    style.save(fig, FIG / "fig_extrap_z")


def fig_energy_drift(slices, sweep, regime="A"):
    d = slices[(slices.metric == "energy_drift") & (slices.regime == regime) & (slices.exp == "n_extrap")
               & (slices.tag == "")]
    if d.empty:
        return
    sps = [s for s in ["interp", "far" if regime == "A" else "extrap"] if s in set(d.split)]
    fig, axes = plt.subplots(1, len(sps), figsize=(4.6 * len(sps), 3.6), squeeze=False, sharey=True)
    for ax, sp in zip(axes[0], sps):
        dd = d[d.split == sp]
        for m in _models(dd):
            g = dd[dd.model == m].groupby("xi")["value"]
            x = np.array(sorted(g.groups))
            lab, col, mk, ls = MODELS[m]
            y = np.maximum(g.median().loc[x].values, 1e-8)
            ax.plot(x[1:], y[1:], color=col, ls=ls, label=lab)
            ax.plot(x[1::25], y[1::25], color=col, marker=mk, ls="none")
        ax.set(yscale="log", xlabel=r"$\xi$", title=f"{REG_TITLE[regime]}, {sp} N (median over N, seeds)")
    axes[0, 0].set_ylabel(r"$|E(\xi)-E(0)|/E(0)$")
    axes[0, 0].legend(fontsize=7)
    fig.tight_layout()
    style.save(fig, FIG / f"fig_energy_drift{'' if regime == 'A' else '_' + regime}")


def fig_pareto(long, runs, regime="A"):
    cost = ROOT / "results" / "ssfm_cost.csv"
    d = long[(long.metric == "rel_l2") & (long.regime == regime) & (long.exp == "n_extrap") & (long.tag == "")]
    if d.empty:
        return
    r = runs[(runs.regime == regime) & (runs.exp == "n_extrap") & (runs.tag == "")]
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.8), sharey=False)
    sp_far = "far" if regime == "A" else "extrap"
    for ax, sp in [(axes[0], "interp"), (axes[1], sp_far)]:
        for m in _models(d):
            e = d[(d.model == m) & (d.split == sp)].groupby("seed")["value"].mean()
            rr = r[r.model == m].set_index("seed").reindex(e.index)
            lab, col, mk, _ = MODELS[m]
            ax.scatter(rr["train_wall_s"], e, color=col, marker=mk, s=36, label=lab + " (training)",
                       edgecolor="white", linewidth=0.8, zorder=3)
            ax.scatter(rr["inference_s_per_field"], e, facecolor="none", edgecolor=col, marker=mk, s=36,
                       linewidth=1.2, zorder=3, label=None)
        ax.scatter([], [], facecolor="none", edgecolor=style.INK2, marker="o", s=36, label="hollow: model inference")
        if cost.exists():
            c = pd.read_csv(cost)
            c = c[(c.regime == regime) & (c.rel_l2 > 1e-8)]  # drop configs equal to the reference itself
            ns = [n for n in sorted(c.N.unique())]
            for k, n in enumerate(ns):
                cc = c[c.N == n].sort_values("wall_s")
                front = cc[cc.rel_l2 == cc.rel_l2.cummin()]
                ax.plot(front.wall_s, front.rel_l2, color=style.INK2, ls=":", marker="*", ms=6, lw=1,
                        label="SSFM (cost ladder, one field)" if k == 0 else None)
                ax.annotate(f"N={n:g}", (front.wall_s.iloc[-1], front.rel_l2.iloc[-1]), fontsize=6.5,
                            color=style.INK2, xytext=(3, 0), textcoords="offset points")
        ax.set(xscale="log", yscale="log", xlabel="wall-clock (s): model training / inference, SSFM solve",
               ylabel="relative L2 error",
               title=f"{REG_TITLE[regime]}: {sp} split")
    axes[0].legend(fontsize=6.5, loc="best")
    fig.text(0.5, -0.02, "Models: one training run serves all N; SSFM: cost of ONE field per point "
             "(single-threaded NumPy).", ha="center", fontsize=7, color=style.INK2)
    fig.tight_layout()
    style.save(fig, FIG / f"fig_pareto{'' if regime == 'A' else '_' + regime}")


def _preds(sweep, model, regime, exp, seed=0):
    p = ROOT / "runs" / sweep / exp / model / regime / f"seed{seed}" / "preds.npz"
    return np.load(p) if p.exists() else None


def fig_heatmaps_compare(sweep, regime="A", n_pair=(1.5, 4.0)):
    from nlse_ml.data.datasets import load_eval_field

    try:
        truths = {N: load_eval_field(regime, N) for N in n_pair}
    except FileNotFoundError:
        print("skip heatmaps_compare: data/ missing (run scripts/04_generate_data.py)")
        return
    models = [m for m in MODELS if _preds(sweep, m, regime, "n_extrap") is not None]
    if not models:
        print("skip heatmaps_compare: no preds")
        return
    tl = 30 if regime == "A" else 10
    fig, axes = plt.subplots(len(n_pair) * 2, len(models) + 1, figsize=(2.1 * (len(models) + 1), 6.8),
                             squeeze=False)
    from matplotlib.colors import LogNorm, PowerNorm

    for i, N in enumerate(n_pair):
        f = truths[N]
        m_t = np.abs(f.tau) <= tl
        I = np.abs(f.U[:, m_t]) ** 2
        vmax = I.max()
        ext = [f.tau[m_t][0], f.tau[m_t][-1], f.xi[0], f.xi[-1]]
        ax = axes[2 * i, 0]
        ax.imshow(I, origin="lower", aspect="auto", extent=ext, cmap=style.SEQ_CMAP, norm=PowerNorm(0.4, 0, vmax))
        ax.set_title(f"SSFM, N={N:g}", fontsize=8)
        axes[2 * i + 1, 0].axis("off")
        axes[2 * i + 1, 0].text(0.5, 0.5, f"|error| rows:\n|P - T| / max|T|\n(log scale)", ha="center",
                                va="center", fontsize=7, color=style.INK2, transform=axes[2 * i + 1, 0].transAxes)
        for j, m in enumerate(models, start=1):
            P = _preds(sweep, m, regime, "n_extrap")[f"N_{N:.4f}"][:, m_t]
            axes[2 * i, j].imshow(np.abs(P) ** 2, origin="lower", aspect="auto", extent=ext, cmap=style.SEQ_CMAP,
                                  norm=PowerNorm(0.4, 0, vmax))
            axes[2 * i, j].set_title(f"{MODELS[m][0]}", fontsize=8)
            err = np.abs(P - f.U[:, m_t]) / np.abs(f.U).max()
            im = axes[2 * i + 1, j].imshow(np.maximum(err, 1e-5), origin="lower", aspect="auto", extent=ext,
                                           cmap="Greys", norm=LogNorm(1e-4, 1))
    for ax in axes.ravel():
        ax.grid(False)
        ax.tick_params(labelsize=6)
    fig.subplots_adjust(hspace=0.45, wspace=0.25)
    fig.colorbar(im, ax=axes[:, -1], fraction=0.05, label="|error| / max|U|")
    fig.suptitle(f"{REG_TITLE[regime]}: intensity (power-law colour scale) and error, seed 0; "
                 f"rows: N={n_pair[0]:g} (interp), N={n_pair[1]:g} (far)", fontsize=9)
    style.save(fig, FIG / f"fig_heatmaps_compare{'' if regime == 'A' else '_' + regime}")


def fig_spectra_far(sweep, regime="A", N=4.0):
    from nlse_ml.data.datasets import load_eval_field
    from nlse_ml.eval.metrics import omega_axis, power_spectrum

    try:
        f = load_eval_field(regime, N)
    except FileNotFoundError:
        print("skip spectra_far: data/ missing")
        return
    models = [m for m in MODELS if _preds(sweep, m, regime, "n_extrap") is not None]
    if not models:
        return
    w = omega_axis(f.tau)
    St = power_spectrum(f.U[-1:])[0]
    ref = St.max()
    db = lambda S: 10 * np.log10(np.maximum(S / ref, 1e-12))
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    ax.plot(w, db(St), color=style.INK, lw=2.4, label=f"SSFM (truth)")
    for m in models:
        P = _preds(sweep, m, regime, "n_extrap")[f"N_{N:.4f}"][-1:]
        lab, col, mk, ls = MODELS[m]
        ax.plot(w, db(power_spectrum(P)[0]), color=col, ls=ls, lw=1.4, label=lab)
    wl = 15 if regime == "A" else 30
    ax.set(xlim=(-wl, wl), ylim=(-60, 5), xlabel=r"$\omega T_0$", ylabel="power spectrum (dB, rel. true peak)",
           title=f"{REG_TITLE[regime]}: output spectrum at ξ=4, N={N:g} (far extrapolation, seed 0)")
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    style.save(fig, FIG / f"fig_spectra_far{'' if regime == 'A' else '_' + regime}")


ALL = ["extrap_N", "extrap_z", "energy_drift", "pareto", "heatmaps_compare", "spectra_far"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep", default="main")
    ap.add_argument("--only", nargs="+", default=ALL)
    a = ap.parse_args()
    style.apply()
    long, runs, slices = _load(a.sweep)
    for reg in ["A", "B_sech", "B_gauss"]:
        if "extrap_N" in a.only:
            fig_extrap_N(long, reg, a.sweep)
        if "pareto" in a.only and reg == "A":
            fig_pareto(long, runs, reg)
    if "extrap_z" in a.only and not slices.empty:
        fig_extrap_z(slices, a.sweep)
    if "energy_drift" in a.only and not slices.empty:
        for reg in ["A", "B_sech"]:
            fig_energy_drift(slices, a.sweep, reg)
    if "heatmaps_compare" in a.only:
        fig_heatmaps_compare(a.sweep, "A", (1.5, 4.0))
        fig_heatmaps_compare(a.sweep, "B_sech", (1.3, 3.0))
    if "spectra_far" in a.only:
        fig_spectra_far(a.sweep, "A", 4.0)
        fig_spectra_far(a.sweep, "B_sech", 3.0)
    print("figures done")


if __name__ == "__main__":
    main()
