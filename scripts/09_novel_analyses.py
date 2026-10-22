"""Phase 9 analyses A (spectral shift vs error) and B (label-free drift vs error).

  python scripts/09_novel_analyses.py [--sweep main]          # needs data/ + results/<sweep>_*.csv
  python scripts/09_novel_analyses.py --figures-only          # regenerate figures from results/ alone

Writes results/analysis_A_points.csv.gz, results/analysis_A_spearman.csv,
results/analysis_B_points.csv, results/analysis_B_spearman.csv,
figures/fig_error_vs_spectral_shift.{png,pdf}, figures/fig_drift_vs_error.{png,pdf}.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nlse_ml.eval.analysis import spearman_ci, spectral_shift_table  # noqa: E402
from nlse_ml.viz import style  # noqa: E402
from nlse_ml.viz.style import MODELS  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402

RES = ROOT / "results"


def build_A(sweep):
    sl = pd.read_csv(RES / f"{sweep}_slices.csv.gz")
    sl = sl[(sl.metric == "rel_l2") & (sl.tag.fillna("") == "")].rename(columns={"value": "err"})
    shift = pd.concat([spectral_shift_table(r) for r in sorted(sl.regime.unique())])
    pts = sl.merge(shift, on=["regime", "N", "xi"], how="left")
    pts = pts[pts.xi > 0]
    is_z = pts.exp == "z_extrap"
    pts["wf_train"] = np.where(is_z, pts.wf_train_z, pts.wf_train_n)
    pts["phi_excess"] = np.where(is_z, pts.phi_excess_z, pts.phi_excess_n)
    # z_extrap: only the extrapolated slices count as "test" points; n_extrap: all slices of all N
    pts = pts[~is_z | (pts.xi > 2.0 + 1e-9)]
    keep = ["model", "regime", "exp", "split", "N", "seed", "xi", "err", "wf_train", "wf_input", "phi_eff",
            "phi_excess"]
    pts[keep].to_csv(RES / "analysis_A_points.csv.gz", index=False, compression="gzip")
    return pts[keep]


def stats_A(pts):
    rows = []
    for (m, exp), d in pts.groupby(["model", "exp"]):
        for x in ["wf_train", "phi_excess", "phi_eff", "wf_input"]:
            rho, lo, hi, n = spearman_ci(d, x, "err")
            rows.append({"model": m, "exp": exp, "predictor": x, "rho": rho, "ci_lo": lo, "ci_hi": hi, "n_points": n,
                         "n_clusters": d.groupby(["regime", "N", "seed"]).ngroups})
    out = pd.DataFrame(rows)
    out.to_csv(RES / "analysis_A_spearman.csv", index=False)
    return out


def build_B(sweep):
    lg = pd.read_csv(RES / f"{sweep}_long.csv")
    lg = lg[lg.tag.fillna("") == ""]
    w = lg.pivot_table(index=["model", "regime", "exp", "split", "N", "seed"], columns="metric",
                       values="value").reset_index()
    cols = ["rel_l2", "phase_err", "rel_l2_intensity", "energy_drift_max", "hamiltonian_drift_max",
            "grid_residual", "autograd_residual_rms"]
    w = w[[c for c in ["model", "regime", "exp", "split", "N", "seed", *cols] if c in w.columns]]
    w.to_csv(RES / "analysis_B_points.csv", index=False)
    return w


def stats_B(w):
    rows = []
    for (m, exp), d in w.groupby(["model", "exp"]):
        d = d.assign(cl=d.regime + d.N.astype(str))
        for x in ["energy_drift_max", "hamiltonian_drift_max", "grid_residual", "autograd_residual_rms"]:
            if x not in d or d[x].isna().all():
                continue
            rho, lo, hi, n = spearman_ci(d, x, "rel_l2", cluster=("cl", "seed"))
            rows.append({"model": m, "exp": exp, "estimator": x, "rho": rho, "ci_lo": lo, "ci_hi": hi, "n": n})
    # Within-N test (removes the N confound): at each (regime, exp, N), rank all model/seed predictions
    # by the label-free estimator and by true error; average Spearman rho over N, bootstrap CI over N.
    for exp, d in w.groupby("exp"):
        for x in ["energy_drift_max", "hamiltonian_drift_max", "grid_residual"]:
            per = []
            for _, g in d.groupby(["regime", "N"]):
                g = g[[x, "rel_l2"]].replace([np.inf, -np.inf], np.nan).dropna()
                if len(g) >= 5:
                    per.append(spearmanr(g[x], g.rel_l2).statistic)
            per = np.array(per)
            rng = np.random.default_rng(0)
            boots = [np.nanmean(rng.choice(per, per.size)) for _ in range(2000)] if per.size else [np.nan]
            rows.append({"model": "ALL (within-N, across models)", "exp": exp, "estimator": x,
                         "rho": float(np.nanmean(per)) if per.size else np.nan,
                         "ci_lo": float(np.nanpercentile(boots, 2.5)), "ci_hi": float(np.nanpercentile(boots, 97.5)),
                         "n": int(per.size)})
    out = pd.DataFrame(rows)
    out.to_csv(RES / "analysis_B_spearman.csv", index=False)
    return out


def fig_A(pts, st):
    style.apply()
    exps = [e for e in ["n_extrap", "z_extrap"] if e in set(pts.exp)]
    preds = [("wf_train", "WF distance to nearest training spectrum"),
             ("phi_excess", r"SPM phase beyond training box, $\Delta\phi$")]
    fig, axes = plt.subplots(len(exps), 2, figsize=(10, 3.9 * len(exps)), squeeze=False)
    for i, exp in enumerate(exps):
        for j, (x, xl) in enumerate(preds):
            ax = axes[i, j]
            d = pts[pts.exp == exp]
            for m in [m for m in MODELS if m in set(d.model)]:
                dm = d[d.model == m]
                dm = dm.iloc[:: max(1, len(dm) // 1500)]  # thin for readability; stats use all points
                r = st[(st.model == m) & (st.exp == exp) & (st.predictor == x)]
                lab, col, mk, _ = MODELS[m]
                rl = f" ρ={r.rho.iloc[0]:.2f} [{r.ci_lo.iloc[0]:.2f},{r.ci_hi.iloc[0]:.2f}]" if len(r) else ""
                xv = dm[x] + (1e-4 if x == "phi_excess" else 0)
                ax.scatter(xv, dm.err, s=5, color=col, marker=mk, alpha=0.35, lw=0, label=lab + rl)
            ax.set(xscale="symlog" if x == "phi_excess" else "log", yscale="log", xlabel=xl,
                   ylabel="relative L2 per slice", title=f"{exp}: error vs {x}")
            if x == "phi_excess":
                ax.set_xscale("symlog", linthresh=0.1)
                ax.set_xlim(left=0)
                ax.text(0.01, 0.02, "Δφ = 0: inside the training box", transform=ax.transAxes, fontsize=7,
                        color=style.INK2)
            ax.legend(fontsize=6, markerscale=2.5)
    fig.tight_layout()
    style.save(fig, ROOT / "figures" / "fig_error_vs_spectral_shift")


def fig_B(w, st):
    style.apply()
    ests = [("energy_drift_max", r"max$_\xi$ $|\Delta E|/E$"), ("hamiltonian_drift_max", r"max$_\xi$ $|\Delta H|/|H|$"),
            ("grid_residual", "grid PDE residual (RMS, rel.)")]
    d = w[w.exp == "n_extrap"]
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))
    for ax, (x, xl) in zip(axes, ests):
        for m in [m for m in MODELS if m in set(d.model)]:
            dm = d[d.model == m]
            r = st[(st.model == m) & (st.exp == "n_extrap") & (st.estimator == x)]
            lab, col, mk, _ = MODELS[m]
            rl = f" ρ={r.rho.iloc[0]:.2f} [{r.ci_lo.iloc[0]:.2f},{r.ci_hi.iloc[0]:.2f}]" if len(r) else ""
            ax.scatter(np.maximum(dm[x], 1e-9), dm.rel_l2, s=16, color=col, marker=mk, alpha=0.8,
                       edgecolor="white", linewidth=0.5, label=lab + rl)
        ax.set(xscale="log", yscale="log", xlabel=xl + "  (no ground truth needed)", ylabel="true relative L2",
               title=f"N-extrap: error vs {x}")
        ax.legend(fontsize=5.8)
    fig.tight_layout()
    style.save(fig, ROOT / "figures" / "fig_drift_vs_error")
    # failure case: energy conserved but phase wrong
    fig, ax = plt.subplots(figsize=(5.4, 3.9))
    for m in [m for m in MODELS if m in set(d.model)]:
        dm = d[d.model == m]
        lab, col, mk, _ = MODELS[m]
        ax.scatter(np.maximum(dm.energy_drift_max, 1e-9), dm.phase_err, s=16, color=col, marker=mk,
                   edgecolor="white", linewidth=0.5, label=lab)
    ax.set(xscale="log", yscale="log", xlabel=r"max$_\xi$ $|\Delta E|/E$", ylabel="intensity-weighted phase error (rad)",
           title="Energy conservation does not certify the phase")
    ax.legend(fontsize=7)
    fig.tight_layout()
    style.save(fig, ROOT / "figures" / "fig_energy_vs_phase")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep", default="main")
    ap.add_argument("--figures-only", action="store_true")
    a = ap.parse_args()
    if a.figures_only:
        pts = pd.read_csv(RES / "analysis_A_points.csv.gz")
        stA = pd.read_csv(RES / "analysis_A_spearman.csv")
        w = pd.read_csv(RES / "analysis_B_points.csv")
        stB = pd.read_csv(RES / "analysis_B_spearman.csv")
    else:
        pts = build_A(a.sweep)
        stA = stats_A(pts)
        w = build_B(a.sweep)
        stB = stats_B(w)
    fig_A(pts, stA)
    fig_B(w, stB)
    with pd.option_context("display.width", 200, "display.max_rows", 200):
        print(stA.round(3).to_string())
        print(stB.round(3).to_string())


if __name__ == "__main__":
    main()
