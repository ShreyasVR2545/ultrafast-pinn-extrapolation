"""Phase 3: grid-adequacy study for the hardest cases, validation figure and gallery figures.

Per regime, picks the cheapest candidate (tau_max, n_tau) whose hardest cases satisfy
  * energy fraction in the outer 5% of the time window AND of the frequency window < 1e-6 (all xi)
  * doubling n_tau (same window) changes the field by < 1e-6 relative L2
then the largest dxi such that halving it changes the field by < 1e-6 and the Hamiltonian
drift is < 1e-6. Writes configs/grid.yaml, results/grid_study.json,
figures/ssfm_validation.{png,pdf} and figures/gallery_*.{png,pdf}.

Usage: python scripts/03_validate_ssfm.py [--smoke] [--gallery-only]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nlse_ml.physics import analytic, conservation as C  # noqa: E402
from nlse_ml.physics.nlse import initial_condition, tau_grid  # noqa: E402
from nlse_ml.physics.ssfm import SOLVER_VERSION, ssfm  # noqa: E402
from nlse_ml.viz import style  # noqa: E402

XI_MAX, N_SAVE, TOL = 4.0, 201, 1e-6
HARDEST = {"A": [("gaussian", 1, 4.0)], "B": [("sech", -1, 3.0), ("gaussian", -1, 3.0)]}
CANDIDATES = {"A": [(50.0, 1024), (60.0, 1024), (60.0, 2048), (80.0, 2048)],
              "B": [(30.0, 1024), (30.0, 2048), (40.0, 2048)]}
DXI = [2e-4, 1e-4, 5e-5, 2.5e-5, 1.25e-5]


def rel_l2(a, b):
    return float(np.linalg.norm(a - b) / np.linalg.norm(b))


def edge_fractions(U):
    n = U.shape[-1]
    k = int(round(0.05 * n))
    I = np.abs(U) ** 2
    S = np.abs(np.fft.fftshift(np.fft.fft(U, axis=-1), axes=-1)) ** 2
    et = (I[:, :k].sum(1) + I[:, -k:].sum(1)) / I.sum(1)
    ef = (S[:, :k].sum(1) + S[:, -k:].sum(1)) / S.sum(1)
    return et, ef


def run(shape, s, N, tau_max, n_tau, dxi, xi_max=XI_MAX, n_save=N_SAVE):
    tau, _ = tau_grid(n_tau, tau_max)
    xi, U = ssfm(initial_condition(shape, tau), tau, s, N, xi_max, n_save, dxi)
    return xi, tau, U


def study_regime(reg, smoke):
    rec = {"candidates": [], "dxi": []}
    xi_max = 1.0 if smoke else XI_MAX
    chosen = None
    for tau_max, n_tau in CANDIDATES[reg]:
        ok, row = True, {"tau_max": tau_max, "n_tau": n_tau, "cases": []}
        for shape, s, N in HARDEST[reg]:
            _, _, U = run(shape, s, N, tau_max, n_tau, 1e-4, xi_max)
            _, _, U2 = run(shape, s, N, tau_max, 2 * n_tau, 1e-4, xi_max)
            et, ef = edge_fractions(U)
            d_tau = rel_l2(U, U2[:, ::2])
            c = {"shape": shape, "s": s, "N": N, "edge_time_max": float(et.max()),
                 "edge_freq_max": float(ef.max()), "dtau_halving_change": d_tau}
            c["pass"] = bool(et.max() < TOL and ef.max() < TOL and d_tau < TOL)
            ok &= c["pass"]
            row["cases"].append(c)
        row["pass"] = ok
        rec["candidates"].append(row)
        print(reg, json.dumps(row))
        if ok:
            chosen = (tau_max, n_tau)
            break
    if chosen is None:
        raise SystemExit(f"HARD STOP: no candidate grid passes for regime {reg}")
    tau_max, n_tau = chosen
    dxi_chosen = None
    for h in DXI:
        ok, row = True, {"dxi": h, "cases": []}
        for shape, s, N in HARDEST[reg]:
            t0 = time.time()
            xi, tau, U = run(shape, s, N, tau_max, n_tau, h, xi_max)
            wall = time.time() - t0
            _, _, Uh = run(shape, s, N, tau_max, n_tau, h / 2, xi_max)
            dH = C.relative_drift(C.hamiltonian(U, tau, s, N)).max()
            dE = C.relative_drift(C.energy(U, tau)).max()
            c = {"shape": shape, "N": N, "dxi_halving_change": rel_l2(U, Uh), "H_drift": float(dH),
                 "E_drift": float(dE), "wall_s": wall}
            c["pass"] = bool(c["dxi_halving_change"] < TOL and dH < TOL)
            ok &= c["pass"]
            row["cases"].append(c)
        row["pass"] = ok
        rec["dxi"].append(row)
        print(reg, json.dumps(row))
        if ok and dxi_chosen is None:
            dxi_chosen = h
            break
    if dxi_chosen is None:
        raise SystemExit(f"HARD STOP: no dxi passes for regime {reg}")
    rec["chosen"] = {"tau_max": tau_max, "n_tau": n_tau, "dxi": dxi_chosen,
                     "xi_max": XI_MAX, "n_save": N_SAVE}
    return rec


def validation_figure(grids, out):
    import matplotlib.pyplot as plt

    style.apply()
    fig, ax = plt.subplots(2, 2, figsize=(9, 6.5))
    cases = [("A", "gaussian", 1, 4.0, "A: Gaussian, s=+1, N=4"),
             ("B", "sech", -1, 3.0, "B: sech, s=-1, N=3"),
             ("B", "gaussian", -1, 3.0, "B: Gaussian, s=-1, N=3")]
    # (a) step convergence vs a fine reference
    a = ax[0, 0]
    for k, (reg, shape, s, N, lab) in enumerate(cases):
        g = grids[reg]
        _, _, ref = run(shape, s, N, g["tau_max"], g["n_tau"], 2.5e-6, n_save=5)
        hs = np.array([4e-3, 2e-3, 1e-3, 5e-4, 2.5e-4, 1.25e-4])
        err = [rel_l2(run(shape, s, N, g["tau_max"], g["n_tau"], h, n_save=5)[2], ref) for h in hs]
        a.loglog(hs, err, marker="os^"[k], color=style.PALETTE[k], label=lab)
    a.loglog(hs, 0.5 * err[0] * (hs / hs[0]) ** 2, color=style.INK2, ls=":", lw=1, label=r"slope 2")
    for reg, lsty, ytxt in (("A", "--", 1e-2), ("B", "-.", 2e-7)):
        a.axvline(grids[reg]["dxi"], color=style.INK2, lw=0.8, ls=lsty)
        a.text(grids[reg]["dxi"] * 1.1, ytxt, f"chosen, {reg}", color=style.INK2, fontsize=7)
    a.set(xlabel=r"step $\Delta\xi$", ylabel="rel. L2 error vs fine reference", title="(a) SSFM convergence")
    a.legend()
    # (b) conservation along xi at the chosen grid
    b = ax[0, 1]
    for k, (reg, shape, s, N, lab) in enumerate(cases):
        g = grids[reg]
        xi, tau, U = run(shape, s, N, g["tau_max"], g["n_tau"], g["dxi"])
        dH = C.relative_drift(C.hamiltonian(U, tau, s, N))
        dE = C.relative_drift(C.energy(U, tau))
        b.semilogy(xi[1:], dH[1:], color=style.PALETTE[k], label=f"|ΔH|/H, {lab}")
        b.semilogy(xi[1:], np.maximum(dE[1:], 1e-17), color=style.PALETTE[k], ls=":", lw=1.2)
    b.axhline(TOL, color=style.INK2, lw=0.8, ls="--")
    b.text(0.05, TOL * 1.5, "1e-6 target", color=style.INK2, fontsize=7)
    b.set(xlabel=r"$\xi$", ylabel="relative drift", title="(b) Conservation (solid H, dotted E)")
    b.legend(fontsize=6.5)
    # (c) edge energy at the chosen grid
    c = ax[1, 0]
    for k, (reg, shape, s, N, lab) in enumerate(cases):
        g = grids[reg]
        xi, tau, U = run(shape, s, N, g["tau_max"], g["n_tau"], 1e-4)
        et, ef = edge_fractions(U)
        c.semilogy(xi, np.maximum(et, 1e-30), color=style.PALETTE[k], label=f"time, {lab}")
        c.semilogy(xi, np.maximum(ef, 1e-30), color=style.PALETTE[k], ls=":", lw=1.2)
    c.axhline(TOL, color=style.INK2, lw=0.8, ls="--")
    c.set(xlabel=r"$\xi$", ylabel="energy fraction in outer 5%", title="(c) Window adequacy (solid τ, dotted ω)")
    c.text(0.05, TOL * 1.5, "1e-6 limit", color=style.INK2, fontsize=7)
    c.legend(fontsize=6.5, loc="center left")
    # (d) analytic checks
    d = ax[1, 1]
    tau, _ = tau_grid(1024, 40.0)
    for k, s_ in enumerate([1, -1]):
        xi, U = ssfm(initial_condition("gaussian", tau), tau, s_, 0.0, 4.0, 41, 0.1)
        e = np.linalg.norm(U - analytic.gvd_gaussian(xi, tau, s_), axis=1) / np.linalg.norm(U, axis=1)
        d.semilogy(xi[1:], e[1:], color=style.PALETTE[k], marker="os"[k], markevery=5,
                   label=f"GVD-only Gaussian, s={s_:+d}")
    xi, U = ssfm(initial_condition("sech", tau), tau, -1, 1.0, 2 * np.pi, 41, 1e-3)
    e = np.linalg.norm(U - analytic.fundamental_soliton(xi, tau), axis=1) / np.linalg.norm(U, axis=1)
    d.semilogy(xi[1:], e[1:], color=style.PALETTE[2], marker="^", markevery=5, label=r"N=1 soliton, $\Delta\xi$=1e-3")
    d.set(xlabel=r"$\xi$", ylabel="rel. L2 vs analytic", title="(d) Analytic solutions")
    d.legend(fontsize=7)
    fig.tight_layout()
    style.save(fig, out)


def gallery(grids):
    import matplotlib.pyplot as plt

    style.apply()
    cases = [("GVD only (N=0), Gaussian, s=+1", "A", "gaussian", 1, 0.0, "gallery_gvd_only"),
             ("SPM only (no dispersion), Gaussian, N=2", "A", "gaussian", 0, 2.0, "gallery_spm_only"),
             ("Regime A: Gaussian, s=+1, N=2", "A", "gaussian", 1, 2.0, "gallery_A_N2"),
             ("Regime A: Gaussian, s=+1, N=4", "A", "gaussian", 1, 4.0, "gallery_A_N4"),
             ("Regime B: sech, s=-1, N=1", "B", "sech", -1, 1.0, "gallery_B_N1"),
             ("Regime B: sech, s=-1, N=2", "B", "sech", -1, 2.0, "gallery_B_N2")]
    for title, reg, shape, s, N, name in cases:
        g = grids[reg]
        xi, tau, U = run(shape, s, N, g["tau_max"], g["n_tau"], g["dxi"])
        I = np.abs(U) ** 2
        omega = np.fft.fftshift(2 * np.pi * np.fft.fftfreq(tau.size, tau[1] - tau[0]))
        S = np.abs(np.fft.fftshift(np.fft.fft(U, axis=1), axes=1)) ** 2
        SdB = 10 * np.log10(S / S.max() + 1e-30)
        fig, ax = plt.subplots(1, 2, figsize=(9, 3.4))
        tl = min(g["tau_max"], 12 if reg == "B" else 45)
        m = np.abs(tau) <= tl
        from matplotlib.colors import PowerNorm

        im = ax[0].pcolormesh(tau[m], xi, I[:, m], shading="auto", cmap=style.SEQ_CMAP, rasterized=True,
                              norm=PowerNorm(0.4, vmin=0, vmax=I.max()))
        fig.colorbar(im, ax=ax[0], label=r"$|U|^2$ (power-law colour scale, $\gamma$ = 0.4)")
        ax[0].set(xlabel=r"$\tau = T/T_0$", ylabel=r"$\xi = z/L_D$", title="Intensity")
        wl = 25 if reg == "B" else 15
        mw = np.abs(omega) <= wl
        im = ax[1].pcolormesh(omega[mw], xi, SdB[:, mw], shading="auto", cmap=style.SEQ_CMAP,
                              vmin=-60, vmax=0, rasterized=True)
        fig.colorbar(im, ax=ax[1], label="spectrum (dB)")
        ax[1].set(xlabel=r"$\omega T_0$", ylabel=r"$\xi$", title="Spectrum (dB, floor -60)")
        for a_ in ax:
            a_.grid(False)
        fig.suptitle(title)
        fig.tight_layout()
        style.save(fig, ROOT / "figures" / name)
        print("wrote", name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--gallery-only", action="store_true")
    ap.add_argument("--figures-only", action="store_true", help="redo all figures from configs/grid.yaml")
    a = ap.parse_args()
    t0 = time.time()
    if a.gallery_only or a.figures_only:
        grids = yaml.safe_load((ROOT / "configs" / "grid.yaml").read_text())["regimes"]
        if a.figures_only:
            validation_figure(grids, ROOT / "figures" / "ssfm_validation")
        gallery(grids)
        return
    recs = {reg: study_regime(reg, a.smoke) for reg in ("A", "B")}
    if a.smoke:
        print("smoke ok", {k: v["chosen"] for k, v in recs.items()}, f"{time.time() - t0:.0f}s")
        return
    grids = {reg: r["chosen"] for reg, r in recs.items()}
    cfg = {"solver": SOLVER_VERSION, "method": "symmetric SSFM, float64",
           "criteria": {"edge_fraction_outer5pct": TOL, "dtau_halving_change": TOL,
                        "dxi_halving_change": TOL, "hamiltonian_drift": TOL},
           "hardest_cases": {k: [list(c) for c in v] for k, v in HARDEST.items()},
           "regimes": grids}
    (ROOT / "configs" / "grid.yaml").write_text(
        "# Generated by scripts/03_validate_ssfm.py; details in results/grid_study.json\n"
        + yaml.safe_dump(cfg, sort_keys=False))
    (ROOT / "results" / "grid_study.json").write_text(json.dumps(recs, indent=2))
    validation_figure(grids, ROOT / "figures" / "ssfm_validation")
    gallery(grids)
    print(f"done in {time.time() - t0:.0f}s", grids)


if __name__ == "__main__":
    main()
