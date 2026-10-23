"""Novel analyses (Phase 9).

A. Spectral shift as an error predictor (extension of Fesser et al. 2023 to parametric NLSE):
   for every (run, N, xi slice): true error vs
   * wf_train : Wasserstein-Fourier distance from the TRUE spectrum at (N, xi) to the nearest
                spectrum the model was trained on (n_extrap: training N at the same xi;
                z_extrap: same N at any training slice xi' <= 2). Needs ground truth, so it is an
                a-posteriori explanatory variable, not a deployable estimator.
   * phi_excess : closed-form, label-free SPM driver: N^2 asinh(xi) minus its largest value inside
                the training box (0 inside the training distribution).
B. Label-free error estimators: energy drift, Hamiltonian drift, grid PDE residual (and autograd
   residual for MLP models) vs true error, per (run, N).

Statistics: Spearman rho per model, 95% CI by cluster bootstrap over (N, seed) groups.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from ..data import splits
from ..data.datasets import load_eval_field
from .metrics import omega_axis, power_spectrum
from .spectral_shift import spm_phase, wasserstein_fourier


def _spectra(regime):
    out = {}
    for N in splits.all_n(regime):
        f = load_eval_field(regime, N)
        out[N] = power_spectrum(f.U)
    w = omega_axis(f.tau)
    return out, w, f.xi


def spectral_shift_table(regime: str) -> pd.DataFrame:
    """Per (exp, N, xi): wf_train, wf_input, phi_eff, phi_excess (ground-truth spectra only)."""
    S, w, xi = _spectra(regime)
    tr = splits.train_n(regime)
    rows = []
    for N, SN in S.items():
        # n_extrap: nearest training N at the same xi
        d_n = np.min(np.stack([wasserstein_fourier(SN, S[t], w) for t in tr]), 0)
        # z_extrap: same N, nearest training slice xi' <= 2
        tr_idx = np.nonzero(xi <= 2.0 + 1e-9)[0]
        d_z = np.array([np.min(wasserstein_fourier(np.broadcast_to(SN[k], SN[tr_idx].shape), SN[tr_idx], w))
                        for k in range(len(xi))])
        d_in = wasserstein_fourier(np.broadcast_to(SN[0], SN.shape), SN, w)
        phi = spm_phase(N, xi)
        exc_n = np.maximum(0.0, phi - spm_phase(max(tr), xi))
        exc_z = np.maximum(0.0, phi - spm_phase(N, 2.0))
        for k in range(len(xi)):
            rows.append({"regime": regime, "N": N, "xi": xi[k], "wf_input": d_in[k], "phi_eff": phi[k],
                         "wf_train_n": d_n[k], "wf_train_z": d_z[k], "phi_excess_n": exc_n[k],
                         "phi_excess_z": exc_z[k]})
    return pd.DataFrame(rows)


PREDICTORS = ["wf_input", "phi_eff", "wf_train_n", "wf_train_z", "phi_excess_n", "phi_excess_z"]


def merge_shift(slices: pd.DataFrame, shift: pd.DataFrame, decimals: int = 6) -> pd.DataFrame:
    """Left-merge per-slice errors with the spectral-shift table on (regime, N, xi).

    The keys are rounded to `decimals` on BOTH sides: the committed slice CSV stores xi rounded to 4
    decimals while the shift table uses the raw np.linspace values (e.g. 0.7 vs 0.7000000000000001),
    and an exact-float merge silently left 24 of 201 slices without predictors (DECISIONS D-031).
    Raises if any predictor is missing after the merge.
    """
    a, b = slices.copy(), shift.copy()
    for d in (a, b):
        d["_xi"] = d["xi"].round(decimals)
        d["_N"] = d["N"].round(decimals)
    out = a.merge(b.drop(columns=["xi", "N"]), on=["regime", "_N", "_xi"], how="left").drop(columns=["_xi", "_N"])
    missing = out[PREDICTORS].isna().any(axis=1)
    if missing.any():
        raise ValueError(f"{int(missing.sum())} rows have no spectral-shift predictors after the merge, e.g. "
                         f"{out.loc[missing, ['regime', 'N', 'xi']].head(3).to_dict('records')}")
    return out


def spearman_ci(df: pd.DataFrame, x: str, y: str, cluster=("N", "seed"), n_boot=1000, seed=0):
    d = df[[x, y, *cluster]].replace([np.inf, -np.inf], np.nan).dropna()
    if len(d) < 5 or d[x].nunique() < 3:
        return np.nan, np.nan, np.nan, len(d)
    rho = spearmanr(d[x], d[y]).statistic
    groups = [g for _, g in d.groupby(list(cluster))]
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n_boot):
        pick = rng.integers(0, len(groups), len(groups))
        b = pd.concat([groups[i] for i in pick])
        if b[x].nunique() > 2:
            boots.append(spearmanr(b[x], b[y]).statistic)
    lo, hi = np.nanpercentile(boots, [2.5, 97.5]) if boots else (np.nan, np.nan)
    return float(rho), float(lo), float(hi), len(d)
