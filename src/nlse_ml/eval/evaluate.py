"""Evaluate a trained model on every eval N of its experiment and write metrics.json + preds.npz.

metrics.json layout:
  {"per_N": {"<N>": {"split": ..., <scalar metrics>}}, "per_split": {split: {metric: mean over N}},
   "per_slice": {"<N>": {"xi": [...], "rel_l2": [...], ...}}, "inference_s_per_field": ...}
"""
from __future__ import annotations

import time

import numpy as np

from ..data import splits
from ..data.datasets import load_eval_field
from . import metrics as M


def xi_mask(xi, exp):
    lo, hi = splits.load_splits()["experiments"][exp]["xi_eval"]
    if exp == "z_extrap":
        return (xi > lo + 1e-9) & (xi <= hi + 1e-9)
    return (xi >= lo - 1e-9) & (xi <= hi + 1e-9)


def evaluate(predict_fn, regime: str, exp: str, residual_fn=None) -> tuple[dict, dict]:
    """predict_fn(N, xi_eval, tau, U0) -> complex (len(xi_eval), n_tau).
    residual_fn(N, xi_range) -> autograd PDE residual RMS (MLP models only) or None."""
    s = splits.regime_cfg(regime)["s"]
    per_N, per_slice, preds = {}, {}, {}
    t_inf = []
    for split, n_list in splits.eval_n(regime, exp).items():
        for N in n_list:
            f = load_eval_field(regime, N)
            m = xi_mask(f.xi, exp)
            xi_e = f.xi[m]
            t0 = time.perf_counter()
            P = predict_fn(N, xi_e, f.tau, f.U[0])
            t_inf.append(time.perf_counter() - t0)
            T = f.U[m]
            fm = M.field_metrics(P, T, xi_e, f.tau, s, N)
            fm["split"] = split
            if not np.all(np.isfinite(P)):
                fm["nonfinite_pred"] = True
            if residual_fn is not None:
                fm["autograd_residual_rms"] = residual_fn(N, (float(xi_e.min()), float(xi_e.max())))
            per_N[f"{N:.4f}"] = fm
            ps = M.per_slice(P, T, f.tau, s, N)
            per_slice[f"{N:.4f}"] = {"xi": xi_e.tolist(), **{k: np.asarray(v).tolist() for k, v in ps.items()}}
            preds[f"N_{N:.4f}"] = P.astype(np.complex64)
    per_split = {}
    for split in {v["split"] for v in per_N.values()}:
        rows = [v for v in per_N.values() if v["split"] == split]
        keys = [k for k, v in rows[0].items() if isinstance(v, (int, float)) and not isinstance(v, bool)]
        per_split[split] = {k: float(np.nanmean([r[k] for r in rows])) for k in keys}
    out = {"per_N": per_N, "per_split": per_split, "per_slice": per_slice,
           "inference_s_per_field": float(np.median(t_inf))}
    return out, preds
