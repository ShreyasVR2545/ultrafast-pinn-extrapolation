"""SSFM accuracy-vs-cost ladder (Phase 6.6): for each regime and a few N values, coarsen the step
dxi and the tau grid (same window), time a single-threaded NumPy SSFM run, and measure its relative
L2 error against the validated reference field (coarse-grid results are spectrally interpolated back
onto the reference tau grid by zero-padding in Fourier space). Writes results/ssfm_cost.csv.

The cheapest configuration reaching a model's error gives the "SSFM time at matched accuracy".

Usage: python scripts/05_ssfm_cost.py [--smoke]
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nlse_ml.data import splits  # noqa: E402
from nlse_ml.data.datasets import load_eval_field  # noqa: E402
from nlse_ml.physics.nlse import initial_condition, tau_grid  # noqa: E402
from nlse_ml.physics.ssfm import ssfm  # noqa: E402

CASES = {"A": [1.5, 2.5, 4.0], "B_sech": [1.3, 3.0], "B_gauss": [1.3, 3.0]}
DXI = [0.02, 0.01, 5e-3, 2.5e-3, 1e-3, 5e-4, 2e-4, 1e-4]


def upsample(U, n_ref):
    """Spectral interpolation along tau to n_ref points (same periodic window)."""
    n = U.shape[-1]
    if n == n_ref:
        return U
    F = np.fft.fftshift(np.fft.fft(U, axis=-1), axes=-1)
    pad = (n_ref - n) // 2
    F = np.pad(F, [(0, 0), (pad, n_ref - n - pad)])
    return np.fft.ifft(np.fft.ifftshift(F, axes=-1), axis=-1) * (n_ref / n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    rows = []
    cases = {"A": [1.5]} if a.smoke else CASES
    for regime, Ns in cases.items():
        g = splits.load_grid(regime)
        rc = splits.regime_cfg(regime)
        n_list = [g["n_tau"] // 4, g["n_tau"] // 2, g["n_tau"]] if not a.smoke else [g["n_tau"] // 4]
        for N in Ns:
            ref = load_eval_field(regime, N).U
            for n_tau in n_list:
                tau, _ = tau_grid(n_tau, g["tau_max"])
                u0 = initial_condition(rc["pulse"], tau)
                for dxi in (DXI[:2] if a.smoke else DXI):
                    best = np.inf
                    for _ in range(2):  # best of 2 timings
                        t0 = time.perf_counter()
                        _, U = ssfm(u0, tau, rc["s"], N, g["xi_max"], g["n_save"], dxi)
                        best = min(best, time.perf_counter() - t0)
                    err = float(np.linalg.norm(upsample(U, g["n_tau"]) - ref) / np.linalg.norm(ref))
                    rows.append({"regime": regime, "N": N, "n_tau": n_tau, "dxi": dxi, "wall_s": best, "rel_l2": err})
                    print(rows[-1])
    if not a.smoke:
        pd.DataFrame(rows).to_csv(ROOT / "results" / "ssfm_cost.csv", index=False)


if __name__ == "__main__":
    main()
