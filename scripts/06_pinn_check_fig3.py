"""Implementation check: our PINN code on the starter repo's fig3 problem (single N = sqrt(1.3),
normal dispersion, Gaussian, tau in [-24, 24], xi in [0, 4]); the repo reaches ~3% (results/repro_repo.md).
If ours is comparable at a comparable budget, the PINN implementation is sound and harder results
elsewhere come from the problem, not a bug. Writes results/pinn_check_fig3.json.

Usage: python scripts/06_pinn_check_fig3.py [--smoke] [--adam 10000 --lbfgs 2000 --n-col 20000]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nlse_ml.data import splits  # noqa: E402
from nlse_ml.physics.nlse import gaussian, tau_grid  # noqa: E402
from nlse_ml.physics.ssfm import ssfm  # noqa: E402
from nlse_ml.train import train_pinn  # noqa: E402

N0 = float(np.sqrt(1.3))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--adam", type=int, default=10000)
    ap.add_argument("--lbfgs", type=int, default=2000)
    ap.add_argument("--n-col", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tau-max", type=float, default=24.0)
    a = ap.parse_args()
    if a.smoke:
        a.adam, a.lbfgs, a.n_col = 50, 5, 1000
    # Patch the split/grid accessors for this one-off problem (no files touched).
    grid = {"n_tau": 1024, "tau_max": a.tau_max, "xi_max": 4.0, "n_save": 201, "dxi": 2e-4}
    splits.load_grid = lambda regime: grid
    splits.regime_cfg = lambda regime: {"s": 1, "pulse": "gaussian"}
    splits.collocation_n_range = lambda regime, v: (N0 - 1e-3, N0 + 1e-3)
    splits.xi_train_range = lambda exp: (0.0, 4.0)
    cfg = {"arch": {"width": 100, "depth": 4, "act": "tanh"}, "loss": {"w_ic": 10.0, "w_r": 1.0},
           "deriv_mode": "taylor", "collocation": "strict",
           "schedule": {"adam_steps": a.adam, "lr": 1e-3, "lr_final": 1e-4, "resample_every": 100,
                        "lbfgs_steps": a.lbfgs, "lbfgs_history": 50, "log_every": 500},
           "sampling": {"n_col": a.n_col, "n_col_lbfgs": a.n_col, "n_ic": 1024}}
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, res = train_pinn.train(cfg, "check", "n_extrap", a.seed, dev, log=lambda m: None)
    tau, _ = tau_grid(grid["n_tau"], grid["tau_max"])
    xi, U = ssfm(gaussian(tau), tau, 1, N0, 4.0, 201, 2e-4)
    P = train_pinn.predict(model, "check", N0, xi, tau, dev)
    err = float(np.linalg.norm(P - U) / np.linalg.norm(U))
    out = {"rel_l2": err, "wall_s": res.wall_s, "adam": a.adam, "lbfgs": a.lbfgs, "n_col": a.n_col, "seed": a.seed, "tau_max": a.tau_max,
           "final_loss": res.history["loss"][-1]}
    print(out)
    if not a.smoke:
        p = ROOT / "results" / "pinn_check_fig3.json"
        prev = json.loads(p.read_text()) if p.exists() else []
        p.write_text(json.dumps(prev + [out], indent=2))


if __name__ == "__main__":
    main()
