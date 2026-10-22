"""DD-MLP: same network and optimizer schedule as the PINN, trained with MSE on SSFM fields at the
training N values (all grid points in the experiment's xi range). No PDE information.

Loss = mean (u - u_true)^2 + mean (v - v_true)^2 over a random batch of grid points
(batch size = the PINN's collocation batch size, resampled every K Adam steps; one fixed
batch for L-BFGS).
"""
from __future__ import annotations

import numpy as np
import torch

from ..data import splits
from ..data.datasets import load_training_fields
from ..models.mlp import MLP
from .common import Schedule, fit
from .train_pinn import focus_tau


def build_model(cfg, regime, exp, device):
    g = splits.load_grid(regime)
    xi_lo, xi_hi = splits.xi_train_range(exp)
    tr = splits.train_n(regime)
    a = cfg["arch"]
    return MLP(3, 2, a["width"], a["depth"], a["act"], [xi_lo, -g["tau_max"], min(tr)],
               [xi_hi, g["tau_max"], max(tr)]).to(device)


def train(cfg, regime, exp, seed, device, log=print, data_dir=None):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    kw = {} if data_dir is None else {"data_dir": data_dir}
    fields = load_training_fields(regime, exp, **kw)
    if cfg.get("n_train_subset"):  # data-efficiency ablation (subset of the train split only)
        fields = [f for f in fields if any(abs(f.N - n) < 1e-9 for n in cfg["n_train_subset"])]
    X, Y = [], []
    for f in fields:
        xx, tt = np.meshgrid(f.xi, f.tau, indexing="ij")
        X.append(np.column_stack([xx.ravel(), tt.ravel(), np.full(xx.size, f.N)]))
        Y.append(np.column_stack([f.U.real.ravel(), f.U.imag.ravel()]))
    X = torch.as_tensor(np.concatenate(X), dtype=torch.float32, device=device)
    Y = torch.as_tensor(np.concatenate(Y), dtype=torch.float32, device=device)
    log(f"DD-MLP data: {X.shape[0]} points from N={[f.N for f in fields]}, xi<= {float(X[:, 0].max()):.2f}")
    model = build_model(cfg, regime, exp, device)
    c = cfg["sampling"]

    n_tau = fields[0].tau.size
    tau_grid_np = fields[0].tau
    focus = c.get("tau_focus")

    def sample(phase):
        n = c["n_col_lbfgs"] if phase == "lbfgs" else c["n_col"]
        if not focus:
            idx = rng.choice(X.shape[0], n, replace=False)
        else:  # same tau sampling density as the PINN collocation (uniform/normal mixture)
            rows = rng.integers(0, X.shape[0] // n_tau, n)  # (field, xi) row
            t = rng.uniform(-tau_grid_np.max(), tau_grid_np.max(), n)
            t = focus_tau(t, focus, float(-tau_grid_np[0]), rng)
            j = np.clip(np.searchsorted(tau_grid_np, t), 0, n_tau - 1)
            idx = rows * n_tau + j
        idx = torch.as_tensor(idx, device=device)
        return X[idx], Y[idx]

    def loss_fn(batch, phase):
        xb, yb = batch
        L = ((model(xb) - yb) ** 2).mean(0).sum()
        return L, {"data": L.item()}

    res = fit(model, loss_fn, sample, Schedule(**cfg["schedule"]), log=log)
    return model, res
