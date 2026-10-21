"""cFNO training (direct and autoregressive). Same total optimizer-step budget as the MLP models
(Adam only: adam_steps + lbfgs_steps of the shared schedule). Loss: mean relative L2 per sample
(standard for neural operators, insensitive to the N-dependent field amplitude).

Autoregressive training: one-step teacher forcing on pairs (xi_k -> xi_{k+1..k+K}) inside the
training xi range, with Gaussian input-noise injection (sigma = noise_std) for rollout stability.
"""
from __future__ import annotations

import numpy as np
import torch

from ..data import splits
from ..data.datasets import load_training_fields
from ..models.fno import CondFNO, StepFNO
from .common import Schedule, fit


def _ri(U):  # complex (..., n) -> real (..., 2, n)
    return np.stack([U.real, U.imag], -2).astype(np.float32)


def build_model(cfg, regime, exp, device):
    a = cfg["arch"]
    tr = splits.train_n(regime)
    xi_lo, xi_hi = splits.xi_train_range(exp)
    if cfg["variant"] == "direct":
        return CondFNO(a["n_modes"], a["hidden"], a["n_layers"], 4, 2, [min(tr), xi_lo], [max(tr), xi_hi]).to(device)
    return StepFNO(a["n_modes"], a["hidden"], a["n_layers"], cfg["k_out"], [min(tr)], [max(tr)]).to(device)


def train(cfg, regime, exp, seed, device, log=print, data_dir=None):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    kw = {} if data_dir is None else {"data_dir": data_dir}
    fields = load_training_fields(regime, exp, **kw)
    if cfg.get("n_train_subset"):
        fields = [f for f in fields if any(abs(f.N - n) < 1e-9 for n in cfg["n_train_subset"])]
    model = build_model(cfg, regime, exp, device)
    U = torch.as_tensor(np.stack([_ri(f.U) for f in fields]), device=device)  # (nN, n_xi, 2, n_tau)
    Ns = torch.as_tensor([f.N for f in fields], dtype=torch.float32, device=device)
    xi = torch.as_tensor(fields[0].xi, dtype=torch.float32, device=device)
    nN, nxi = U.shape[:2]
    bs, K = cfg["batch_size"], cfg.get("k_out", 1)
    log(f"cFNO-{cfg['variant']} data: N={[f.N for f in fields]}, {nxi} slices, xi<={float(xi.max()):.2f}")

    def sample(phase):
        i = torch.as_tensor(rng.integers(0, nN, bs), device=device)
        if cfg["variant"] == "direct":
            k = torch.as_tensor(rng.integers(0, nxi, bs), device=device)
            return U[i, 0], torch.stack([Ns[i], xi[k]], 1), U[i, k]
        k = torch.as_tensor(rng.integers(0, nxi - K, bs), device=device)
        tgt = torch.stack([U[i, k + j] for j in range(1, K + 1)], 1)
        inp = U[i, k] + cfg.get("noise_std", 0.0) * torch.randn_like(U[i, k])
        return inp, Ns[i][:, None], tgt

    def loss_fn(batch, phase):
        inp, cond, tgt = batch
        pred = model(inp, cond)
        d = (pred - tgt).flatten(1).norm(dim=1) / tgt.flatten(1).norm(dim=1)
        L = d.mean()
        return L, {"relL2": L.item()}

    s = dict(cfg["schedule"])
    s["adam_steps"] = s["adam_steps"] + s.pop("lbfgs_steps")
    s["lbfgs_steps"] = 0
    s["resample_every"] = 1
    res = fit(model, loss_fn, sample, Schedule(**s), log=log)
    return model, res


@torch.no_grad()
def predict(model, cfg, regime, N, xi_eval, tau, U0, device):
    """Returns complex (len(xi_eval), n_tau). AR rollout always starts from U0 at xi = 0 and
    xi_eval must lie on the saved grid (multiples of the slice spacing)."""
    model.eval()
    u0 = torch.as_tensor(_ri(U0)[None], device=device)
    if cfg["variant"] == "direct":
        out = []
        for i in range(0, len(xi_eval), 64):
            x = torch.as_tensor(xi_eval[i:i + 64], dtype=torch.float32, device=device)
            cond = torch.stack([torch.full_like(x, N), x], 1)
            y = model(u0.expand(len(x), -1, -1), cond)
            out.append(y.cpu().numpy())
        y = np.concatenate(out)
        return y[:, 0] + 1j * y[:, 1]
    g = splits.load_grid(regime)
    dxi = g["xi_max"] / (g["n_save"] - 1)
    K = cfg["k_out"]
    n_steps = int(round(xi_eval.max() / dxi))
    traj = [u0[0]]
    cur = u0
    cond = torch.tensor([[N]], dtype=torch.float32, device=device)
    while len(traj) <= n_steps:
        nxt = model(cur, cond)[0]  # (K, 2, n)
        traj.extend(list(nxt))
        cur = nxt[-1:].clone()
    traj = torch.stack(traj[: n_steps + 1]).cpu().numpy()
    idx = np.rint(xi_eval / dxi).astype(int)
    return traj[idx, 0] + 1j * traj[idx, 1]
