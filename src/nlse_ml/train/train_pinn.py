"""Parametric PINN family: pinn_strict, pinn_extended, pinn_plus.

Loss = w_ic * [mean (u - u0)^2 + mean (v - v0)^2]  +  w_r * [mean f^2 + mean g^2]
* IC points (xi = xi_lo, tau, N): tau and N by Latin hypercube. The input pulse is N-independent,
  so the IC is exact at any N without any SSFM label.
* Collocation (xi, tau, N): Latin hypercube over the training box, resampled every K Adam steps;
  one fixed batch for L-BFGS.
* strict: N in the train range. extended: N spans train + test ranges (physics only, no labels).
* pinn_plus adds (each toggleable): periodic Fourier features in tau, causal weights in xi
  (Wang, Sankaran & Perdikaris; weights detached, epsilon annealed [1e-2 .. 1e2], stage advances
  when min w > 0.99), and gradient-norm loss balancing (EMA 0.9, every `adapt_every` steps).
  During L-BFGS the adaptive weights are frozen; causal weighting continues at the reached epsilon
  only if plus.causal_lbfgs is true.
"""
from __future__ import annotations

import numpy as np
import torch

from ..data import splits
from ..models.mlp import FourierMLP, MLP
from ..models.pinn import residual
from ..physics.nlse import initial_condition
from .common import Schedule, fit, lhs


def focus_tau(tau, focus, tau_max, rng):
    """Optional sampling-density change: replace a fraction `frac` of the tau samples by draws from a
    centred normal of width `sigma` (clipped to the window). None -> uniform (unchanged)."""
    if not focus:
        return tau
    k = rng.random(tau.size) < focus["frac"]
    tau = tau.copy()
    tau[k] = np.clip(rng.normal(0.0, focus["sigma"], k.sum()), -tau_max, tau_max)
    return tau


def build_model(cfg: dict, regime: str, exp: str, device) -> torch.nn.Module:
    g = splits.load_grid(regime)
    xi_lo, xi_hi = splits.xi_train_range(exp)
    n_lo, n_hi = splits.collocation_n_range(regime, cfg.get("collocation", "strict"))
    lb, ub = [xi_lo, -g["tau_max"], n_lo], [xi_hi, g["tau_max"], n_hi]
    a = cfg["arch"]
    ff = cfg.get("plus", {}).get("fourier_features", False)
    if ff:
        m = FourierMLP(3, 2, a["width"], a["depth"], a["act"], lb, ub,
                       n_harmonics=cfg["plus"]["n_harmonics"], period=2 * g["tau_max"])
    else:
        m = MLP(3, 2, a["width"], a["depth"], a["act"], lb, ub)
    return m.to(device)


class PINNLoss:
    def __init__(self, model, cfg, regime, exp, device, seed):
        self.m, self.cfg, self.dev = model, cfg, device
        self.rc = splits.regime_cfg(regime)
        g = splits.load_grid(regime)
        self.s = self.rc["s"]
        self.xi_lo, self.xi_hi = splits.xi_train_range(exp)
        self.n_lo, self.n_hi = splits.collocation_n_range(regime, cfg.get("collocation", "strict"))
        self.tau_max = g["tau_max"]
        self.rng = np.random.default_rng(seed)
        self.w_ic = float(cfg["loss"]["w_ic"])
        self.w_r = float(cfg["loss"].get("w_r", 1.0))
        p = cfg.get("plus", {})
        self.causal = p.get("causal", False)
        self.causal_lbfgs = p.get("causal_lbfgs", False)  # keep causal weights (at the reached epsilon) in L-BFGS
        self.eps_list = list(p.get("causal_eps", [1e-2, 1e-1, 1.0, 10.0, 100.0]))
        self.n_bins = int(p.get("causal_bins", 32))
        self.eps_stage, self.stage_steps = 0, 0
        self.stage_len = max(1, cfg["schedule"]["adam_steps"] // len(self.eps_list))
        self.adaptive = p.get("adaptive_weights", False)
        self.adapt_every = int(p.get("adapt_every", 100))
        self.min_w = 1.0
        self._adapt_due = self.adaptive

    def sample(self, phase):
        c = self.cfg["sampling"]
        n_col = c["n_col_lbfgs"] if phase == "lbfgs" else c["n_col"]
        X = lhs(n_col, [self.xi_lo, -self.tau_max, self.n_lo], [self.xi_hi, self.tau_max, self.n_hi], self.rng)
        X[:, 1] = focus_tau(X[:, 1], c.get("tau_focus"), self.tau_max, self.rng)
        T = lhs(c["n_ic"], [-self.tau_max, self.n_lo], [self.tau_max, self.n_hi], self.rng)
        T[:, 0] = focus_tau(T[:, 0], c.get("tau_focus"), self.tau_max, self.rng)
        u0 = initial_condition(self.rc["pulse"], T[:, 0])
        Xic = np.column_stack([np.full(len(T), self.xi_lo), T])
        f = lambda a: torch.as_tensor(a, dtype=torch.float32, device=self.dev)
        return f(X), f(Xic), f(np.column_stack([u0.real, u0.imag]))

    def residual_terms(self, X):
        fr, gr = residual(self.m, X, self.s, self.cfg.get("deriv_mode", "taylor"))
        return fr**2 + gr**2

    def __call__(self, batch, phase):
        X, Xic, Y0 = batch
        r2 = self.residual_terms(X)
        L_ic = ((self.m(Xic) - Y0) ** 2).mean(0).sum()
        parts = {}
        if self.causal and (phase == "adam" or self.causal_lbfgs):
            eps = self.eps_list[self.eps_stage]
            idx = ((X[:, 0] - self.xi_lo) / (self.xi_hi - self.xi_lo) * self.n_bins).long().clamp(0, self.n_bins - 1)
            Lb = torch.zeros(self.n_bins, device=X.device).scatter_add_(0, idx, r2[:, 0])
            cnt = torch.zeros(self.n_bins, device=X.device).scatter_add_(0, idx, torch.ones_like(r2[:, 0]))
            Lb = Lb / cnt.clamp_min(1)
            with torch.no_grad():
                w = torch.exp(-eps * (torch.cumsum(Lb, 0) - Lb))
            L_r = (w * Lb).mean()
            self.min_w = float(w.min())
            parts["eps"] = eps
            parts["min_w"] = self.min_w
        else:
            L_r = r2.mean()
        if self._adapt_due and phase == "adam":
            self._update_weights(L_ic, L_r)
        parts.update({"ic": L_ic.item(), "res": L_r.item()})
        if self.adaptive:
            parts["w_ic"] = self.w_ic
        return self.w_ic * L_ic + self.w_r * L_r, parts

    def _update_weights(self, L_ic, L_r):
        params = [p for p in self.m.parameters() if p.requires_grad]
        g_ic = torch.autograd.grad(L_ic, params, retain_graph=True, allow_unused=True)
        g_r = torch.autograd.grad(L_r, params, retain_graph=True, allow_unused=True)
        n_ic = torch.sqrt(sum((g**2).sum() for g in g_ic if g is not None)).item()
        n_r = torch.sqrt(sum((g**2).sum() for g in g_r if g is not None)).item()
        tot = n_ic + n_r
        a = 0.9
        self.w_ic = a * self.w_ic + (1 - a) * tot / max(n_ic, 1e-12)
        self.w_r = a * self.w_r + (1 - a) * tot / max(n_r, 1e-12)
        self._adapt_due = False

    def on_step(self, step, phase, parts):
        if self.adaptive and (step + 1) % self.adapt_every == 0:
            self._adapt_due = True
        if self.causal:
            self.stage_steps += 1
            # Advance epsilon ONLY when causality is resolved at the current one (min w > 0.99), as in
            # Wang et al. A forced, timer-based advance (used in an earlier pilot) drove min w to ~1e-20
            # with a 2k-step Adam phase and froze training (DECISIONS D-027).
            force = self.cfg.get("plus", {}).get("causal_force_advance", False) and self.stage_steps >= self.stage_len
            if self.eps_stage < len(self.eps_list) - 1 and (self.min_w > 0.99 or force):
                self.eps_stage += 1
                self.stage_steps = 0


def train(cfg: dict, regime: str, exp: str, seed: int, device, log=print):
    torch.manual_seed(seed)
    model = build_model(cfg, regime, exp, device)
    lossf = PINNLoss(model, cfg, regime, exp, device, seed)
    res = fit(model, lossf, lossf.sample, Schedule(**cfg["schedule"]), log=log, on_step=lossf.on_step)
    res.history["final_weights"] = {"w_ic": lossf.w_ic, "w_r": lossf.w_r}
    return model, res


@torch.no_grad()
def predict(model, regime: str, N: float, xi: np.ndarray, tau: np.ndarray, device, chunk=262144):
    X, T = np.meshgrid(xi, tau, indexing="ij")
    pts = np.column_stack([X.ravel(), T.ravel(), np.full(X.size, N)])
    out = []
    for i in range(0, len(pts), chunk):
        out.append(model(torch.as_tensor(pts[i:i + chunk], dtype=torch.float32, device=device)).cpu().numpy())
    y = np.concatenate(out)
    return (y[:, 0] + 1j * y[:, 1]).reshape(X.shape)


def residual_rms(model, regime: str, N: float, xi_range, n: int, device, seed=12345):
    """RMS of the autograd PDE residual on random held-out points (any MLP model, incl. DD-MLP)."""
    g = splits.load_grid(regime)
    rng = np.random.default_rng(seed)
    X = np.column_stack([rng.uniform(*xi_range, n), rng.uniform(-g["tau_max"], g["tau_max"], n), np.full(n, N)])
    X = torch.as_tensor(X, dtype=torch.float32, device=device)
    s = splits.regime_cfg(regime)["s"]
    out = []
    for i in range(0, n, 20000):
        f, gg = residual(model, X[i:i + 20000], s, "taylor")
        out.append((f**2 + gg**2).detach().cpu())
    return float(torch.cat(out).mean().sqrt())
