"""Parametric PINN for the normalized NLSE: network (xi, tau, N) -> (u, v), U = u + i v.

Residuals (derived in physics/nlse.py, tested on the analytic GVD solution):
    f = -v_xi - (s/2) u_tautau + N^2 (u^2 + v^2) u
    g =  u_xi - (s/2) v_tautau + N^2 (u^2 + v^2) v

Two derivative back-ends give identical results (tests/test_residuals.py):
  "autograd": nested torch.autograd.grad with create_graph=True (reference)
  "taylor":   exact forward-mode propagation via model.forward_derivs (about 1.5x faster)
"""
from __future__ import annotations

import torch


def _grad(y, x):
    return torch.autograd.grad(y, x, grad_outputs=torch.ones_like(y), create_graph=True)[0]


def _assemble(y, y_xi, y_tt, Nv, s):
    u, v = y[:, 0:1], y[:, 1:2]
    amp2 = u**2 + v**2
    N2 = Nv**2
    f = -y_xi[:, 1:2] - 0.5 * s * y_tt[:, 0:1] + N2 * amp2 * u
    g = y_xi[:, 0:1] - 0.5 * s * y_tt[:, 1:2] + N2 * amp2 * v
    return f, g


def field_fn_residual(field_fn, xi, tau, Nv, s: int):
    """Autograd residual of any differentiable map (xi, tau, N) -> (u, v) of shape (B, 2).

    xi, tau, Nv: tensors of shape (B, 1). Returns f, g of shape (B, 1).
    """
    xi = xi.detach().requires_grad_(True)
    tau = tau.detach().requires_grad_(True)
    out = field_fn(torch.cat([xi, tau, Nv], dim=1))
    u, v = out[:, 0:1], out[:, 1:2]
    u_t, v_t = _grad(u, tau), _grad(v, tau)
    y_xi = torch.cat([_grad(u, xi), _grad(v, xi)], 1)
    y_tt = torch.cat([_grad(u_t, tau), _grad(v_t, tau)], 1)
    return _assemble(out, y_xi, y_tt, Nv, s)


def residual(model, x, s: int, mode: str = "taylor"):
    """x: (B, 3) columns (xi, tau, N). Returns f, g of shape (B, 1)."""
    if mode == "autograd":
        return field_fn_residual(model, x[:, 0:1], x[:, 1:2], x[:, 2:3], s)
    y, y_xi, _, y_tt = model.forward_derivs(x)
    return _assemble(y, y_xi, y_tt, x[:, 2:3], s)


def residual_loss(model, xi, tau, Nv, s: int, mode: str = "autograd"):
    f, g = residual(model, torch.cat([xi, tau, Nv], 1), s, mode)
    return (f**2).mean() + (g**2).mean()
