"""PINN residuals vanish on exact solutions; Taylor-mode derivatives equal nested autograd."""
import pytest
import torch

from nlse_ml.models.mlp import MLP
from nlse_ml.models.pinn import field_fn_residual, residual


def _gvd(x, s):
    xi, tau = x[:, 0:1], x[:, 1:2]
    q = torch.complex(torch.ones_like(xi), -s * xi)
    U = q**-0.5 * torch.exp(-tau.to(q.dtype) ** 2 / (2 * q))
    return torch.cat([U.real, U.imag], 1)


def _soliton(x):  # s = -1, N = 1: U = sech(tau) exp(i xi / 2)
    xi, tau = x[:, 0:1], x[:, 1:2]
    a = 1.0 / torch.cosh(tau)
    return torch.cat([a * torch.cos(xi / 2), a * torch.sin(xi / 2)], 1)


def _pts(n=2000, seed=0):
    g = torch.Generator().manual_seed(seed)
    xi = torch.rand(n, 1, generator=g, dtype=torch.float64) * 4
    tau = (torch.rand(n, 1, generator=g, dtype=torch.float64) - 0.5) * 12
    return xi, tau


@pytest.mark.parametrize("s", [1, -1])
def test_residual_zero_on_gvd_gaussian(s):
    xi, tau = _pts()
    f, g = field_fn_residual(lambda x: _gvd(x, s), xi, tau, torch.zeros_like(xi), s)
    assert f.abs().max() < 1e-12 and g.abs().max() < 1e-12
    # wrong dispersion sign must give an O(1) residual (guards against sign errors)
    f, g = field_fn_residual(lambda x: _gvd(x, -s), xi, tau, torch.zeros_like(xi), s)
    assert f.abs().max() > 0.1


def test_residual_zero_on_fundamental_soliton():
    xi, tau = _pts()
    f, g = field_fn_residual(_soliton, xi, tau, torch.ones_like(xi), -1)
    assert f.abs().max() < 1e-12 and g.abs().max() < 1e-12
    # wrong N or wrong nonlinear sign is detected
    f, _ = field_fn_residual(_soliton, xi, tau, 1.2 * torch.ones_like(xi), -1)
    assert f.abs().max() > 0.1


@pytest.mark.parametrize("act", ["tanh", "sin"])
def test_taylor_mode_equals_autograd(act):
    torch.manual_seed(0)
    m = MLP(3, 2, 64, 4, act, lb=[0, -10, 1], ub=[2, 10, 2]).double()
    xi, tau = _pts(500)
    Nv = 1 + torch.rand_like(xi)
    x = torch.cat([xi, tau, Nv], 1)
    for s in (1, -1):
        fa, ga = residual(m, x, s, "autograd")
        ft, gt = residual(m, x, s, "taylor")
        assert torch.allclose(fa, ft, atol=1e-12) and torch.allclose(ga, gt, atol=1e-12)
