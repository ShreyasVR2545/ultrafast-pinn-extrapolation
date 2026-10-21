"""Shared fully connected network used by the PINN, PINN+ and the DD-MLP (identical trunk).

Inputs x = (xi, tau, N). The default embedding maps each input affinely to [-1, 1] using the
model's TRAINING input range (lb, ub passed at construction). Inputs outside that range (e.g.
extrapolation N, or xi > 2 in z-extrapolation) simply map outside [-1, 1]; nothing is
re-normalized at test time.
"""
from __future__ import annotations

import math

import torch
from torch import nn


class Sine(nn.Module):
    def forward(self, x):
        return torch.sin(x)


ACTS = {"tanh": nn.Tanh, "sin": Sine}


class MLP(nn.Module):
    I_XI, I_TAU = 0, 1

    def __init__(self, in_dim: int, out_dim: int, width: int, depth: int, act: str = "tanh",
                 lb=None, ub=None):
        super().__init__()
        lb = torch.as_tensor(lb if lb is not None else [-1.0] * in_dim, dtype=torch.float32)
        ub = torch.as_tensor(ub if ub is not None else [1.0] * in_dim, dtype=torch.float32)
        self.register_buffer("lb", lb)
        self.register_buffer("ub", ub)
        layers, d = [], self.embed_dim(in_dim)
        for _ in range(depth):
            layers += [nn.Linear(d, width), ACTS[act]()]
            d = width
        layers.append(nn.Linear(d, out_dim))
        self.net = nn.Sequential(*layers)
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    # ---- embedding (overridden by PINN+) ----
    def embed_dim(self, in_dim: int) -> int:
        return in_dim

    def normalize(self, x):
        return 2.0 * (x - self.lb) / (self.ub - self.lb) - 1.0

    def embed(self, x):
        return self.normalize(x)

    def embed_derivs(self, x):
        """Embedding and its derivatives w.r.t. tau (1st, 2nd) and xi (1st)."""
        scale = 2.0 / (self.ub - self.lb)
        z = self.normalize(x)
        z_t = torch.zeros_like(z)
        z_t[:, self.I_TAU] = scale[self.I_TAU]
        z_x = torch.zeros_like(z)
        z_x[:, self.I_XI] = scale[self.I_XI]
        return z, z_t, torch.zeros_like(z), z_x

    def forward(self, x):
        return self.net(self.embed(x))

    def forward_derivs(self, x):
        """Exact forward-mode (Taylor) propagation of d/dxi, d/dtau, d2/dtau2 through the network.

        Mathematically identical to nested autograd (tests/test_residuals.py) but training then
        needs only a first-order backward pass. Returns (y, y_xi, y_tau, y_tautau), each (B, out).
        """
        z, z_t, z_tt, z_x = self.embed_derivs(x)
        B = x.shape[0]
        mods = list(self.net)
        for k in range(0, len(mods) - 1, 2):
            lin, act = mods[k], mods[k + 1]
            a = lin(z)
            d = torch.cat([z_t, z_tt, z_x], 0) @ lin.weight.T
            a_t, a_tt, a_x = d[:B], d[B:2 * B], d[2 * B:]
            if isinstance(act, nn.Tanh):
                z = torch.tanh(a)
                s1 = 1.0 - z * z
                s2 = -2.0 * z * s1
            else:  # Sine
                z, s1 = torch.sin(a), torch.cos(a)
                s2 = -z
            z_t, z_x = s1 * a_t, s1 * a_x
            z_tt = s2 * a_t * a_t + s1 * a_tt
        lin = mods[-1]
        y = lin(z)
        d = torch.cat([z_t, z_tt, z_x], 0) @ lin.weight.T
        return y, d[2 * B:], d[:B], d[B:2 * B]


class FourierMLP(MLP):
    """PINN+ trunk: periodic Fourier-feature embedding in tau (Wang, Sankaran & Perdikaris):
    v(tau) = (cos(k w tau), sin(k w tau))_{k=1..m}, w = 2 pi / L with L the tau window, which makes
    the network exactly periodic in tau (matching the periodic SSFM domain). xi and N enter
    affinely normalized, as in the plain MLP.
    """

    def __init__(self, in_dim, out_dim, width, depth, act="tanh", lb=None, ub=None,
                 n_harmonics: int = 32, period: float = 2.0):
        self.n_harmonics = n_harmonics
        super().__init__(in_dim, out_dim, width, depth, act, lb, ub)
        self.w0 = 2 * math.pi / period  # python float: exact in any dtype
        self.register_buffer("n", torch.arange(1, n_harmonics + 1, dtype=torch.float32))

    def embed_dim(self, in_dim):
        return in_dim - 1 + 2 * self.n_harmonics

    def _parts(self, x):
        xn = self.normalize(x)
        other = torch.cat([xn[:, :self.I_TAU], xn[:, self.I_TAU + 1:]], 1)  # (xi, N)
        ph = (x[:, self.I_TAU:self.I_TAU + 1] * self.w0) * self.n  # raw tau times harmonics
        return other, ph

    def embed(self, x):
        other, ph = self._parts(x)
        return torch.cat([other, torch.cos(ph), torch.sin(ph)], 1)

    def embed_derivs(self, x):
        other, ph = self._parts(x)
        c, s = torch.cos(ph), torch.sin(ph)
        k = self.n * self.w0
        z = torch.cat([other, c, s], 1)
        zo = torch.zeros_like(other)
        z_t = torch.cat([zo, -k * s, k * c], 1)
        z_tt = torch.cat([zo, -(k**2) * c, -(k**2) * s], 1)
        ox = torch.zeros_like(other)
        ox[:, 0] = 2.0 / (self.ub[self.I_XI] - self.lb[self.I_XI])  # xi is column 0 of `other`
        z_x = torch.cat([ox, torch.zeros_like(c), torch.zeros_like(s)], 1)
        return z, z_t, z_tt, z_x
