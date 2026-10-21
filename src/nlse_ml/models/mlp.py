"""Shared fully connected network used by the PINN and the DD-MLP (identical architecture).

Inputs (xi, tau, N) are affinely mapped to [-1, 1] using TRAINING-RANGE bounds only
(lb, ub passed in at construction). Inputs outside the training range (e.g. extrapolation N,
or xi > 2 in z-extrapolation) simply map outside [-1, 1]; nothing is re-normalized at test time.
"""
from __future__ import annotations

import torch
from torch import nn

ACTS = {"tanh": nn.Tanh, "sin": None, "gelu": nn.GELU, "silu": nn.SiLU}


class Sine(nn.Module):
    def forward(self, x):
        return torch.sin(x)


class MLP(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, width: int, depth: int, act: str = "tanh",
                 lb=None, ub=None):
        super().__init__()
        lb = torch.as_tensor(lb if lb is not None else [-1.0] * in_dim, dtype=torch.float32)
        ub = torch.as_tensor(ub if ub is not None else [1.0] * in_dim, dtype=torch.float32)
        self.register_buffer("lb", lb)
        self.register_buffer("ub", ub)
        act_cls = Sine if act == "sin" else ACTS[act]
        layers, d = [], in_dim
        for _ in range(depth):
            layers += [nn.Linear(d, width), act_cls()]
            d = width
        layers.append(nn.Linear(d, out_dim))
        self.net = nn.Sequential(*layers)
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def normalize(self, x):
        return 2.0 * (x - self.lb) / (self.ub - self.lb) - 1.0

    def forward(self, x):
        return self.net(self.normalize(x))

    def forward_derivs(self, x, i_xi: int = 0, i_tau: int = 1):
        """Exact forward-mode (Taylor) propagation of d/dxi, d/dtau, d2/dtau2 through the network.

        Mathematically identical to nested autograd (tested in tests/test_residuals.py) but needs
        only a first-order backward pass for training, which is several times cheaper.
        Returns (y, y_xi, y_tau, y_tautau), each of shape (B, out_dim).
        """
        scale = 2.0 / (self.ub - self.lb)
        z = self.normalize(x)
        B = x.shape[0]
        z_t = torch.zeros_like(z)
        z_t[:, i_tau] = scale[i_tau]
        z_x = torch.zeros_like(z)
        z_x[:, i_xi] = scale[i_xi]
        z_tt = torch.zeros_like(z)
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
            elif isinstance(act, Sine):
                z, s1 = torch.sin(a), torch.cos(a)
                s2 = -z
            else:
                raise NotImplementedError("forward_derivs supports tanh and sin activations")
            z_t, z_x = s1 * a_t, s1 * a_x
            z_tt = s2 * a_t * a_t + s1 * a_tt
        lin = mods[-1]
        y = lin(z)
        d = torch.cat([z_t, z_tt, z_x], 0) @ lin.weight.T
        return y, d[2 * B:], d[:B], d[B:2 * B]
