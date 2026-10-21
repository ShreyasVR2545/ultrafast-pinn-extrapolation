"""Conditional 1-D Fourier neural operators (neuraloperator FNO trunk).

cfno (direct):  [Re U0, Im U0, N, xi] (each broadcast over tau) -> [Re U, Im U](xi).
    Conditioning on xi as an input channel lets one network produce every output slice and be
    queried at any xi. N and xi are normalized to [-1, 1] with TRAINING-range bounds.
cfno_ar (autoregressive step operator): [Re U(xi_k), Im U(xi_k), N] -> U at the next K slices
    (residual form: output = input slice + network), rolled out from U0 for z-extrapolation.

Note: in this study U0 is the same for every N (only N varies), so the operator's input function
carries no information about N; conditioning is entirely through the N channel.
"""
from __future__ import annotations

import torch
from neuralop.models import FNO
from torch import nn


class CondFNO(nn.Module):
    def __init__(self, n_modes, hidden, n_layers, in_ch, out_ch, cond_lb, cond_ub):
        super().__init__()
        self.fno = FNO(n_modes=(n_modes,), hidden_channels=hidden, in_channels=in_ch, out_channels=out_ch,
                       n_layers=n_layers)
        self.register_buffer("lb", torch.as_tensor(cond_lb, dtype=torch.float32))
        self.register_buffer("ub", torch.as_tensor(cond_ub, dtype=torch.float32))

    def norm(self, c):  # c: (B, n_cond)
        return 2 * (c - self.lb) / (self.ub - self.lb) - 1

    def forward(self, field, cond):
        """field: (B, 2, n_tau) real/imag; cond: (B, n_cond) raw values (N[, xi])."""
        c = self.norm(cond)[:, :, None].expand(-1, -1, field.shape[-1])
        return self.fno(torch.cat([field, c], 1))


class StepFNO(CondFNO):
    def __init__(self, n_modes, hidden, n_layers, k_out, cond_lb, cond_ub):
        super().__init__(n_modes, hidden, n_layers, 3, 2 * k_out, cond_lb, cond_ub)
        self.k_out = k_out

    def forward(self, field, cond):
        delta = super().forward(field, cond)
        B, _, n = field.shape
        return field[:, None] + delta.view(B, self.k_out, 2, n)  # (B, K, 2, n_tau)
