"""Spectral-shift quantities for analysis A (extension of Fesser et al. 2023 to parametric NLSE).

wasserstein_fourier(S1, S2, omega): 1-D Wasserstein-1 distance between L1-normalized power spectra,
computed from CDFs:  W = sum_k |F1(omega_k) - F2(omega_k)| * d_omega. (Fesser et al. use the
unit-spacing sum; multiplying by d_omega gives the distance in units of omega T0, so a pure
frequency shift by delta gives W = delta.)

spm_phase(N, xi, pulse): closed-form, label-free proxy for the accumulated SPM phase (a driver of spectral
broadening). Note: as an ERROR predictor it does not beat coordinate-only baselines such as N^2*xi (D-031).
  phi_max = N^2 xi                        (no dispersion)
  phi_eff = N^2 asinh(xi)                 (peak power of a GVD-broadened Gaussian is (1+xi^2)^(-1/2),
                                           so int_0^xi P_peak dxi' = asinh(xi); used as the
                                           dispersion-scaled version for both pulse shapes)
"""
from __future__ import annotations

import numpy as np

from .metrics import omega_axis, power_spectrum


def wasserstein_fourier(S1, S2, omega, axis=-1):
    S1 = np.asarray(S1, float)
    S2 = np.asarray(S2, float)
    dw = omega[1] - omega[0]
    F1 = np.cumsum(S1 / S1.sum(axis=axis, keepdims=True), axis=axis)
    F2 = np.cumsum(S2 / S2.sum(axis=axis, keepdims=True), axis=axis)
    return np.sum(np.abs(F1 - F2), axis=axis) * dw


def wf_fields(U1, U2, tau):
    """Per-xi-slice WF distance between the spectra of two fields of shape (n_xi, n_tau)."""
    return wasserstein_fourier(power_spectrum(U1), power_spectrum(U2), omega_axis(tau))


def wf_from_input(U, tau):
    """WF distance between U(0) and U(xi) for every slice: how far the spectrum has moved."""
    S = power_spectrum(U)
    return wasserstein_fourier(np.broadcast_to(S[0], S.shape), S, omega_axis(tau))


def spm_phase(N, xi, dispersion_scaled: bool = True):
    xi = np.asarray(xi, float)
    return N**2 * (np.arcsinh(xi) if dispersion_scaled else xi)
