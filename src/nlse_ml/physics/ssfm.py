"""Reference solvers for the normalized NLSE (float64): symmetric SSFM and RK4IP.

Both integrate  dU/dxi = -i (s/2) U_tautau + i N^2 |U|^2 U  on a periodic tau grid
(see nlse.py for conventions) and return the field at n_save equally spaced xi slices.
"""
from __future__ import annotations

import math

import numpy as np

from .nlse import tau_grid

SOLVER_VERSION = "ssfm-v1"


def _segments(xi_max: float, n_save: int, dxi: float) -> tuple[np.ndarray, int, float]:
    xi = np.linspace(0.0, xi_max, n_save)
    seg = xi_max / (n_save - 1)
    m = max(1, math.ceil(seg / dxi - 1e-12))
    return xi, m, seg / m


def ssfm(u0: np.ndarray, tau: np.ndarray, s: int, N: float, xi_max: float, n_save: int, dxi: float):
    """Symmetric (Strang) split-step: L(h/2) NL(h) L(h/2), half-steps merged inside a save interval.

    The nonlinear sub-flow U -> U exp(i N^2 |U|^2 h) is exact, so the scheme is unitary
    (energy conserved to round-off) and second order in h.
    Returns xi (n_save,), U (n_save, n_tau) complex128.
    """
    xi, m, h = _segments(xi_max, n_save, dxi)
    n = tau.size
    omega = 2.0 * np.pi * np.fft.fftfreq(n, d=tau[1] - tau[0])
    lin_half = np.exp(0.5j * s * omega**2 * (h / 2))
    lin_full = lin_half * lin_half
    g = N**2 * h
    out = np.empty((n_save, n), dtype=np.complex128)
    u = np.asarray(u0, dtype=np.complex128).copy()
    out[0] = u
    for k in range(1, n_save):
        uh = np.fft.fft(u) * lin_half
        for j in range(m):
            u = np.fft.ifft(uh)
            u = u * np.exp(1j * g * (u.real**2 + u.imag**2))
            uh = np.fft.fft(u) * (lin_full if j < m - 1 else lin_half)
        u = np.fft.ifft(uh)
        out[k] = u
    return xi, out


def rk4ip(u0: np.ndarray, tau: np.ndarray, s: int, N: float, xi_max: float, n_save: int, dxi: float):
    """Fourth-order Runge-Kutta in the interaction picture (Hult, J. Lightwave Technol. 25, 3770, 2007)."""
    xi, m, h = _segments(xi_max, n_save, dxi)
    n = tau.size
    omega = 2.0 * np.pi * np.fft.fftfreq(n, d=tau[1] - tau[0])
    D = np.exp(0.5j * s * omega**2 * (h / 2))  # exp(L h/2) in Fourier space

    def Nop(uh):  # nonlinear operator acting in Fourier space
        u = np.fft.ifft(uh)
        return np.fft.fft(1j * N**2 * (u.real**2 + u.imag**2) * u)

    out = np.empty((n_save, n), dtype=np.complex128)
    uh = np.fft.fft(np.asarray(u0, dtype=np.complex128))
    out[0] = u0
    for k in range(1, n_save):
        for _ in range(m):
            ui = D * uh
            k1 = D * (h * Nop(uh))
            k2 = h * Nop(ui + k1 / 2)
            k3 = h * Nop(ui + k2 / 2)
            k4 = h * Nop(D * (ui + k3))
            uh = D * (ui + k1 / 6 + k2 / 3 + k3 / 3) + k4 / 6
        out[k] = np.fft.ifft(uh)
    return xi, out


def solve(shape: str, s: int, N: float, grid: dict, method: str = "ssfm"):
    """Convenience wrapper: grid = {n_tau, tau_max, xi_max, n_save, dxi}."""
    from .nlse import initial_condition

    tau, _ = tau_grid(grid["n_tau"], grid["tau_max"])
    u0 = initial_condition(shape, tau)
    fn = {"ssfm": ssfm, "rk4ip": rk4ip}[method]
    xi, U = fn(u0, tau, s, N, grid["xi_max"], grid["n_save"], grid["dxi"])
    return xi, tau, U
