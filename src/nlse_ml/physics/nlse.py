"""Normalized NLSE conventions (Agrawal), shared by every module.

    i dU/dxi = (s/2) d2U/dtau2 - N^2 |U|^2 U

U = A/sqrt(P0), xi = z/L_D, tau = T/T0, s = sgn(beta2), L_D = T0^2/|beta2|,
L_NL = 1/(gamma P0), N^2 = L_D/L_NL = gamma P0 T0^2/|beta2|.
s = +1: normal dispersion, s = -1: anomalous dispersion.

Equivalent evolution form:  dU/dxi = -i (s/2) U_tautau + i N^2 |U|^2 U.
In Fourier space (any sign convention, since d2/dtau2 -> -omega^2) the linear part is
    dU^/dxi = +i (s/2) omega^2 U^   =>   U^(xi) = exp(i s omega^2 xi / 2) U^(0).

Real/imaginary residuals for U = u + i v (derived by hand, tested in tests/test_residuals.py):
    i(u_xi + i v_xi) = (s/2)(u_tt + i v_tt) - N^2 (u^2+v^2)(u + i v)
    real part:  -v_xi - (s/2) u_tt + N^2 (u^2+v^2) u = 0     =: f
    imag part:   u_xi - (s/2) v_tt + N^2 (u^2+v^2) v = 0     =: g
"""
from __future__ import annotations

import numpy as np


def tau_grid(n_tau: int, tau_max: float) -> tuple[np.ndarray, np.ndarray]:
    """Periodic grid tau in [-tau_max, tau_max) and the matching angular frequencies (FFT order)."""
    tau = np.linspace(-tau_max, tau_max, n_tau, endpoint=False)
    dtau = tau[1] - tau[0]
    omega = 2.0 * np.pi * np.fft.fftfreq(n_tau, d=dtau)
    return tau, omega


def gaussian(tau: np.ndarray) -> np.ndarray:
    return np.exp(-tau**2 / 2.0).astype(np.complex128)


def sech(tau: np.ndarray) -> np.ndarray:
    return (1.0 / np.cosh(tau)).astype(np.complex128)


PULSES = {"gaussian": gaussian, "sech": sech}


def initial_condition(shape: str, tau: np.ndarray) -> np.ndarray:
    return PULSES[shape](tau)


def physical_units(beta2_ps2_per_km: float, gamma_per_W_km: float, T0_ps: float, N: float) -> dict:
    """Map normalized (N, xi) to physical units for the report. beta2 in ps^2/km."""
    L_D = T0_ps**2 / abs(beta2_ps2_per_km)  # km
    P0 = N**2 * abs(beta2_ps2_per_km) / (gamma_per_W_km * T0_ps**2)  # W
    return {"L_D_km": L_D, "P0_W": P0, "L_NL_km": 1.0 / (gamma_per_W_km * P0)}
