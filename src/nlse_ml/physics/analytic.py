"""Closed-form solutions of the normalized NLSE (conventions in nlse.py)."""
from __future__ import annotations

import numpy as np


def gvd_gaussian(xi: np.ndarray, tau: np.ndarray, s: int) -> np.ndarray:
    """N = 0, U(0, tau) = exp(-tau^2/2).

    U(xi, tau) = (1 - i s xi)^(-1/2) exp(-tau^2 / (2 (1 - i s xi)))   (Agrawal Eq. 3.2.7, normalized).
    Intensity FWHM broadens by sqrt(1 + xi^2) for either sign of s.
    Returns array of shape (len(xi), len(tau)).
    """
    q = 1.0 - 1j * s * np.asarray(xi, dtype=float)[:, None]
    return q**-0.5 * np.exp(-tau[None, :] ** 2 / (2.0 * q))


def fundamental_soliton(xi: np.ndarray, tau: np.ndarray) -> np.ndarray:
    """s = -1, N = 1: U = sech(tau) exp(i xi / 2)."""
    xi = np.asarray(xi, dtype=float)[:, None]
    return (1.0 / np.cosh(tau[None, :])) * np.exp(0.5j * xi)


def spm_only(xi: np.ndarray, u0: np.ndarray, N: float) -> np.ndarray:
    """No dispersion: U(xi) = U0 exp(i N^2 |U0|^2 xi)."""
    xi = np.asarray(xi, dtype=float)[:, None]
    return u0[None, :] * np.exp(1j * N**2 * np.abs(u0[None, :]) ** 2 * xi)
