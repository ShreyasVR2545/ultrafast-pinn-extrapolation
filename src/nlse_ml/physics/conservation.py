"""Conserved quantities of the lossless normalized NLSE, evaluated spectrally on a periodic grid.

E = int |U|^2 dtau
P = (i/2) int (U U*_tau - U* U_tau) dtau
H = int [ (s/2)|U_tau|^2 + (N^2/2)|U|^4 ] dtau
(H is -1 times the functional with i U_xi = dH/dU*; the sign does not affect conservation.)
All functions accept U with shape (..., n_tau).
"""
from __future__ import annotations

import numpy as np


def _dtau(tau):
    return tau[1] - tau[0]


def _deriv(U, tau):
    omega = 2.0 * np.pi * np.fft.fftfreq(tau.size, d=_dtau(tau))
    return np.fft.ifft(1j * omega * np.fft.fft(U, axis=-1), axis=-1)


def energy(U, tau):
    return np.sum(np.abs(U) ** 2, axis=-1) * _dtau(tau)


def momentum(U, tau):
    Ut = _deriv(U, tau)
    return np.real(0.5j * np.sum(U * np.conj(Ut) - np.conj(U) * Ut, axis=-1)) * _dtau(tau)


def hamiltonian(U, tau, s, N):
    Ut = _deriv(U, tau)
    return np.sum(0.5 * s * np.abs(Ut) ** 2 + 0.5 * N**2 * np.abs(U) ** 4, axis=-1) * _dtau(tau)


def relative_drift(q):
    """|q(xi) - q(0)| / |q(0)| (absolute drift if q(0) == 0)."""
    q = np.asarray(q)
    ref = np.abs(q[0])
    return np.abs(q - q[0]) / (ref if ref > 0 else 1.0)
