"""Evaluation metrics. Fields are complex arrays of shape (n_xi, n_tau) on the validated grid.

Definitions (all model-agnostic, computed on the saved (xi, tau) grid):
  rel_l2           ||P - T|| / ||T||                     (complex field; full grid or one slice)
  rel_l2_intensity || |P|^2 - |T|^2 || / || |T|^2 ||
  phase_err        sum |T|^2 |wrap(arg P - arg T)| / sum |T|^2      (intensity weighted, radians)
  spectral_err     || D(P) - D(T) || / || D(T) ||, D = max(10 log10(S / max S_T), -60) + 60
                   (power spectrum in dB relative to the TRUE peak, floored at -60 dB, shifted to >= 0)
  observables      peak power, intensity FWHM, RMS spectral width, -20 dB bandwidth,
                   compression factor max_xi(peak)/peak(0) and its xi position
  physics          |E(xi)-E(0)|/E(0), |H(xi)-H(0)|/|H(0)|, |P(xi)-P(0)| (absolute: P(0)=0 here),
                   grid PDE residual RMS: r = i U_xi - (s/2) U_tautau + N^2 |U|^2 U with 4th-order
                   central differences in xi and spectral derivatives in tau. The SSFM truth's own
                   residual is reported alongside as the discretization floor.
"""
from __future__ import annotations

import numpy as np

from ..physics import conservation as C

DB_FLOOR = -60.0


def rel_l2(pred, true, axis=None):
    num = np.sqrt(np.sum(np.abs(pred - true) ** 2, axis=axis))
    den = np.sqrt(np.sum(np.abs(true) ** 2, axis=axis))
    return num / den


def rel_l2_intensity(pred, true, axis=None):
    return rel_l2(np.abs(pred) ** 2, np.abs(true) ** 2, axis=axis)


def phase_err(pred, true, axis=None):
    w = np.abs(true) ** 2
    dphi = np.abs(np.angle(pred * np.conj(true)))  # wrapped to [0, pi]
    return np.sum(w * dphi, axis=axis) / np.sum(w, axis=axis)


def power_spectrum(U):
    return np.abs(np.fft.fftshift(np.fft.fft(U, axis=-1), axes=-1)) ** 2


def omega_axis(tau):
    return np.fft.fftshift(2 * np.pi * np.fft.fftfreq(tau.size, tau[1] - tau[0]))


def spectral_err(pred, true, axis=None):
    St, Sp = power_spectrum(true), power_spectrum(pred)
    ref = St.max(axis=-1, keepdims=True)

    def D(S):
        return np.maximum(10 * np.log10(np.maximum(S / ref, 1e-300)), DB_FLOOR) - DB_FLOOR

    dt, dp = D(St), D(Sp)
    return np.sqrt(np.sum((dp - dt) ** 2, axis=axis)) / np.sqrt(np.sum(dt**2, axis=axis))


# ---------------- observables (per xi slice) ----------------
def peak_power(U):
    return np.max(np.abs(U) ** 2, axis=-1)


def _width_at(y, x, frac):
    """Full width at y = frac * max(y), linear interpolation at the outermost crossings."""
    out = np.empty(y.shape[0])
    for k, row in enumerate(y):
        thr = frac * row.max()
        idx = np.nonzero(row >= thr)[0]
        if idx.size == 0 or row.max() <= 0:
            out[k] = np.nan
            continue
        i0, i1 = idx[0], idx[-1]

        def cross(a, b):
            if a < 0 or b >= row.size:
                return x[min(max(a, 0), row.size - 1)]
            return x[a] + (thr - row[a]) * (x[b] - x[a]) / (row[b] - row[a])

        left = cross(i0 - 1, i0) if i0 > 0 else x[0]
        right = cross(i1, i1 + 1) if i1 < row.size - 1 else x[-1]
        out[k] = right - left
    return out


def fwhm(U, tau):
    return _width_at(np.abs(U) ** 2, tau, 0.5)


def bandwidth_db(U, tau, db=-20.0):
    return _width_at(power_spectrum(U), omega_axis(tau), 10 ** (db / 10))


def rms_spectral_width(U, tau):
    S = power_spectrum(U)
    w = omega_axis(tau)
    m1 = (S * w).sum(-1) / S.sum(-1)
    return np.sqrt((S * (w - m1[:, None]) ** 2).sum(-1) / S.sum(-1))


def compression(U, xi):
    pk = peak_power(U)
    k = int(np.argmax(pk))
    return float(pk[k] / pk[0]), float(xi[k])


# ---------------- physics consistency ----------------
def energy_drift(U, tau):
    return C.relative_drift(C.energy(U, tau))


def hamiltonian_drift(U, tau, s, N):
    return C.relative_drift(C.hamiltonian(U, tau, s, N))


def momentum_drift(U, tau):
    P = C.momentum(U, tau)
    return np.abs(P - P[0])


def grid_residual(U, xi, tau, s, N):
    """RMS over interior slices of r = i U_xi - (s/2) U_tt + N^2 |U|^2 U, normalized by RMS |U|."""
    h = xi[1] - xi[0]
    Ux = (-U[4:] + 8 * U[3:-1] - 8 * U[1:-3] + U[:-4]) / (12 * h)
    Uc = U[2:-2]
    w = 2 * np.pi * np.fft.fftfreq(tau.size, tau[1] - tau[0])
    Utt = np.fft.ifft(-(w**2) * np.fft.fft(Uc, axis=-1), axis=-1)
    r = 1j * Ux - 0.5 * s * Utt + N**2 * np.abs(Uc) ** 2 * Uc
    return float(np.sqrt(np.mean(np.abs(r) ** 2)) / np.sqrt(np.mean(np.abs(Uc) ** 2)))


# ---------------- bundle ----------------
def field_metrics(pred, true, xi, tau, s, N) -> dict:
    """All scalar metrics for one (model, N) prediction over the given xi slices."""
    m = {
        "rel_l2": float(rel_l2(pred, true)),
        "rel_l2_final": float(rel_l2(pred[-1], true[-1])),
        "rel_l2_intensity": float(rel_l2_intensity(pred, true)),
        "phase_err": float(phase_err(pred, true)),
        "phase_err_final": float(phase_err(pred[-1], true[-1])),
        "spectral_err": float(spectral_err(pred, true)),
        "spectral_err_final": float(spectral_err(pred[-1], true[-1])),
        "energy_drift_max": float(energy_drift(pred, tau).max()),
        "hamiltonian_drift_max": float(hamiltonian_drift(pred, tau, s, N).max()),
        "momentum_drift_max": float(momentum_drift(pred, tau).max()),
    }
    if xi.size >= 5:
        m["grid_residual"] = grid_residual(pred, xi, tau, s, N)
        m["grid_residual_truth"] = grid_residual(true, xi, tau, s, N)
    for name, fn in [("peak_power", lambda u: peak_power(u)), ("fwhm", lambda u: fwhm(u, tau)),
                     ("rms_bw", lambda u: rms_spectral_width(u, tau)), ("bw20", lambda u: bandwidth_db(u, tau))]:
        p, t = fn(pred[-1:])[0], fn(true[-1:])[0]
        m[f"{name}_final_pred"], m[f"{name}_final_true"] = float(p), float(t)
        m[f"{name}_final_relerr"] = float(abs(p - t) / abs(t)) if t else float("nan")
    cp, xp = compression(pred, xi)
    ct, xt = compression(true, xi)
    m.update({"compression_pred": cp, "compression_true": ct, "xi_maxcomp_pred": xp, "xi_maxcomp_true": xt})
    return m


def per_slice(pred, true, tau, s, N) -> dict:
    """Per-xi arrays used for error-vs-xi plots and the drift/error analyses."""
    return {
        "rel_l2": rel_l2(pred, true, axis=-1),
        "spectral_err": spectral_err(pred, true, axis=-1),
        "phase_err": phase_err(pred, true, axis=-1),
        "energy_drift": energy_drift(pred, tau),
        "hamiltonian_drift": hamiltonian_drift(pred, tau, s, N),
    }
