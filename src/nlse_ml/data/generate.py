"""Generate SSFM ground-truth fields for every N in every split on the validated grid.

File layout: data/<regime>/N_<value>.npz with arrays U (n_save, n_tau) complex128, xi, tau and a
JSON metadata string (grid, s, N, pulse, solver version, git hash, quality diagnostics).
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from ..physics import conservation as C
from ..physics.nlse import initial_condition, tau_grid
from ..physics.ssfm import SOLVER_VERSION, ssfm
from ..utils import REPO_ROOT, git_info
from . import splits

DATA_DIR = REPO_ROOT / "data"


def field_path(regime: str, N: float, data_dir: Path = DATA_DIR) -> Path:
    return Path(data_dir) / regime / f"N_{N:.4f}.npz"


def edge_fractions(U):
    k = int(round(0.05 * U.shape[-1]))
    I = np.abs(U) ** 2
    S = np.abs(np.fft.fftshift(np.fft.fft(U, axis=-1), axes=-1)) ** 2
    et = (I[:, :k].sum(1) + I[:, -k:].sum(1)) / I.sum(1)
    ef = (S[:, :k].sum(1) + S[:, -k:].sum(1)) / S.sum(1)
    return float(et.max()), float(ef.max())


def compute_field(regime: str, N: float):
    rc = splits.regime_cfg(regime)
    g = splits.load_grid(regime)
    tau, _ = tau_grid(g["n_tau"], g["tau_max"])
    t0 = time.perf_counter()
    xi, U = ssfm(initial_condition(rc["pulse"], tau), tau, rc["s"], N, g["xi_max"], g["n_save"], g["dxi"])
    wall = time.perf_counter() - t0
    et, ef = edge_fractions(U)
    meta = {
        "regime": regime, "N": N, "s": rc["s"], "pulse": rc["pulse"], "split": splits.split_of(regime, N),
        "grid": g, "solver": SOLVER_VERSION, "git": git_info(), "wall_s": wall,
        "edge_time_max": et, "edge_freq_max": ef,
        "E_drift_max": float(C.relative_drift(C.energy(U, tau)).max()),
        "H_drift_max": float(C.relative_drift(C.hamiltonian(U, tau, rc["s"], N)).max()),
    }
    return xi, tau, U, meta


def generate(regime: str, N: float, data_dir: Path = DATA_DIR, overwrite: bool = False) -> dict:
    p = field_path(regime, N, data_dir)
    if p.exists() and not overwrite:
        return json.loads(str(np.load(p)["meta"]))
    xi, tau, U, meta = compute_field(regime, N)
    p.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(p, U=U, xi=xi, tau=tau, meta=json.dumps(meta))
    return meta


def check(regime: str, N: float, data_dir: Path = DATA_DIR) -> float:
    """Recompute one file and return the relative L2 difference to the stored field."""
    d = np.load(field_path(regime, N, data_dir))
    _, _, U, _ = compute_field(regime, N)
    return float(np.linalg.norm(U - d["U"]) / np.linalg.norm(d["U"]))
