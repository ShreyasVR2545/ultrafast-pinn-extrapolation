"""Leakage-guarded data access.

* `load_training_fields` is the ONLY loader used by training code. It refuses any N outside the
  pre-registered train split and crops xi to the experiment's training range (z_extrap: xi <= 2).
* `load_eval_field` is for evaluation only and is never imported by train/ modules
  (enforced by tests/test_splits.py).
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import splits
from .generate import DATA_DIR, field_path


class LeakageError(RuntimeError):
    pass


@dataclass
class Field:
    N: float
    xi: np.ndarray
    tau: np.ndarray
    U: np.ndarray
    meta: dict


def _read(regime: str, N: float, data_dir: Path) -> Field:
    d = np.load(field_path(regime, N, data_dir))
    meta = json.loads(str(d["meta"]))
    if abs(meta["N"] - N) > 1e-9 or meta["regime"] != regime:
        raise LeakageError(f"file metadata mismatch for {regime} N={N}: {meta['regime']} N={meta['N']}")
    return Field(N, d["xi"], d["tau"], d["U"], meta)


def load_training_fields(regime: str, exp: str, data_dir: Path = DATA_DIR,
                         n_list: list[float] | None = None) -> list[Field]:
    """Training fields for (regime, exp). n_list may only be a subset of the train split."""
    train = splits.train_n(regime)
    n_list = train if n_list is None else n_list
    for N in n_list:
        if not any(abs(N - t) < 1e-9 for t in train):
            raise LeakageError(f"N={N} is not in the training split of regime {regime}")
    lo, hi = splits.xi_train_range(exp)
    out = []
    for N in n_list:
        f = _read(regime, N, data_dir)
        if f.meta["split"] != "train":
            raise LeakageError(f"file for N={N} is labelled split={f.meta['split']}")
        m = (f.xi >= lo - 1e-12) & (f.xi <= hi + 1e-12)
        out.append(Field(N, f.xi[m], f.tau, f.U[m], f.meta))
    return out


def load_eval_field(regime: str, N: float, data_dir: Path = DATA_DIR) -> Field:
    return _read(regime, N, data_dir)
