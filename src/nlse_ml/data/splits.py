"""Access to the pre-registered splits (configs/splits.yaml). Single source of truth for which N
values and xi ranges each experiment may train on."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[3]
SPLITS_PATH = ROOT / "configs" / "splits.yaml"
GRID_PATH = ROOT / "configs" / "grid.yaml"


@lru_cache(maxsize=None)
def load_splits(path: str | None = None) -> dict:
    return yaml.safe_load(Path(path or SPLITS_PATH).read_text())


@lru_cache(maxsize=None)
def load_grid(regime: str) -> dict:
    """Validated numerical grid. B_sech and B_gauss share regime B's grid."""
    grids = yaml.safe_load(GRID_PATH.read_text())["regimes"]
    return dict(grids["A" if regime == "A" else "B"])


def regime_cfg(regime: str) -> dict:
    return load_splits()["regimes"][regime]


def split_names(regime: str) -> list[str]:
    return list(regime_cfg(regime)["N"].keys())


def n_values(regime: str, split: str) -> list[float]:
    return [float(x) for x in regime_cfg(regime)["N"][split]]


def all_n(regime: str) -> list[float]:
    return sorted({n for sp in split_names(regime) for n in n_values(regime, sp)})


def split_of(regime: str, N: float) -> str:
    hits = [sp for sp in split_names(regime) if any(abs(N - n) < 1e-9 for n in n_values(regime, sp))]
    if len(hits) != 1:
        raise ValueError(f"N={N} belongs to {hits} in regime {regime}")
    return hits[0]


def train_n(regime: str) -> list[float]:
    return n_values(regime, "train")


def xi_train_range(exp: str) -> tuple[float, float]:
    lo, hi = load_splits()["experiments"][exp]["xi_train"]
    return float(lo), float(hi)


def collocation_n_range(regime: str, variant: str) -> tuple[float, float]:
    tr = train_n(regime)
    if variant == "strict":
        return min(tr), max(tr)
    if variant == "extended":
        return min(tr), max(all_n(regime))
    raise ValueError(variant)


def eval_n(regime: str, exp: str) -> dict[str, list[float]]:
    e = load_splits()["experiments"][exp]
    names = split_names(regime) if e.get("eval_splits") == "all" else e["eval_N_splits"]
    return {sp: n_values(regime, sp) for sp in names}
