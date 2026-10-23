"""Analysis A merge: per-slice errors must get spectral-shift predictors for EVERY slice."""
import numpy as np
import pandas as pd
import pytest

from nlse_ml.eval.analysis import PREDICTORS, merge_shift


def _tables():
    xi = np.linspace(0.0, 4.0, 201)  # raw values as in the data files, e.g. 0.7000000000000001
    shift = pd.DataFrame([{"regime": "A", "N": N, "xi": x, **{p: 1.0 for p in PREDICTORS}}
                          for N in (1.1, 2.25) for x in xi])
    slices = pd.DataFrame([{"regime": "A", "N": N, "xi": round(x, 4), "err": 0.1}  # as written by aggregate.py
                           for N in (1.1, 2.25) for x in xi])
    return slices, shift


def test_exact_float_merge_would_lose_slices():
    slices, shift = _tables()
    naive = slices.merge(shift, on=["regime", "N", "xi"], how="left")
    assert naive["phi_eff"].isna().sum() > 0  # documents the bug fixed in D-031


def test_merge_shift_matches_every_slice():
    slices, shift = _tables()
    out = merge_shift(slices, shift)
    assert len(out) == len(slices)
    assert not out[PREDICTORS].isna().any().any()


def test_merge_shift_raises_on_missing_predictors():
    slices, shift = _tables()
    with pytest.raises(ValueError):
        merge_shift(slices, shift[shift.xi < 3.0])
