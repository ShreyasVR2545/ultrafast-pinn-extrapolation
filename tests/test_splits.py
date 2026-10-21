"""Split integrity and no-leakage guarantees."""
import ast
from pathlib import Path

import numpy as np
import pytest

from nlse_ml.data import datasets, generate, splits

ROOT = Path(__file__).resolve().parents[1]
REGIMES = ["A", "B_sech", "B_gauss"]


@pytest.mark.parametrize("regime", REGIMES)
def test_no_n_in_two_splits(regime):
    seen = {}
    for sp in splits.split_names(regime):
        for n in splits.n_values(regime, sp):
            assert n not in seen, f"N={n} in {seen.get(n)} and {sp}"
            seen[n] = sp
    assert splits.split_of(regime, 1.0) == "train"


def test_preregistered_values_match_brief():
    assert splits.n_values("A", "train") == [1.0, 1.2, 1.4, 1.6, 1.8, 2.0]
    assert splits.n_values("A", "far") == [3.0, 4.0]
    assert splits.n_values("B_sech", "extrap") == [2.0, 3.0]
    assert splits.xi_train_range("z_extrap") == (0.0, 2.0)


def test_collocation_ranges():
    assert splits.collocation_n_range("A", "strict") == (1.0, 2.0)
    assert splits.collocation_n_range("A", "extended") == (1.0, 4.0)
    assert splits.collocation_n_range("B_sech", "strict") == (0.8, 1.6)


def _fake_data(tmp_path, regime, N, split, xi_max=4.0):
    xi = np.linspace(0, xi_max, 201)
    tau = np.linspace(-5, 5, 16, endpoint=False)
    U = np.ones((201, 16), complex) * N
    p = generate.field_path(regime, N, tmp_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    import json
    np.savez(p, U=U, xi=xi, tau=tau, meta=json.dumps({"regime": regime, "N": N, "split": split}))


def test_training_loader_refuses_test_n(tmp_path):
    for N in splits.all_n("A"):
        _fake_data(tmp_path, "A", N, splits.split_of("A", N))
    fields = datasets.load_training_fields("A", "n_extrap", tmp_path)
    assert sorted(f.N for f in fields) == splits.train_n("A")
    for bad in [1.1, 2.25, 3.0, 4.0]:
        with pytest.raises(datasets.LeakageError):
            datasets.load_training_fields("A", "n_extrap", tmp_path, n_list=[bad])


def test_training_loader_refuses_mislabelled_file(tmp_path):
    _fake_data(tmp_path, "A", 1.0, "far")  # train N but file says it is a test file
    with pytest.raises(datasets.LeakageError):
        datasets.load_training_fields("A", "n_extrap", tmp_path, n_list=[1.0])


def test_z_split_mask_excludes_xi_above_2(tmp_path):
    for N in splits.train_n("A"):
        _fake_data(tmp_path, "A", N, "train")
    for f in datasets.load_training_fields("A", "z_extrap", tmp_path):
        assert f.xi.max() <= 2.0 + 1e-12 and f.U.shape[0] == f.xi.size
    full = datasets.load_training_fields("A", "n_extrap", tmp_path)
    assert full[0].xi.max() == pytest.approx(4.0)


def test_train_modules_never_import_eval_loader():
    """Training code may only use load_training_fields; eval loader must not appear in train/."""
    for p in (ROOT / "src" / "nlse_ml" / "train").glob("*.py"):
        src = p.read_text()
        assert "load_eval_field" not in src, p
        tree = ast.parse(src)
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module and "eval" in node.module.split("."):
                raise AssertionError(f"{p} imports from eval package")
