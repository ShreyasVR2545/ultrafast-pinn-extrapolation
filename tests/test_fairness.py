"""Fairness rules from the protocol, enforced on the resolved configs."""
import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("run_experiment", ROOT / "scripts" / "run_experiment.py")
rx = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rx)


def test_pinn_and_dd_share_architecture_schedule_and_batch():
    ref = rx.resolve_config("pinn_strict")
    for m in ["pinn_extended", "pinn_plus", "dd_mlp"]:
        c = rx.resolve_config(m)
        assert c["arch"] == ref["arch"], m
        assert c["schedule"] == ref["schedule"], m
        assert c["sampling"]["n_col"] == ref["sampling"]["n_col"], m
        assert c["sampling"]["n_col_lbfgs"] == ref["sampling"]["n_col_lbfgs"], m


def test_fno_gets_same_total_step_budget():
    ref = rx.resolve_config("pinn_strict")["schedule"]
    for m in ["cfno", "cfno_ar"]:
        s = rx.resolve_config(m)["schedule"]
        assert s["adam_steps"] + s["lbfgs_steps"] == ref["adam_steps"] + ref["lbfgs_steps"]


def test_strict_vs_extended_collocation():
    assert rx.resolve_config("pinn_strict")["collocation"] == "strict"
    assert rx.resolve_config("pinn_extended")["collocation"] == "extended"
    assert rx.resolve_config("pinn_plus")["collocation"] == "strict"
