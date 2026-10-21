"""Single entry point: train one (model, regime, experiment, seed), then evaluate it.

  python scripts/run_experiment.py --model pinn_strict --regime A --exp n_extrap --seed 0 [--smoke]
  python scripts/run_experiment.py ... --set schedule.adam_steps=5000 --tag ablation_x --sweep ablations

Writes runs/<sweep>/<exp>/<model>/<regime>/seed<k>[_<tag>]/ with config.yaml, meta.json, log.txt,
loss.png, ckpt.pt, fit.json, metrics.json, preds.npz. Skips if metrics.json exists (unless --force).
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nlse_ml.data import splits  # noqa: E402
from nlse_ml.eval.evaluate import evaluate  # noqa: E402
from nlse_ml.train import train_dd, train_fno, train_pinn  # noqa: E402
from nlse_ml.train.common import plot_loss  # noqa: E402
from nlse_ml.utils import run_meta, seed_everything, write_json  # noqa: E402

MODELS = ["pinn_strict", "pinn_extended", "pinn_plus", "dd_mlp", "cfno", "cfno_ar"]
CFG = ROOT / "configs" / "models"
SMOKE = {"schedule": {"adam_steps": 60, "lbfgs_steps": 10, "log_every": 20, "resample_every": 20},
         "sampling": {"n_col": 2000, "n_col_lbfgs": 2000, "n_ic": 256}}


def _deep_update(d, u):
    for k, v in u.items():
        if isinstance(v, dict) and isinstance(d.get(k), dict):
            _deep_update(d[k], v)
        else:
            d[k] = v
    return d


def _set(cfg, dotted, value):
    keys = dotted.split(".")
    d = cfg
    for k in keys[:-1]:
        d = d.setdefault(k, {})
    d[keys[-1]] = yaml.safe_load(value)


def resolve_config(model, smoke=False, overrides=()):
    shared = yaml.safe_load((CFG / "shared.yaml").read_text())
    if model in ("pinn_strict", "pinn_extended"):
        cfg = _deep_update(copy.deepcopy(shared), yaml.safe_load((CFG / "pinn.yaml").read_text()))
        cfg["collocation"] = "strict" if model == "pinn_strict" else "extended"
    elif model == "pinn_plus":
        cfg = _deep_update(copy.deepcopy(shared), yaml.safe_load((CFG / "pinn_plus.yaml").read_text()))
    elif model == "dd_mlp":
        cfg = _deep_update(copy.deepcopy(shared), yaml.safe_load((CFG / "dd_mlp.yaml").read_text()) or {})
    else:
        f = yaml.safe_load((CFG / "cfno.yaml").read_text())
        cfg = {"arch": f["arch"], "batch_size": f["batch_size"], "schedule": copy.deepcopy(shared["schedule"]),
               **f["variants"][model]}
    cfg["model"] = model
    if smoke:
        _deep_update(cfg, copy.deepcopy(SMOKE))
        if "n_modes" in cfg.get("arch", {}):
            cfg["arch"].update({"n_modes": 16, "hidden": 16})
    for o in overrides:
        k, v = o.split("=", 1)
        _set(cfg, k, v)
    return cfg


def run(model, regime, exp, seed, smoke=False, sweep="main", tag="", overrides=(), force=False):
    exp = "n_extrap" if exp == "indist" else exp
    if model == "pinn_extended" and exp == "z_extrap":
        raise SystemExit("pinn_extended is not run in z_extrap (pre-registered in configs/splits.yaml)")
    sweep = "smoke" if smoke else sweep
    rd = ROOT / "runs" / sweep / exp / model / regime / (f"seed{seed}" + (f"_{tag}" if tag else ""))
    if (rd / "metrics.json").exists() and not force:
        print(f"skip (done): {rd}")
        return rd
    rd.mkdir(parents=True, exist_ok=True)
    cfg = resolve_config(model, smoke, overrides)
    cfg.update({"regime": regime, "exp": exp, "seed": seed, "grid": splits.load_grid(regime)})
    (rd / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    write_json(rd / "meta.json", run_meta(seed, cfg))
    seed_everything(seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logf = open(rd / "log.txt", "w", buffering=1)

    def log(msg):
        logf.write(msg + "\n")

    log(f"{model} {regime} {exp} seed {seed} on {dev}")
    t0 = time.perf_counter()
    if model.startswith("pinn"):
        net, res = train_pinn.train(cfg, regime, exp, seed, dev, log)
        pred = lambda N, xi, tau, U0: train_pinn.predict(net, regime, N, xi, tau, dev)
        resid = lambda N, rng: train_pinn.residual_rms(net, regime, N, rng, 20000, dev)
    elif model == "dd_mlp":
        net, res = train_dd.train(cfg, regime, exp, seed, dev, log)
        pred = lambda N, xi, tau, U0: train_pinn.predict(net, regime, N, xi, tau, dev)
        resid = lambda N, rng: train_pinn.residual_rms(net, regime, N, rng, 20000, dev)
    else:
        net, res = train_fno.train(cfg, regime, exp, seed, dev, log)
        pred = lambda N, xi, tau, U0: train_fno.predict(net, cfg, regime, N, xi, tau, U0, dev)
        resid = None
    train_wall = time.perf_counter() - t0
    torch.save(net.state_dict(), rd / "ckpt.pt")
    plot_loss(res.history, rd / "loss.png", f"{model} {regime} {exp} seed{seed}")
    fit = {"status": res.status, "train_wall_s": train_wall, "adam_wall_s": res.adam_wall_s,
           "lbfgs_wall_s": res.lbfgs_wall_s, "steps_done": res.steps_done, "peak_mem_gb": res.peak_mem_gb,
           "n_params": sum(p.numel() for p in net.parameters()), "history": res.history}
    write_json(rd / "fit.json", fit)
    net.eval()
    m, preds = evaluate(pred, regime, exp, resid)
    m.update({k: v for k, v in fit.items() if k != "history"})
    m.update({"model": model, "regime": regime, "exp": exp, "seed": seed, "tag": tag, "sweep": sweep})
    np.savez_compressed(rd / "preds.npz", **preds)
    write_json(rd / "metrics.json", m)
    logf.close()
    summ = {sp: round(v["rel_l2"], 4) for sp, v in m["per_split"].items()}
    print(f"{model} {regime} {exp} s{seed} [{res.status}] {train_wall:.0f}s rel_l2 {summ}")
    return rd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=MODELS)
    ap.add_argument("--regime", required=True, choices=["A", "B_sech", "B_gauss"])
    ap.add_argument("--exp", required=True, choices=["indist", "n_extrap", "z_extrap"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--sweep", default="main")
    ap.add_argument("--tag", default="")
    ap.add_argument("--set", dest="overrides", action="append", default=[], help="dotted.key=yaml_value")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    run(a.model, a.regime, a.exp, a.seed, a.smoke, a.sweep, a.tag, a.overrides, a.force)


if __name__ == "__main__":
    main()
