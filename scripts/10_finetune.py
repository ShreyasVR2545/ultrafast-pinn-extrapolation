"""Phase 9.3 (optional): test-time physics fine-tuning vs SSFM.

Start from the parametric PINN-strict weights (main sweep, n_extrap) and fine-tune with the PDE
residual + IC at ONE extrapolation N (no labels). Track the true error every `eval_every` L-BFGS
iterations (evaluation only). Report the steps and wall-clock needed to reach the model's own mean
in-distribution (interp) error, next to the SSFM wall-clock at matched accuracy
(results/ssfm_cost.csv). Writes results/finetune.csv.

Usage: python scripts/10_finetune.py [--smoke] [--seeds 0]
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nlse_ml.data import splits  # noqa: E402
from nlse_ml.data.datasets import load_eval_field  # noqa: E402
from nlse_ml.train import train_pinn  # noqa: E402

CASES = {"A": [3.0, 4.0], "B_sech": [2.0, 3.0]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--seeds", nargs="+", type=int, default=[0])
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--eval-every", type=int, default=250)
    a = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows = []
    steps = 20 if a.smoke else a.steps
    every = 10 if a.smoke else a.eval_every
    for regime, Ns in CASES.items():
        for seed in a.seeds:
            rd = ROOT / "runs" / "main" / "n_extrap" / "pinn_strict" / regime / f"seed{seed}"
            if not (rd / "ckpt.pt").exists():
                print("missing", rd)
                continue
            cfg = yaml.safe_load((rd / "config.yaml").read_text())
            m = json.loads((rd / "metrics.json").read_text())
            target = m["per_split"]["interp"]["rel_l2"]
            for N in Ns:
                f = load_eval_field(regime, N)
                net = train_pinn.build_model(cfg, regime, "n_extrap", dev)
                net.load_state_dict(torch.load(rd / "ckpt.pt", map_location=dev))
                c = copy.deepcopy(cfg)
                c["collocation_n_fixed"] = [N - 1e-3, N + 1e-3]
                c["schedule"]["adam_steps"] = 0
                lossf = train_pinn.PINNLoss(net, c, regime, "n_extrap", dev, seed + 100)
                batch = lossf.sample("lbfgs")
                opt = torch.optim.LBFGS(net.parameters(), lr=1.0, max_iter=1, history_size=50,
                                        line_search_fn="strong_wolfe")

                def closure():
                    opt.zero_grad(set_to_none=True)
                    loss, _ = lossf(batch, "lbfgs")
                    loss.backward()
                    return loss

                def err():
                    P = train_pinn.predict(net, regime, N, f.xi, f.tau, dev)
                    return float(np.linalg.norm(P - f.U) / np.linalg.norm(f.U))

                e0, t_train, reached = err(), 0.0, None
                rows.append({"regime": regime, "N": N, "seed": seed, "step": 0, "wall_s": 0.0, "rel_l2": e0,
                             "target_interp": target})
                for it in range(1, steps + 1):
                    if it % 2000 == 0:  # same resampling rule as training (D-029)
                        batch = lossf.sample("lbfgs")
                        opt = torch.optim.LBFGS(net.parameters(), lr=1.0, max_iter=1, history_size=50,
                                                line_search_fn="strong_wolfe")
                    t0 = time.perf_counter()
                    opt.step(closure)
                    if dev.type == "cuda":
                        torch.cuda.synchronize()
                    t_train += time.perf_counter() - t0
                    if it % every == 0:
                        e = err()
                        rows.append({"regime": regime, "N": N, "seed": seed, "step": it, "wall_s": t_train,
                                     "rel_l2": e, "target_interp": target})
                        if reached is None and e <= target:
                            reached = it
                print(regime, N, seed, f"err {e0:.3f} -> {rows[-1]['rel_l2']:.3f}, target {target:.3f}, "
                      f"reached at step {reached}, {t_train:.0f}s")
    if not a.smoke:
        pd.DataFrame(rows).to_csv(ROOT / "results" / "finetune.csv", index=False)


if __name__ == "__main__":
    main()
