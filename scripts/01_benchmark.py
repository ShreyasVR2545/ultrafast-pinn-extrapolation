"""Phase 0 micro-benchmarks. Appends a '## Benchmarks' section to report/env_report.md
and writes results/benchmark.json.

Usage: python scripts/01_benchmark.py [--smoke]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nlse_ml.models.mlp import MLP  # noqa: E402
from nlse_ml.models.pinn import residual_loss  # noqa: E402
from nlse_ml.physics.nlse import gaussian, tau_grid  # noqa: E402
from nlse_ml.physics.ssfm import ssfm  # noqa: E402


def _sync(dev):
    if dev.type == "cuda":
        torch.cuda.synchronize()


def bench_ssfm(dxi):
    tau, _ = tau_grid(512, 20.0)
    t = time.perf_counter()
    ssfm(gaussian(tau), tau, 1, 2.0, 4.0, 201, dxi)
    return time.perf_counter() - t


def bench_pinn(dev, dtype, n_col, adam_steps, lbfgs_steps, mode="autograd"):
    torch.manual_seed(0)
    model = MLP(3, 2, 100, 4, "tanh", lb=[0, -20, 1], ub=[4, 20, 2]).to(dev, dtype)
    x = torch.rand(n_col, 3, device=dev, dtype=dtype) * torch.tensor([4, 40, 1], device=dev, dtype=dtype) \
        + torch.tensor([0, -20, 1], device=dev, dtype=dtype)
    xi, tau, Nv = x[:, 0:1], x[:, 1:2], x[:, 2:3]
    opt = torch.optim.Adam(model.parameters(), 1e-3)
    if dev.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    for i in range(adam_steps + 5):
        if i == 5:  # exclude warm-up
            _sync(dev)
            t = time.perf_counter()
        opt.zero_grad()
        loss = residual_loss(model, xi, tau, Nv, 1, mode)
        loss.backward()
        opt.step()
    _sync(dev)
    adam_sps = adam_steps / (time.perf_counter() - t)

    lb = torch.optim.LBFGS(model.parameters(), lr=1.0, max_iter=1, history_size=50, line_search_fn="strong_wolfe")

    def closure():
        lb.zero_grad()
        loss = residual_loss(model, xi, tau, Nv, 1, mode)
        loss.backward()
        return loss

    _sync(dev)
    t = time.perf_counter()
    for _ in range(lbfgs_steps):
        lb.step(closure)
    _sync(dev)
    lbfgs_sps = lbfgs_steps / (time.perf_counter() - t)
    mem = torch.cuda.max_memory_allocated() / 1e9 if dev.type == "cuda" else float("nan")
    return adam_sps, lbfgs_sps, mem


def bench_dd(dev, n_pts, steps):
    torch.manual_seed(0)
    model = MLP(3, 2, 100, 4, "tanh").to(dev)
    x = torch.rand(n_pts, 3, device=dev)
    y = torch.rand(n_pts, 2, device=dev)
    opt = torch.optim.Adam(model.parameters(), 1e-3)
    for i in range(steps + 5):
        if i == 5:
            _sync(dev)
            t = time.perf_counter()
        idx = torch.randint(0, n_pts, (50_000,), device=dev)
        opt.zero_grad()
        loss = ((model(x[idx]) - y[idx]) ** 2).mean()
        loss.backward()
        opt.step()
    _sync(dev)
    return steps / (time.perf_counter() - t)


def bench_fno(dev, steps, n_tau=512, k_out=21):
    from neuralop.models import FNO

    torch.manual_seed(0)
    model = FNO(n_modes=(64,), hidden_channels=64, in_channels=4, out_channels=2 * k_out, n_layers=4).to(dev)
    x = torch.randn(32, 4, n_tau, device=dev)
    y = torch.randn(32, 2 * k_out, n_tau, device=dev)
    opt = torch.optim.Adam(model.parameters(), 1e-3)
    for i in range(steps + 5):
        if i == 5:
            _sync(dev)
            t = time.perf_counter()
        opt.zero_grad()
        loss = ((model(x) - y) ** 2).mean()
        loss.backward()
        opt.step()
    _sync(dev)
    return steps / (time.perf_counter() - t)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_col = 2000 if a.smoke else 50_000
    adam, lbfgs = (10, 3) if a.smoke else (200, 20)
    r = {"device": str(dev), "n_col": n_col}
    r["ssfm_s_dxi1e-3"] = bench_ssfm(1e-3 if not a.smoke else 1e-2)
    print(f"SSFM 512 pts, 201 saves, xi 0..4: {r['ssfm_s_dxi1e-3']:.2f} s")
    r["pinn_f32_adam_sps"], r["pinn_f32_lbfgs_sps"], r["pinn_f32_mem_gb"] = bench_pinn(dev, torch.float32, n_col, adam, lbfgs)
    print(f"PINN f32: Adam {r['pinn_f32_adam_sps']:.1f} it/s, L-BFGS {r['pinn_f32_lbfgs_sps']:.1f} it/s, mem {r['pinn_f32_mem_gb']:.2f} GB")
    if dev.type == "cuda":
        r["pinn_f64_adam_sps"], r["pinn_f64_lbfgs_sps"], r["pinn_f64_mem_gb"] = bench_pinn(dev, torch.float64, n_col, adam, lbfgs)
        print(f"PINN f64: Adam {r['pinn_f64_adam_sps']:.1f} it/s, L-BFGS {r['pinn_f64_lbfgs_sps']:.1f} it/s")
    r["pinn_taylor_adam_sps"], r["pinn_taylor_lbfgs_sps"], _ = bench_pinn(dev, torch.float32, n_col, adam, lbfgs, "taylor")
    print(f"PINN f32 taylor-mode: Adam {r['pinn_taylor_adam_sps']:.1f} it/s, L-BFGS {r['pinn_taylor_lbfgs_sps']:.1f} it/s")
    r["dd_adam_sps"] = bench_dd(dev, 200_000, adam)
    print(f"DD-MLP: {r['dd_adam_sps']:.1f} it/s (50k batch)")
    r["fno_sps"] = bench_fno(dev, 10 if a.smoke else 50)
    print(f"FNO: {r['fno_sps']:.1f} it/s (batch 32)")
    if a.smoke:
        return
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results" / "benchmark.json").write_text(json.dumps(r, indent=2))

    # Cost extrapolation to the full plan (steps per run are the planned defaults in configs/models).
    plan_adam, plan_lbfgs = 20_000, 5_000
    pinn_h = (plan_adam / r["pinn_taylor_adam_sps"] + plan_lbfgs / r["pinn_taylor_lbfgs_sps"]) / 3600
    dd_h = (plan_adam / r["dd_adam_sps"] + plan_lbfgs / r["dd_adam_sps"]) / 3600
    fno_h = 20_000 / r["fno_sps"] / 3600
    seeds = 3
    rows = [
        ("Phase 2: repo presets (3 presets x 3 seeds)", 9 * pinn_h * 0.5),
        ("Phase 3/4: SSFM validation + data (~40 fields)", 40 * 10 / 3600),
        ("Phase 6: in-distribution, 6 models x 1 seed", 3 * pinn_h + dd_h + 2 * fno_h),
        ("Phase 7a/b: regime A N- & z-extrap", 2 * seeds * (3 * pinn_h + dd_h + 2 * fno_h)),
        ("Phase 7c/d: regime B N- & z-extrap", 2 * seeds * (3 * pinn_h + dd_h + 2 * fno_h)),
        ("Phase 8: ablations (~20 PINN-sized runs)", 20 * pinn_h),
    ]
    lines = ["", "## Benchmarks", "", f"Device: {dev}. Raw numbers in `results/benchmark.json`.", "",
             "| Benchmark | Result |", "|---|---|",
             f"| SSFM, 512 pts, 201 saves, dxi=1e-3, xi in [0,4] | {r['ssfm_s_dxi1e-3']:.2f} s |",
             f"| PINN 4x100 tanh, {n_col} colloc, f32 Adam | {r['pinn_f32_adam_sps']:.1f} it/s |",
             f"| PINN f32 L-BFGS (strong Wolfe, max_iter=1) | {r['pinn_f32_lbfgs_sps']:.1f} it/s |",
             f"| PINN f32 peak memory | {r['pinn_f32_mem_gb']:.2f} GB |"]
    if "pinn_f64_adam_sps" in r:
        lines += [f"| PINN f64 Adam / L-BFGS | {r['pinn_f64_adam_sps']:.1f} / {r['pinn_f64_lbfgs_sps']:.1f} it/s |"]
    lines += [f"| PINN f32 Taylor-mode derivatives, Adam / L-BFGS | {r['pinn_taylor_adam_sps']:.1f} / {r['pinn_taylor_lbfgs_sps']:.1f} it/s |"]
    lines += [f"| DD-MLP Adam (50k batch) | {r['dd_adam_sps']:.1f} it/s |",
              f"| FNO 1D (64 modes, 64 ch, 4 layers), batch 32 | {r['fno_sps']:.1f} it/s |",
              "", f"### Cost extrapolation (assumes {plan_adam} Adam + {plan_lbfgs} L-BFGS steps per PINN/MLP run, "
              f"20k FNO steps, {seeds} seeds, sequential, Taylor-mode PINN; NOTE: naive, see DECISIONS.md for the actual plan)", "",
              f"Per run: PINN {pinn_h:.2f} h, DD-MLP {dd_h:.2f} h, FNO {fno_h:.2f} h.", "",
              "| Phase | Est. hours |", "|---|---|"]
    lines += [f"| {k} | {v:.1f} |" for k, v in rows]
    lines += [f"| **Total** | **{sum(v for _, v in rows):.1f}** |", ""]
    out = ROOT / "report" / "env_report.md"
    txt = out.read_text() if out.exists() else ""
    if "## Benchmarks" in txt:
        txt = txt[: txt.index("## Benchmarks")].rstrip() + "\n"
    out.write_text(txt + "\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
