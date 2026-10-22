"""Shared optimizer schedule: Adam warm-up (exponential lr decay, periodic resampling), then L-BFGS
(strong Wolfe line search) on a fixed batch. Identical for the PINN family and the DD-MLP, so both
get the same number of optimizer steps (fairness rule; wall-clock is logged too).

Diverged runs (NaN/inf loss) stop early and are reported with status "diverged"; they are results,
not silently dropped.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

import numpy as np
import torch


@dataclass
class Schedule:
    adam_steps: int = 10000
    lr: float = 1e-3
    lr_final: float = 1e-4
    resample_every: int = 100
    lbfgs_steps: int = 2000
    lbfgs_history: int = 50
    lbfgs_resample_every: int = 0  # >0: fresh batch + fresh L-BFGS state every K iterations (prevents
    #                                 overfitting a fixed collocation set; DECISIONS D-029)
    log_every: int = 100


@dataclass
class FitResult:
    status: str = "ok"
    wall_s: float = 0.0
    adam_wall_s: float = 0.0
    lbfgs_wall_s: float = 0.0
    steps_done: int = 0
    peak_mem_gb: float = float("nan")
    history: dict = field(default_factory=lambda: {"step": [], "loss": [], "parts": []})


def _sync(dev):
    if dev.type == "cuda":
        torch.cuda.synchronize()


def fit(model, loss_fn, sampler, sched: Schedule, log=print, on_step=None) -> FitResult:
    """loss_fn(batch, phase) -> (loss tensor, dict of float parts); sampler(phase) -> batch.

    on_step(step, phase, parts) is called after each Adam step (used by PINN+ for the causal
    epsilon schedule and adaptive loss weights).
    """
    dev = next(model.parameters()).device
    res = FitResult()
    if dev.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    opt = torch.optim.Adam(model.parameters(), lr=sched.lr)
    gamma = (sched.lr_final / sched.lr) ** (1.0 / max(1, sched.adam_steps))
    lr_sched = torch.optim.lr_scheduler.ExponentialLR(opt, gamma)
    batch = None
    for step in range(sched.adam_steps):
        if batch is None or step % sched.resample_every == 0:
            batch = sampler("adam")
        opt.zero_grad(set_to_none=True)
        loss, parts = loss_fn(batch, "adam")
        if not torch.isfinite(loss):
            res.status = "diverged"
            log(f"adam {step}: non-finite loss, stopping")
            break
        loss.backward()
        opt.step()
        lr_sched.step()
        res.steps_done += 1
        if on_step is not None:
            on_step(step, "adam", parts)
        if step % sched.log_every == 0 or step == sched.adam_steps - 1:
            _record(res, step, loss.item(), parts, log, "adam")
    _sync(dev)
    res.adam_wall_s = time.perf_counter() - t0

    if res.status == "ok" and sched.lbfgs_steps > 0:
        t1 = time.perf_counter()
        def new_lbfgs():
            return torch.optim.LBFGS(model.parameters(), lr=1.0, max_iter=1, history_size=sched.lbfgs_history,
                                     tolerance_grad=1e-12, tolerance_change=1e-15, line_search_fn="strong_wolfe")

        batch = sampler("lbfgs")
        lb = new_lbfgs()
        last = {}

        def closure():
            lb.zero_grad(set_to_none=True)
            l, p = loss_fn(batch, "lbfgs")
            l.backward()
            last["loss"], last["parts"] = l.item(), p
            return l

        for it in range(sched.lbfgs_steps):
            if sched.lbfgs_resample_every and it > 0 and it % sched.lbfgs_resample_every == 0:
                batch = sampler("lbfgs")
                lb = new_lbfgs()
            lb.step(closure)
            res.steps_done += 1
            if not math.isfinite(last["loss"]):
                res.status = "diverged"
                log(f"lbfgs {it}: non-finite loss, stopping")
                break
            if it % sched.log_every == 0 or it == sched.lbfgs_steps - 1:
                _record(res, sched.adam_steps + it, last["loss"], last["parts"], log, "lbfgs")
        _sync(dev)
        res.lbfgs_wall_s = time.perf_counter() - t1
    res.wall_s = time.perf_counter() - t0
    if dev.type == "cuda":
        res.peak_mem_gb = torch.cuda.max_memory_allocated() / 1e9
    return res


def _record(res, step, loss, parts, log, phase):
    res.history["step"].append(step)
    res.history["loss"].append(loss)
    res.history["parts"].append({k: float(v) for k, v in parts.items()})
    log(f"{phase} {step:6d} loss {loss:.4e} " + " ".join(f"{k} {v:.3e}" for k, v in parts.items()))


def lhs(n: int, lo, hi, rng: np.random.Generator) -> np.ndarray:
    """Latin hypercube sample of n points in the box [lo, hi] (d dims)."""
    lo, hi = np.asarray(lo, float), np.asarray(hi, float)
    d = lo.size
    u = (rng.random((n, d)) + np.stack([rng.permutation(n) for _ in range(d)], 1)) / n
    return lo + u * (hi - lo)


def plot_loss(history: dict, path, title=""):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from ..viz import style

    style.apply()
    fig, ax = plt.subplots(figsize=(5, 3.2))
    steps = np.array(history["step"])
    ax.semilogy(steps, history["loss"], color=style.INK, label="total")
    keys = sorted({k for p in history["parts"] for k in p})
    for i, k in enumerate(keys):
        vals = [p.get(k, np.nan) for p in history["parts"]]
        ax.semilogy(steps, vals, color=style.PALETTE[i % len(style.PALETTE)], lw=1.2, label=k)
    ax.set(xlabel="optimizer step", ylabel="loss", title=title)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
