"""Run blocks of experiments with resume (skip-if-done) and a status file.

  python scripts/run_sweep.py --blocks a b c d            # main study (configs in BLOCKS)
  python scripts/run_sweep.py --blocks a --seeds 0         # subset
  python scripts/run_sweep.py --plan my_plan.yaml          # custom list of jobs (ablations)

Each job runs in its own subprocess (a crash or OOM cannot kill the sweep). Progress goes to
runs/<sweep>/status.txt and runs/<sweep>/sweep_log.jsonl. Failed jobs are recorded, not dropped.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import subprocess
import sys
import time
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]

N_MODELS = ["pinn_strict", "pinn_extended", "pinn_plus", "dd_mlp", "cfno", "cfno_ar"]
Z_MODELS = ["pinn_strict", "pinn_plus", "dd_mlp", "cfno", "cfno_ar"]
BLOCKS = {  # priority order from the brief
    "a": [("A", "n_extrap", N_MODELS)],
    "b": [("A", "z_extrap", Z_MODELS)],
    "c": [("B_sech", "n_extrap", N_MODELS), ("B_gauss", "n_extrap", N_MODELS)],
    "d": [("B_sech", "z_extrap", Z_MODELS)],          # B_gauss z_extrap dropped (D-028)
    "d2": [("B_gauss", "z_extrap", Z_MODELS)],        # only if time remains
}


def jobs_from_blocks(blocks, seeds):
    out = []
    for b in blocks:
        for seed in seeds:  # seed-major: all models get seed 0 first, so partial sweeps stay balanced
            for regime, exp, models in BLOCKS[b]:
                for m in models:
                    out.append({"model": m, "regime": regime, "exp": exp, "seed": seed})
    return out


def run_dir(sweep, j):
    return (ROOT / "runs" / sweep / j["exp"] / j["model"] / j["regime"]
            / (f"seed{j['seed']}" + (f"_{j['tag']}" if j.get("tag") else "")))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--blocks", nargs="+", default=[])
    ap.add_argument("--plan", default=None)
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--sweep", default="main")
    ap.add_argument("--models", nargs="+", default=None, help="restrict to these models")
    a = ap.parse_args()
    jobs = jobs_from_blocks(a.blocks, a.seeds)
    if a.plan:
        jobs += yaml.safe_load(Path(a.plan).read_text())["jobs"]
    if a.models:
        jobs = [j for j in jobs if j["model"] in a.models]
    sd = ROOT / "runs" / a.sweep
    sd.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    n_done = n_fail = n_skip = 0
    for i, j in enumerate(jobs):
        rd = run_dir(a.sweep, j)
        if (rd / "metrics.json").exists():
            n_skip += 1
            continue
        cmd = [sys.executable, str(ROOT / "scripts" / "run_experiment.py"), "--model", j["model"], "--regime",
               j["regime"], "--exp", j["exp"], "--seed", str(j["seed"]), "--sweep", a.sweep]
        if j.get("tag"):
            cmd += ["--tag", j["tag"]]
        for o in j.get("set", []):
            cmd += ["--set", o]
        (sd / "status.txt").write_text(
            f"{dt.datetime.now():%Y-%m-%d %H:%M:%S} job {i + 1}/{len(jobs)} running: {j}\n"
            f"done {n_done}, failed {n_fail}, skipped(existing) {n_skip}, elapsed {(time.time() - t0) / 3600:.2f} h\n")
        ts = time.time()
        p = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
        ok = p.returncode == 0 and (rd / "metrics.json").exists()
        n_done += ok
        n_fail += not ok
        rec = {"job": j, "ok": ok, "rc": p.returncode, "wall_s": time.time() - ts,
               "stdout_tail": p.stdout[-400:], "stderr_tail": "" if ok else p.stderr[-2000:]}
        with open(sd / "sweep_log.jsonl", "a") as f:
            f.write(json.dumps(rec) + "\n")
    (sd / "status.txt").write_text(
        f"{dt.datetime.now():%Y-%m-%d %H:%M:%S} FINISHED {len(jobs)} jobs: done {n_done}, failed {n_fail}, "
        f"skipped {n_skip}, {(time.time() - t0) / 3600:.2f} h\n")
    print(f"done {n_done} failed {n_fail} skipped {n_skip}")


if __name__ == "__main__":
    main()
