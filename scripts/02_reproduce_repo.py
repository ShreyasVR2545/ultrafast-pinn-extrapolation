"""Phase 2: reproduce starter-repo presets across seeds, without modifying the submodule.

Runs `third_party/fiber-nlse-pinn/main.py full` with the README's recommended flags, writing to
runs/repro/<preset>/seed<k>/ and parsing the reported rel_l2. Aggregates to results/repro_repo.csv.

Usage:
  python scripts/02_reproduce_repo.py --smoke
  python scripts/02_reproduce_repo.py --presets fig3 fig7 --seeds 0 1 2
  python scripts/02_reproduce_repo.py --aggregate
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT / "third_party" / "fiber-nlse-pinn"
sys.path.insert(0, str(ROOT / "src"))
from nlse_ml.utils import run_meta, write_json  # noqa: E402

# README "Recommended commands" and reported seed-42 errors
PRESETS = {
    "fig3": (["--hidden", "100", "--layers", "4", "--adam-steps", "0", "--lbfgs-epochs", "500"], 0.0261),
    "fig7": (["--hidden", "100", "--layers", "4", "--adam-steps", "5000", "--lbfgs-epochs", "300",
              "--ic-weight", "100"], 0.0288),
    "fig8": (["--hidden", "100", "--layers", "6", "--adam-steps", "5000", "--lbfgs-epochs", "300",
              "--ic-weight", "100", "--decompose", "8"], 0.0048),
}
SMOKE = ["--adam-steps", "20", "--lbfgs-epochs", "2", "--lbfgs-max-iter", "5", "--n-pde", "2000"]


def _replace_flags(base: list[str], new: list[str]) -> list[str]:
    out = list(base)
    for i in range(0, len(new), 2):
        if new[i] in out:
            out[out.index(new[i]) + 1] = new[i + 1]
        else:
            out += [new[i], new[i + 1]]
    return out


def run_one(preset: str, seed: int, smoke: bool) -> dict:
    flags, reported = PRESETS[preset]
    if smoke:
        flags = _replace_flags(flags, SMOKE)
    exp = "repro_smoke" if smoke else "repro"
    out = ROOT / "runs" / exp / preset / f"seed{seed}"
    if (out / "metrics.json").exists():
        print(f"skip {preset} seed{seed} (done)")
        return json.loads((out / "metrics.json").read_text())
    out.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, "main.py", "full", "--config", preset, "--device", "cuda",
           "--seed", str(seed), "--out-dir", str(out), *flags]
    write_json(out / "meta.json", run_meta(seed, {"preset": preset, "cmd": cmd, "smoke": smoke}))
    t0 = time.time()
    env = {**os.environ, "PYTHONUNBUFFERED": "1", "PYTHONIOENCODING": "utf-8"}
    with open(out / "stdout.txt", "w", encoding="utf-8") as f:
        rc = subprocess.call(cmd, cwd=REPO, stdout=f, stderr=subprocess.STDOUT, env=env)
    wall = time.time() - t0
    text = (out / "stdout.txt").read_text(encoding="utf-8", errors="replace")
    m = re.findall(r"rel_l2\s*=\s*([0-9.eE+-]+)", text)
    res = {"preset": preset, "seed": seed, "rc": rc, "wall_s": wall,
           "rel_l2": float(m[-1]) if m else None, "reported_seed42": reported, "smoke": smoke}
    if rc == 0 and m:
        write_json(out / "metrics.json", res)
    print(res)
    return res


def aggregate():
    import numpy as np

    rows = []
    for p in sorted((ROOT / "runs" / "repro").glob("*/seed*/metrics.json")):
        rows.append(json.loads(p.read_text()))
    if not rows:
        print("no runs")
        return
    out = ROOT / "results" / "repro_repo.csv"
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["preset", "seed", "rel_l2", "wall_s", "reported_seed42"])
        w.writeheader()
        for r in rows:
            w.writerow({k: r[k] for k in w.fieldnames})
    lines = ["| Preset | Paper fig. | Seeds | rel L2 mean ± std | median | README (seed 42) | wall s/run |",
             "|---|---|---|---|---|---|---|"]
    paper_fig = {"fig3": "5", "fig7": "9", "fig8": "10"}
    for preset in PRESETS:
        rs = [r for r in rows if r["preset"] == preset]
        if not rs:
            continue
        e = np.array([r["rel_l2"] for r in rs])
        lines.append(f"| {preset} | {paper_fig[preset]} | {len(rs)} | {e.mean():.4f} ± {e.std(ddof=1) if len(e) > 1 else float('nan'):.4f} "
                     f"| {np.median(e):.4f} | {rs[0]['reported_seed42']:.4f} | {np.mean([r['wall_s'] for r in rs]):.0f} |")
    (ROOT / "results" / "repro_repo.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--presets", nargs="+", default=list(PRESETS))
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--aggregate", action="store_true")
    a = ap.parse_args()
    if a.aggregate:
        aggregate()
        return
    for preset in a.presets:
        for seed in a.seeds:
            run_one(preset, seed, a.smoke)
    if not a.smoke:
        aggregate()


if __name__ == "__main__":
    main()
