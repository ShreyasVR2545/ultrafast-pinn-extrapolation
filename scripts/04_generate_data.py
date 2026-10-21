"""Phase 4: generate SSFM datasets for every regime and N in configs/splits.yaml.

Usage:
  python scripts/04_generate_data.py                 # all regimes, skip existing files
  python scripts/04_generate_data.py --smoke         # one N of regime A into data_smoke/
  python scripts/04_generate_data.py --check         # recompute one file per regime and compare
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nlse_ml.data import generate as G, splits  # noqa: E402

REGIMES = ["A", "B_sech", "B_gauss"]


def _job(args):
    regime, N, data_dir, overwrite = args
    return G.generate(regime, N, Path(data_dir), overwrite)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()
    if a.check:
        for reg in REGIMES:
            N = splits.n_values(reg, "far" if reg == "A" else "extrap")[-1]
            d = G.check(reg, N)
            print(f"check {reg} N={N}: rel diff {d:.2e}", "OK" if d < 1e-12 else "MISMATCH")
        return
    data_dir = ROOT / ("data_smoke" if a.smoke else "data")
    jobs = [("A", 1.0, str(data_dir), True)] if a.smoke else \
        [(reg, N, str(data_dir), a.overwrite) for reg in REGIMES for N in splits.all_n(reg)]
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        metas = list(ex.map(_job, jobs))
    rows = []
    for m in metas:
        rows.append({k: m[k] for k in ["regime", "N", "split", "wall_s", "edge_time_max", "edge_freq_max",
                                       "E_drift_max", "H_drift_max"]})
        print(json.dumps(rows[-1]))
    bad = [r for r in rows if r["edge_time_max"] > 1e-6 or r["edge_freq_max"] > 1e-6 or r["H_drift_max"] > 1e-6]
    size_mb = sum(p.stat().st_size for p in data_dir.rglob("*.npz")) / 1e6
    print(f"{len(rows)} fields, {size_mb:.0f} MB, {time.time() - t0:.0f} s wall; quality failures: {len(bad)}")
    if not a.smoke:
        (ROOT / "results" / "data_summary.json").write_text(json.dumps(
            {"n_fields": len(rows), "size_mb": size_mb, "wall_s": time.time() - t0, "fields": rows,
             "quality_failures": bad}, indent=2))
    if bad:
        print("WARNING: fields failing grid-quality criteria:", bad)


if __name__ == "__main__":
    main()
