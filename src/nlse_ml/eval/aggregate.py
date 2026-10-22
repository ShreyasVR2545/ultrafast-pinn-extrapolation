"""Scan runs/<sweep>/**/metrics.json and write tidy CSVs to results/.

results/<sweep>_long.csv   : model, regime, exp, split, N, seed, tag, metric, value  (per-N scalars)
results/<sweep>_runs.csv   : one row per run (status, wall-clock, memory, steps, params, inference)
results/<sweep>_slices.csv : model, regime, exp, split, N, seed, tag, xi, metric, value (per-xi)
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from ..utils import REPO_ROOT

RUN_KEYS = ["status", "train_wall_s", "adam_wall_s", "lbfgs_wall_s", "steps_done", "peak_mem_gb", "n_params",
            "inference_s_per_field"]


def scan(sweep: str):
    long, runs, slices = [], [], []
    for p in sorted((REPO_ROOT / "runs" / sweep).rglob("metrics.json")):
        m = json.loads(p.read_text())
        base = {k: m.get(k) for k in ["model", "regime", "exp", "seed", "tag"]}
        base["tag"] = base["tag"] or ""
        runs.append({**base, **{k: m.get(k) for k in RUN_KEYS}, "run_dir": str(p.parent.relative_to(REPO_ROOT))})
        for nkey, vals in m["per_N"].items():
            for k, v in vals.items():
                if isinstance(v, (int, float)) and not isinstance(v, bool):
                    long.append({**base, "split": vals["split"], "N": float(nkey), "metric": k, "value": v})
        for nkey, d in m.get("per_slice", {}).items():
            split = m["per_N"][nkey]["split"]
            for k in d:
                if k == "xi":
                    continue
                for xi, v in zip(d["xi"], d[k]):
                    slices.append({**base, "split": split, "N": float(nkey), "xi": xi, "metric": k, "value": v})
    return pd.DataFrame(long), pd.DataFrame(runs), pd.DataFrame(slices)


def write(sweep: str, out_dir: Path | None = None):
    out_dir = Path(out_dir or REPO_ROOT / "results")
    out_dir.mkdir(exist_ok=True)
    long, runs, slices = scan(sweep)
    long.to_csv(out_dir / f"{sweep}_long.csv", index=False)
    runs.to_csv(out_dir / f"{sweep}_runs.csv", index=False)
    if not slices.empty:  # keep the committed CSV small: only the metrics used in figures
        # committed file must stay < 5 MB: only the per-xi metrics used by figures/analyses, 4 sig. figs
        s = slices[slices["metric"].isin(["rel_l2", "energy_drift"])].copy()
        s["value"] = s["value"].map(lambda v: float(f"{v:.4g}"))
        s["xi"] = s["xi"].round(4)
        s.to_csv(out_dir / f"{sweep}_slices.csv.gz", index=False, compression="gzip")
    return long, runs, slices


if __name__ == "__main__":
    import sys

    for s in sys.argv[1:] or ["main"]:
        lg, rn, _ = write(s)
        print(s, len(lg), "rows,", len(rn), "runs")
