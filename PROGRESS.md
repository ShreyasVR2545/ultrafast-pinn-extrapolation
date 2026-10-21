# PROGRESS

## Checkpoint 0: Environment audit and compute feasibility (2026-10-22)
**Status:** done
**What I did:**
- Wrote `scripts/00_env_check.py`, giving `report/env_report.md` (hardware and software table).
- Created `.venv` (Python 3.12.13) with torch 2.7.1+cu128, numpy 2.2.6, scipy, matplotlib, pyyaml, pytest, pandas, neuraloperator 1.0.2. Lockfile in `requirements-lock.txt`.
- Wrote `scripts/01_benchmark.py`, giving `results/benchmark.json` and the benchmark and cost table in `report/env_report.md`.
- Implemented the core physics early, because the benchmarks needed it: `physics/{nlse,ssfm,analytic,conservation}.py`, `models/{mlp,pinn}.py`.
- Added an exact Taylor-mode derivative path for PINN residuals (1.4-1.7x faster than nested autograd).
**Key results:** (all in `report/env_report.md` / `results/benchmark.json`)
- Intel Core Ultra 9 275HX (24 threads), 34 GB RAM, RTX 5070 Laptop GPU, 8.5 GB, sm_120, CUDA 12.8.
- The PINN (4x100 tanh, 50k collocation points) is the bottleneck. It runs 18 Adam it/s with autograd and ~25 it/s in Taylor mode. float64 is ~12x slower.
- DD-MLP runs ~230 it/s, FNO ~75 it/s, and SSFM is negligible (<0.2 s per field on 512 points).
**Sanity checks:** a quick SSFM check gave a GVD analytic error of 6e-15 for both signs, an N=1 soliton shape error of 8e-8, and energy drift of 1e-12. The PINN residual on the exact GVD solution is ~1e-16, and O(1) when the sign of s is flipped.
**Problems / surprises:** the naive sequential estimate (~30 h) overshoots the 12 h budget. That estimate double-counts in-distribution and N-extrapolation, which share one training run. Running GPU jobs concurrently does not help. PINN step budgets will be set in a Phase 6 pilot so the total fits (D-003).
**Commits:** see Phase 1.
**Next:** Phase 0.5, literature grounding.
