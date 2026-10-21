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

## Checkpoint 0.5: Literature grounding (2026-10-22)
**Status:** done
**What I did:** Read the full texts of Jiang et al., Fesser et al. and Wang et al. (causal), and the abstracts of Krishnapriyan et al. and the neuraloperator FNO docs. Ran 6 searches for 2023+ work. Wrote `report/literature_notes.md`, `report/references.bib` and `report/brief_corrections.md`.
**Key results:**
- Confirmed the L_D typo, and found that N² has the same typo. Confirmed β2 is printed in ps/km. Jiang's generalization test is interpolation in peak power p ∈ [0.5, 1.5], i.e. N ∈ [0.71, 1.22] (`report/brief_corrections.md`).
- **Novelty:** Fesser et al. already apply the Wasserstein-Fourier distance to the NLSE (a single 2·sech time-extrapolation case, PINN only). Component A is reframed as an extension (D-012).
- The causal paper's Fourier features are deterministic periodic harmonics, so PINN+ uses those (D-013).
- The Zeng et al. 2026 cFNO reference could not be verified.
**Problems / surprises:** see the novelty reframing above.
**Next:** Phase 1 (git), then Phase 2.

## Checkpoint 1: Git and GitHub setup (2026-10-22)
**Status:** done
**What I did:** git init (main) and two commits. Created a private GitHub repo and pushed. Added the starter-repo submodule pinned at a41e838.
**Key results:** remote https://github.com/ShreyasVR2545/ultrafast-pinn-extrapolation. Teammate invite skipped (no username given).
**Commits:** `chore: scaffold repo structure, env check, benchmark scripts`; `docs(lit): literature notes, references and brief corrections`; `chore: add starter repo as pinned submodule`. Pushed: y.
**Next:** Phase 2, audit the starter repo and reproduce fig3, fig7 and fig8.

## Checkpoint 3: Independent ground truth (SSFM) and validation (2026-10-22)
**Status:** done
**What I did:** Symmetric SSFM + RK4IP (float64), analytic solutions, E/P/H. `tests/test_physics.py` has 14 tests. Grid study `scripts/03_validate_ssfm.py` produces `configs/grid.yaml`, `results/grid_study.json`, `figures/ssfm_validation.png` and 6 gallery figures (`figures/gallery_*.png`).
**Key results:** All physics tests pass: GVD analytic error <1e-8, N=1 soliton over 4 periods <1e-6, N=2 recurrence <1e-6, Hamiltonian drift <1e-6, second-order convergence, RK4IP agreement <1e-6. The starter repo's SSFM agrees to <1e-4, limited by its dz=2e-3. Chosen grids are in D-017. Every criterion is <1e-6 on the hardest cases.
**Sanity checks:** the N=2 soliton compresses at ξ≈π/4+kπ/2 (gallery_B_N2). Regime A N=4 shows the expected SPM broadening and saturation in normal dispersion.
**Problems / surprises:** two tests needed corrected setups, not looser tolerances (D-018).
**Commits:** physics tests, residual tests, grid study, figures. Merged to main, tag `m1-ground-truth`, pushed: y.

## Checkpoint 4: Datasets and pre-registered splits (2026-10-22)
**Status:** done
- `configs/splits.yaml` was committed in 278e132, before any model training.
- `scripts/04_generate_data.py` produced 37 fields (A: 15, B_sech: 11, B_gauss: 11), 146 MB, in 126 s wall with 6 workers. Zero grid-quality failures (edge energy, H drift <1e-6 for every field). Details in `results/data_summary.json`.
- `tests/test_splits.py` covers disjoint splits, a training loader that refuses test N and mislabelled files, the ξ≤2 mask for z_extrap, and a check that train/ never imports the eval loader.

## Checkpoint 5: Metrics and evaluation library (2026-10-22)
**Status:** done
- `eval/metrics.py`: field, intensity, phase and spectral errors, observables, and drift and grid-residual physics checks. `eval/spectral_shift.py`: WF distance and SPM predictor. `tests/test_metrics.py` has 10 tests (identical fields give 0, phase offset, shifted Gaussians give W = shift, analytic drift ≈ 0, observables vs closed form).
- `eval/evaluate.py`, `eval/aggregate.py` (tidy CSVs), `scripts/make_tables.py` (md + LaTeX, mean ± std and median over seeds), `scripts/run_sweep.py` (resumable).
