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

## Checkpoint 6: Models and training, in-distribution first (2026-10-22)
**Status:** done. The single-seed table comes from the final-config pilots, and the main sweep's seed 0 reproduces it with all 6 models.
**What I did:** Implemented the shared MLP (with an exact Taylor-mode derivative path), PINN-strict/extended, PINN+ (periodic Fourier features, causal ξ weights, gradient-norm loss weights), DD-MLP, cFNO (direct, conditioned on (N, ξ)) and cFNO-AR, plus `run_experiment.py`, `run_sweep.py`, evaluation, aggregation, tables and figures. All models pass `--smoke` in under 15 s.
**Key results** (Regime A, seed 0, final config; `report/tables/pilot_s4_all.md`, `results/pilot_*.csv`):
- interp rel-L2: cFNO 0.95%, DD-MLP 2.0%, PINN+ 6.3%, PINN-strict 8.3% (cFNO-AR 1.9%).
- near: DD 13%, cFNO 25%, PINN+ 36%, PINN-strict 46%. far: all models ≈ 100-140% (cFNO-AR 241%).
- Cost per run: PINN-strict 15 min, PINN+ 24 min, DD-MLP 4.5 min, cFNO 6.4 min.
**Sanity checks:** error grows with N for every model. The DD-MLP fits its training N at ~1-4%. Our PINN matches the starter repo on its own fig3 problem (3.5% vs 3.0%), which validates the implementation.
**Problems / surprises:** the first PINN pilots failed (16-94% in-distribution). Three causes, all ours and all fixed: an L-BFGS-starved schedule, a forced causal-ε advance, and PINN+ overfitting a fixed L-BFGS collocation set (D-025 to D-029). The PINNs remain worse than the data-driven models *in distribution* at equal optimizer steps, and they cost 3-5× more wall-clock.
**Next:** the Phase 7 main sweep was launched (blocks a→b→c→d, 3 seeds), estimated ~17 GPU-h (D-028 scope: B_gauss z_extrap dropped unless time remains).

## Checkpoint 7: Main experiments (2026-10-22, finished 21:19)
**Status:** done. 84/84 jobs ok (0 diverged, 0 crashed) in 14.8 h. Scope per D-028: B_gauss z_extrap was not run.
**What I did:** `run_sweep.py` blocks a→d, 3 seeds. Aggregated to `results/main_*.csv`, tables in `report/tables/main_*`, figures `fig_extrap_N_*`, `fig_extrap_z`, `fig_heatmaps_compare*`, `fig_spectra_far*`, `fig_energy_drift*`, `fig_pareto`. Analyses A and B ran on all runs (`results/analysis_*`). All numbers and findings are generated in `report/results_summary.md` (`scripts/make_summary.py`).
**Key results (plain language; numbers in report/results_summary.md):**
- In distribution, the data-driven models (cFNO, DD-MLP) are 3-10× more accurate than every PINN variant, in every regime, at equal optimizer steps and 2-3× less wall-clock.
- Near extrapolation (Regime A, N = 2.25-2.5), the DD-MLP is still best. Robust across seeds.
- Far extrapolation (A: N = 3-4; B: N = 2-3), **every model fails on the complex field** (≥ ~100% rel-L2, phase lost). Physics-informed models keep the intensity and energy far more plausible: the lowest far-N energy drift in all three regimes is PINN-extended or PINN+, and the DD-MLP is worst on far-N intensity in all regimes. The best far-N intensity model is PINN-extended (A, robust), cFNO-AR (B_sech, robust) or PINN+ (B_gauss, not robust).
- z-extrapolation has no class winner: DD-MLP is best in A and worst in B_sech, where the autoregressive cFNO is best.
- In Regime B the DD-MLP memorizes its training N (accurate at train N, ~10× worse at interp N in between). The PINN error curves are smooth in N and smallest at the N = 1 soliton.
- Cost: an SSFM at the best model's accuracy takes ≤ 9 ms in Regime A, less than model inference. In the hardest B cases it takes up to ~230 ms, but it is never less accurate, and the models' training cost comes on top.
**Sanity checks:** error grows with N and beyond ξ = 2 for every model. There is no too-good extrapolation, so no leakage concern. Seed-to-seed std is small relative to the between-model gaps that are labelled robust.
**Problems / surprises:** the PINN in-distribution fit is markedly worse in Regime B (hyperparameters were tuned on Regime A interp only). cFNO-AR rollouts diverge on some seeds at far N (A: huge intensity errors).
**Commits:** results + figures per block. Merged to main, tag `m3-main-results`, pushed: y.
**Next:** Phase 8 ablations (running, 23 jobs, ~4.3 h), then fine-tuning (Phase 9.3), then Phase 10.

## Checkpoint 8: Ablations (2026-10-23, finished 01:57)
**Status:** done. 23/23 runs ok, 4.6 h. Regime A, n_extrap. Results in `results/ablations_*.csv`, figures `fig_ablation_range_width` and `fig_ablation_components`, findings 24-27 in `report/results_summary.md`.
- Training-range width (seeds 0-1): PINN-strict's error vs distance-to-range is nearly independent of the range width; the data-driven models trained narrow degrade faster.
- PINN+: the Fourier features are essential (removing them takes interp from 6.2% to 115%). Causal weighting helps slightly; adaptive weights are neutral.
- Collocation 10k/20k/50k: no clear effect. **IC weight: w_ic=1 roughly halves PINN-strict's in-distribution error** (4.9% vs 8.3%, single seed). w_ic was never tuned, so this is logged as a limitation and does not change the ranking.
- Data efficiency: with 2 training N values the data-driven models fail (DD-MLP 82% interp); with 4 they match 6.
- Skipped: the float64 L-BFGS ablation (12× slower on this GPU, D-006) and noise robustness (time).

## Checkpoint 9: Novel analyses (2026-10-23)
**Status:** done (A, B, and the optional fine-tuning; inverse retrieval skipped for time).
- A: the closed-form SPM phase N² asinh ξ predicts PINN error (ρ ≈ 0.84-0.87) better than data-driven error (0.72-0.77). The WF distance to the training spectra is weaker (0.44-0.64). Extension of Fesser et al., not novel (D-012).
- B: within a fixed N, energy drift ranks predictions by accuracy (ρ = 0.63 N-extrap, 0.74 z-extrap). The grid PDE residual is anti-correlated across model families in N-extrap (ρ = −0.30).
- Fine-tuning (seed 0, 6000 physics-only L-BFGS steps, ~10-13 min): no case reaches the model's own in-distribution error (A N=3: 134% → 60%; A N=4: 135% → 120%). SSFM reaches that accuracy in milliseconds. Honest conclusion: SSFM wins.
- Optional inverse retrieval (9.4): not done (time).
**Commits:** ablation and fine-tune results and figures. Merged, tag `m4-analysis`, pushed: y.
