# Figure index

All figures are in `figures/` as 300-dpi PNG plus PDF. Commands run from the repo root. "From results/" means the
figure regenerates from committed CSVs on a fresh clone. "Needs data/ + runs/" means it needs the generated SSFM
data and run predictions (`scripts/04_generate_data.py` and the sweep).

| Figure | What it shows | Suggested slide | Regenerate |
|---|---|---|---|
| `ssfm_validation` | (a) second-order SSFM convergence on the hardest cases; (b) energy and Hamiltonian drift < 1e-6; (c) edge-energy window adequacy; (d) agreement with analytic GVD and soliton solutions | Methods: "a ground truth we trust" | `python scripts/03_validate_ssfm.py --figures-only` (CPU, ~10 min) |
| `gallery_gvd_only`, `gallery_spm_only` | Pure dispersion broadening and pure SPM spectral broadening (intensity and dB spectrum vs ξ) | Physics background | `python scripts/03_validate_ssfm.py --gallery-only` |
| `gallery_A_N2`, `gallery_A_N4` | Regime A (normal GVD + SPM): training-range edge (N=2) vs far test (N=4), with much wider spectrum and pulse | Why N-extrapolation is hard | same as above |
| `gallery_B_N1`, `gallery_B_N2` | Regime B: fundamental soliton (invariant) vs N=2 breather (period π/2) | Regime B setup | same as above |
| `fig_extrap_N_regimeA` | Complex-field and intensity rel-L2 vs N, all models, training range shaded, mean ± std over 3 seeds | **Main result 1** | `python scripts/make_figures.py --only extrap_N` (from results/) |
| `fig_extrap_N_B_sech`, `fig_extrap_N_B_gauss` | Same for Regime B (anomalous dispersion) | Main result, Regime B | same |
| `fig_extrap_z` | Per-slice error vs ξ for z-extrapolation (trained on ξ ≤ 2) | **Main result 2** | `python scripts/make_figures.py --only extrap_z` (from results/) |
| `fig_heatmaps_compare` (+ `_B_sech`) | True / predicted intensity and log error maps at one interp N and one far N, seed 0 | Visual intuition | `python scripts/make_figures.py --only heatmaps_compare` (needs data/ + runs/) |
| `fig_spectra_far` (+ `_B_sech`) | Predicted vs true output spectrum (dB) at ξ=4 for the far-extrapolation N | Spectral failure modes | `python scripts/make_figures.py --only spectra_far` (needs data/ + runs/) |
| `fig_energy_drift` (+ `_B_sech`) | Energy drift vs ξ per model, interp vs far | Physical plausibility | `python scripts/make_figures.py --only energy_drift` (from results/) |
| `fig_pareto` | Accuracy vs wall-clock: model training (filled), model inference (hollow), SSFM cost ladder (stars) | **Cost: is it worth it?** | `python scripts/make_figures.py --only pareto` (from results/) |
| `fig_error_vs_spectral_shift` | Analysis A: per-slice error vs WF distance to the training spectra and vs excess SPM phase, Spearman ρ with 95% CI | Analysis A | `python scripts/09_novel_analyses.py --figures-only` (from results/) |
| `fig_drift_vs_error` | Analysis B: true error vs label-free energy drift, Hamiltonian drift and grid PDE residual, per model | Analysis B | same |
| `fig_energy_vs_phase` | Energy drift vs phase error: conserving energy does not certify the phase | Analysis B caveat | same |
| `fig_ablation_*` | Phase 8 ablations (training-range width, PINN+ components, collocation, IC weight, data efficiency) | Backup / Q&A | `python scripts/make_ablation_figures.py` (from results/) |
| `fig_finetune` | Test-time physics fine-tuning: error vs wall-clock at far N, against the SSFM time | Optional extra | `python scripts/make_ablation_figures.py` (from results/) |

Every number shown in these figures comes from files in `results/` that scripts produce. None was typed by hand.
