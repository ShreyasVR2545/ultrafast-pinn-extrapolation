# DECISIONS

One line of reason per decision. Newest at the bottom.

- **D-001 (Phase 0) Setup answers.** Repo `ultrafast-pinn-extrapolation`, private; venv; 12 h budget, Colab allowed; teammate invite skipped (default, the picker could not take free text; the user can send a username later).
- **D-002 (Phase 0) Python 3.12 venv, not system 3.14.** torch/neuraloperator wheels are most reliable on 3.12; a uv-managed CPython 3.12.13 was already on the machine.
- **D-003 (Phase 0) Budget interpretation.** The user said the presentation is in ~4 days and asked not to reduce scope unless necessary, "as long as the entire project compute fits in the next 1-2 days". Plan to the 12 GPU-h target, but allow up to ~24 h of wall-clock before applying the brief's automatic reductions (seeds, then Regime B, then ablations).
- **D-004 (Phase 0) torch 2.7.1+cu128.** The RTX 5070 Laptop GPU is Blackwell (sm_120), which needs CUDA 12.8 wheels.
- **D-005 (Phase 0) Compute tier: Tier 1 (local GPU, 8 GB VRAM), no Colab.** The PINN at 50k collocation points uses 1.8 GB. Colab T4s would be slower than this GPU.
- **D-006 (Phase 0) float64 PINN training is impractical on this GPU** (1.5 vs 18 Adam it/s, consumer FP64 rate). The f64 L-BFGS ablation will be one short run only.
- **D-007 (Phase 0) Taylor-mode (forward) derivative propagation for PINN residuals.** It is exactly equal to nested autograd (max diff ~1e-16 in float64, see tests/test_residuals.py) and 1.4-1.7x faster. Autograd stays as the reference path.
- **D-008 (Phase 0) Concurrent GPU jobs reduce total throughput** (3 jobs x 9.9 it/s < 1 job x 41 it/s at 20k points), so PINN runs go through the GPU sequentially. Cheap CPU work (SSFM, aggregation) runs in parallel.
- **D-009 (Phase 0.5) The equation convention matches Jiang et al.** Their Eq. 3, with beta2 → s, is algebraically our NLSE. The L_D and N² typos are confirmed (report/brief_corrections.md #1-2).
- **D-010 (Phase 0.5) Shortlisted technique 1: periodic Fourier-feature embedding in tau** (Wang, Sankaran & Perdikaris). It enforces the periodic BC of the SSFM window exactly. It lives in PINN+ (`models/pinn_plus.py`) and can be toggled for the ablation.
- **D-011 (Phase 0.5) Shortlisted technique 2: last-layer transfer / test-time fine-tuning** (Fesser et al. Sec. 4, following Pellegrin et al.). It is used only in the optional PINN-finetune model (Phase 9.3), where both full fine-tuning and last-layer-only are labelled. It is never used to touch splits.
- **D-012 (Phase 0.5) Novelty reframing for component A.** Fesser et al. already computed WF distances for the NLSE (single 2·sech time-extrapolation case, PINN only). Component A is therefore called an *extension* of Fesser et al.: parametric N-extrapolation, PINN vs data-driven, and a closed-form SPM bandwidth driver. It is not called novel. Component B (conservation drift as a label-free estimator) had no prior NLSE work in our searches, so "to our knowledge".
- **D-013 (Phase 0.5) PINN+ Fourier features are deterministic periodic harmonics by default**, as in the cited causal paper, not random. A random-feature option exists in config.
- **D-014 (Phase 0.5) Shortlisted technique 3: N-curriculum** (Krishnapriyan et al. curriculum regularization). It is held as a failure-handling fallback only and is not part of the main comparison unless training fails.
- **D-015 (Phase 1) Starter repo submodule pinned** at `a41e83879d5d1e992f93ba4188c454a2370b94ce` (xieyulai/fiber-nlse-pinn, MIT). Remote: https://github.com/ShreyasVR2545/ultrafast-pinn-extrapolation (private). No teammate invited (D-001).
- **D-016 (Phase 2) Repo reproduction scope.** fig3 and fig7 use 3 seeds; fig8 uses seed 0 only (8-segment decomposition, ~75 min per run at the README's 4586 s). It is a side reproduction, so its GPU time goes to the core study. More fig8 seeds run only if the GPU is idle later.
