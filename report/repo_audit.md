# Audit of the starter repo `xieyulai/fiber-nlse-pinn` (pinned at a41e838)

I read all files (`nlse.py`, `ssfm.py`, `pinn.py`, `main.py`, `visualize.py`, `scripts/`, `baselines/SUMMARY.md`) before running anything. The code was not modified.

## Equation, signs and normalization
- Implements Jiang et al. Eq. 3: `h_z + (α/2)h + i(β2/2)h_tt − (β3/6)h_ttt − iN²[|h|²h + i s(|h|²h)_t − τ_R(|h|²)_t h] = 0`, with β2 → sgn(β2) (`beta2_norm`), ζ = Z/L_D, τ = T/T0, h = H/√P0.
- `L_D = T0²/|β2|` and `N² = γ P0 L_D` are implemented correctly. The paper's printed typos are **not** reproduced in the code.
- PINN residual (`pinn.py: pde_residual`): f = u_z − (β2/2) v_tt + N²|h|² v, g = v_z + (β2/2) u_tt − N²|h|² u. I re-derived this by hand. It is the real and imaginary split of `h_z = −i(β2/2) h_tt + i N²|h|² h`, so it is **consistent with our convention** (our f, g are the same equations with the roles of the real and imaginary parts swapped, see `src/nlse_ml/models/pinn.py`).
- SSFM linear operator: `exp(i (β2/2) ω² dz/2)` per half step with numpy/torch FFT order. **Same as ours** (`exp(i s ω² h/2)`).

## SSFM scheme
- Symmetric split-step: L(dz/2), N(dz), L(dz/2). It is not merged across steps (correct, just 2× more FFTs than needed).
- SPM uses the exact phase rotation. SS and IRS use a first-order Euler correction inside the nonlinear step, so the scheme is only **first order** when SS or IRS is on (fig5, fig6). Not relevant to fig3, fig7 or fig8.
- Grids are hard-coded per preset: τ ∈ [−24, 24) with n_t = 2048 and n_z = 2000 steps over 4 L_D (dz = 2e-3). fig5 uses n_t = 6144 and n_z = 3e5. Runs on the GPU in complex128. There is no convergence or edge-energy check.

## PINN
- Network: tanh MLP (z, t) → (u, v). Inputs are mapped to [−1, 1] using the domain bounds. Default PyTorch init (not Xavier).
- Loss: `ic_weight · (MSE_u0 + MSE_v0) + mean(f²) + mean(g²)`. The IC has 256 points (linspace on the closed interval). There are **no boundary terms** (the window is wide enough that the field is ≈ 0 at the edges).
- Collocation: 50k Latin hypercube points over (z, t), **fixed** unless `--resample` is given.
- Optimizer: optional Adam (lr 1e-3), then L-BFGS (lr 1, strong Wolfe, `max_iter=50` per "epoch", tolerance_change 1e-13). "500 L-BFGS epochs" means up to 25,000 L-BFGS iterations. There is no stopping rule beyond the iteration cap (the paper used a relative-change tolerance, its Eq. 8).
- Evaluation: the PINN on a 201 × 2048 grid against the SSFM interpolated linearly onto it. Relative L2 of the complex field over the whole (z, t) domain. Same metric as the paper, but on a denser grid (the paper used 201 × 256).

## Presets vs paper figures (mismatch)
The preset names are **off by two** from the arXiv figure numbering:

| Preset | Content | arXiv figure |
|---|---|---|
| fig3 | GVD + SPM, normal, Gaussian, N² = 1.3 | Fig. 5 |
| fig4 | TOD + SPM, β2 = 0 | Fig. 6 |
| fig5 | GVD + SPM + SS, anomalous, 10 L_D | Fig. 7 |
| fig6 | IRS, anomalous, N² = 1.96 | Fig. 8 |
| fig7 | fundamental soliton, sech, N = 1 | Fig. 9 |
| fig8 | second-order soliton (N² = 1 PDE with a 2·sech IC, equivalent to N = 2) | Fig. 10 |

(The arXiv Figs. 3 and 4 are the GVD-only and TOD-only cases, which have no preset.)

## Suspicious items and possible bugs
1. **`main.py: make_params` drops `pulse_amplitude`** whenever `--n-t`, `--n-z` or `--pulse-type` is overridden. The rebuilt `NLSEParams` omits it, so it resets to 1.0. For fig8 this silently turns the N = 2 soliton into an N = 1 soliton. It does not affect the README commands, which do not override these flags.
2. **The causal loss does not detach the weights.** In `causal_pde_loss`, `causal_w = exp(−ε·cumsum(bin_means))` keeps the gradient. In Wang et al. the weights are treated as constants (stop-gradient). Our PINN+ detaches them. Not used by the README commands.
3. **The self-steepening term omits N²** relative to the paper's Eq. 3. This is documented by the author as a deliberate deviation (fig5 only).
4. **fig8's 0.48% uses 8-segment domain decomposition** (time-marching: each segment's IC is the previous segment's PINN prediction). That makes it a different method from the single-network PINN in the paper. Its wall time is also 6× the others (4586 s in `baselines/SUMMARY.md`).
5. Single seed (42) only. The error is computed on interpolated SSFM data, which is fine because the SSFM is saved every dz = 2e-3.
6. **No grid-adequacy or conservation checks** on the SSFM reference.
7. `main.py` creates `outputs/` inside the repo at import time. Harmless.

## Reproduction (Phase 2)
Wrapper: `scripts/02_reproduce_repo.py` (README flags unchanged, seeds {0, 1, 2}). The numbers are generated in `results/repro_repo.md` and `results/repro_repo.csv`, and summarized in PROGRESS.md Checkpoint 2.
Scope reduction: fig8 runs with seed 0 only, because each fig8 run takes ~75 min of GPU (D-016).
