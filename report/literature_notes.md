# Literature notes

Status key: **read**: full text read (PDF text extracted into `report/papers/`, gitignored). **abstract**: abstract page only. **not verified**: could not access or locate it; details come from the project brief only.

---

## 1. Jiang et al., PINN for nonlinear dynamics in fiber optics (foundation): **read**
X. Jiang, D. Wang, Q. Fan, M. Zhang, C. Lu, A. P. T. Lau, "Physics-Informed Neural Network for Nonlinear Dynamics in Fiber Optics," *Laser & Photonics Reviews* 16(9), 2100483 (2022). doi:10.1002/lpor.202100483. arXiv:2109.00526.
- Normalized NLSE (their Eq. 3): `h_z + (alpha/2) h + (i/2) beta2 h_tt - (1/6) beta3 h_ttt - i N^2 [|h|^2 h + ...] = 0`. With beta2 → sgn(beta2) this is identical to our `i U_xi = (s/2) U_tt - N^2|U|^2 U` (multiply by i).
- Typos in Eq. 4: `L_D = T0^2/|beta2|^2` and `N^2 = gamma P0 T0^2/|beta2|^2`, where both should have |beta2| to the first power. Sec. 3 also gives beta2 as "≈ -20 ps/km" (should be ps^2/km).
- Setup: domain [-24 T0, 24 T0] x [0, 4 L_D]; Nf = 50,000 Latin-hypercube collocation points; N0 = 256 initial-condition points; 4 hidden layers x 100, tanh; L-BFGS until the relative loss change is below eps (their Eq. 8). Error metric: relative L2 over a 201 x 256 reference grid.
- Parametric generalization (Sec. 3.3, Fig. 14): input (z, t, p), with p = peak power in [0.5, 1.5] in steps of 0.1. Six p values {0.5, 0.7, ..., 1.5} are used in the loss and five held out. Error stays below 1.8e-2 and grows with p. **Interpolation only**, which confirms the gap stated in the brief.
- PINN vs data-driven NN (Sec. 4.1, Fig. 17): same 4x100 tanh network, L-BFGS, 5 repeats. The DDNN needs data at 5 or more distances, and the PINN wins even when the DDNN has lots of data. In-distribution only, so no extrapolation comparison.
- Fig. 16: 5,000 collocation points were already "enough" for good accuracy in the GVD+SPM case.
- **Use:** baseline method, hyperparameters (architecture, LHS, L-BFGS), and the gap motivating the project.

## 2. Starter repo: xieyulai/fiber-nlse-pinn: **README read** (code audited in Phase 2)
MIT license, (c) 2026 Xie Yulai. Presets fig3 to fig8 with seed-42 errors: fig3 2.61%, fig4 7.72%, fig5 6.60%, fig6 3.31%, fig7 2.88%, fig8 0.48%.
- Loss = ic_weight x IC MSE + residual MSE; 50k random collocation points; tanh MLP; inputs mapped to [-1, 1].
- Recommended runs: fig3 uses L-BFGS only (500 "epochs"); fig7 and fig8 use 5000 Adam + 300 L-BFGS epochs with ic-weight 100. fig8 uses `--decompose 8` (meaning to be audited).
- The README says it departs from the paper's self-steepening factor (it uses s instead of N^2 s).
- **Use:** cross-check and reproduction target only (Phase 2).

## 3. Fesser, D'Amico-Wong, Qiu (2023), extrapolation failures in PINNs: **read**
L. Fesser, L. D'Amico-Wong, R. Qiu, "Understanding and Mitigating Extrapolation Failures in Physics-Informed Neural Networks," arXiv:2306.09478 (v2, Nov 2023). Preprint, no venue listed.
- Claim: extrapolation failure in time is caused by shifts in the **support** of the Fourier spectrum over time ("spectral shifts"), not by high frequencies. Model size, activation and number of collocation points hardly change extrapolation once the model fits well in-domain.
- Wasserstein-Fourier distance: normalize spectra by their L1 norm, then W(F1, F2) = Σ_x |F1(x) - F2(x)| over CDFs (the 1D Wasserstein-1 distance on a unit grid). Weighted WWF(f) = Σ_{s∈I} Σ_{t∈E} (T_max + s - t) W(f_s, f_t).
- PDEs: Allen-Cahn, Burgers, heat, diffusion, diffusion-reaction, Beltrami flow, **and the focusing NLS** `i h_t + 0.5 h_xx + |h|^2 h = 0`, h0 = 2 sech(x), x ∈ [-5, 5], with training on t ∈ [0, π/4] and extrapolation to (π/4, π/2]. This is a time extrapolation of a single N = 2 soliton. WWF ≈ 0.034/0.036 (real/imag). Interpolation error is 1e-5, but extrapolation max relative L2 is 0.94 (real) and 4.27 (imag).
- Remedy: transfer learning that freezes all but the last layer, trained on a family of nearby PDEs (NLS with 1.95, 2.05, 2.1 sech) and evaluated on 2 sech. This cuts NLS extrapolation error by about 51-55%.
- **Use:** method for component A. **Novelty impact:** the WF distance has already been applied to the NLSE, but only to time extrapolation, for PINNs only, with a single parameter value. Our component A is an *extension* (parametric N-extrapolation, data-driven models, a closed-form SPM driver), not a first application (see DECISIONS D-012).

## 4. Wang, Sankaran, Perdikaris, causal training: **read** (arXiv v1)
S. Wang, S. Sankaran, P. Perdikaris, "Respecting causality is all you need for training physics-informed neural networks," arXiv:2203.07404 (2022). Published as "Respecting causality for training physics-informed neural networks," *CMAME* 421, 116813 (2024). The journal details come from the brief and were not re-verified online.
- The temporal domain is split into Nt chunks with weights w_i = exp(-ε Σ_{k<i} L_r(t_k)), where w_1 = 1. The loss is the mean of w_i L(t_i), and the weights are not differentiated.
- ε is annealed through [1e-2, 1e-1, 1, 10, 100]. A stage ends when min_i w_i > δ = 0.99. They recommend λ_ic = 1e3.
- Fourier features: a deterministic periodic embedding v(x) = (1, cos(ωx), sin(ωx), ..., cos(mωx), sin(mωx)), with ω = 2π/L, enforces periodic BCs exactly. Time is concatenated raw. They also use a "modified MLP".
- Allen-Cahn: relative L2 goes from 0.498 (original PINN) to 1.4e-3 (causal MLP) to 1.4e-4 (causal modified MLP).
- **Use:** PINN+ components (causal weights in xi, periodic Fourier features in tau). The README must note the causal method is patented for commercial use.

## 5. Krishnapriyan et al. (NeurIPS 2021): **abstract**
A. S. Krishnapriyan, A. Gholami, S. Zhe, R. M. Kirby, M. W. Mahoney, "Characterizing possible failure modes in physics-informed neural networks," NeurIPS 2021, arXiv:2109.01050.
- PINNs fail on convection, reaction and diffusion problems with large coefficients. The cause is an ill-conditioned loss landscape from the soft PDE penalty, not a lack of expressivity.
- Remedies: curriculum regularization (start with an easy PDE coefficient and increase it) and sequence-to-sequence (time-marching) training. They report 1-2 orders of magnitude lower error.
- **Use:** fallback remedies in `<failure_handling>` (curriculum in N, marching in xi). Background for "why PINNs fail at large N".

## 6. neuraloperator documentation (`neuralop.models.FNO`, v1.0.2 installed): **read**
- `FNO(n_modes=(k,), in_channels, out_channels, hidden_channels, n_layers=4, positional_embedding="grid", domain_padding=None, ...)`. Dimensionality is inferred from len(n_modes). The input is (batch, channels, n_tau). The default "grid" embedding appends a coordinate channel on [0, 1].
- **Use:** cFNO implementation (Phase 6).

## 7. Newer and related work (2023+), from searches on 2026-10-22
- Jiang et al., "Predicting Ultrafast Nonlinear Dynamics in Fiber Optics by Enhanced Physics-Informed Neural Network," *J. Lightwave Technol.* 42, 1381-1394 (2024). **Not verified** (seen only in a reference list and a PolyU research page). It is an enhanced PINN for the GNLSE, which would be the natural "PINN+" comparison point.
- X. Jiang et al., "FiberKAN: Kolmogorov-Arnold Networks for Nonlinear Fiber Optics," arXiv:2504.18833 (2025). **abstract**: KANs beat equal-size MLPs. The abstract describes no extrapolation study.
- J. Musgrave, S.-W. Huang, "Fourier Domain Physics Informed Neural Network," arXiv:2409.19895 (2024). **abstract**: physics priors in both time and Fourier domains for generalized pulse propagation, plus an SNR study. Relevant to the optional inverse-retrieval extra.
- Murugan & Kanagaraj, "Bidirectional Fourier-Enhanced Deep Operator Network for Spatio-Temporal Propagation in Multi-Mode Fibers," arXiv:2512.15474 (2025). **title/snippet only**: an operator surrogate conditioned on physical parameters, evaluated on unseen cases.
- Chuprov et al. (Huawei), Meta-PINN for fiber parameter generalization. **Not verified** (search snippet only).
- He et al., FNO for optical fiber channel modeling, *J. Lightwave Technol.* (2023). **Not verified** (from a review's reference list, arXiv:2501.07830).
- Zeng et al., *J. Lightwave Technol.* 44(2), 665 (2026), conditional FNO (from the brief). **Not verified**: searches did not find it. Do not cite it in the report without checking.
- **Novelty check (components A and B):** no paper was found that tests N-extrapolation of PINNs against data-driven models for the fiber NLSE. Fesser et al. is the only spectral-shift work touching the NLS (see §3). No paper was found using conservation drift as a label-free error estimator for NLSE surrogates. Absence from a few searches is weak evidence, so the report should say "to our knowledge".
