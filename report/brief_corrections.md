# Corrections to PROJECT_BRIEF.md from primary sources

| # | Brief says | Primary source says | What we do |
|---|---|---|---|
| 1 | arXiv text prints L_D = T0²/\|β2\|² (typo) | Confirmed (Jiang Eq. 4). **Also** N² is printed as γP0T0²/\|β2\|², the same typo. | Use L_D = T0²/\|β2\|, N² = γP0T0²/\|β2\| (dimensionally correct). |
| 2 | β2 units printed as ps/km | Confirmed: "β2 is approximately -20 ps/km" (Sec. 3) | Use ps²/km (SMF-28: β2 ≈ −21.7 ps²/km). |
| 3 | Jiang used "initial-condition MSE plus PDE-residual MSE" with L-BFGS | Confirmed. The initial condition uses N0 = 256 uniform points, the domain is [−24, 24]T0 × [0, 4]L_D, and L-BFGS stops on a relative-loss-change tolerance (their Eq. 8). | Our PINN uses Adam then L-BFGS (the brief's schedule). The paper's L-BFGS-only schedule is noted as a difference. |
| 4 | Jiang's generalization: trained on p ∈ {0.5, 0.7, …, 1.5}, tested in between | Confirmed. Note that p is *peak power*, not N, with p = 1 as reference. Since N² ∝ P0, their N range is √p ∈ [0.71, 1.22]. | Our Regime A training range N ∈ [1, 2] is wider and more nonlinear than theirs. Report this when comparing. |
| 5 | Fesser et al. "following [them], we apply this to the NLSE" (implied first application) | Fesser et al. already apply the WF distance to the focusing NLSE (h0 = 2 sech, time extrapolation t > π/4), PINN only. | Component A is framed as **replication and extension** (parametric N-extrapolation, data-driven models, closed-form SPM driver). See D-012. |
| 6 | PINN+ uses "random Fourier features in τ" (citing the causal paper) | The causal paper (Wang et al.) uses a **deterministic periodic** embedding (1, cos(kωτ), sin(kωτ)), ω = 2π/L, which enforces periodic BCs exactly. Random Fourier features come from a different paper (Tancik et al. / Wang et al. 2021). | Default: periodic harmonic embedding with period equal to the τ window, matching the periodic SSFM domain. Random features are available as a config option (not run unless time allows). D-013. |
| 7 | Causal paper venue: CMAME 421, 116813 (2024) | arXiv v1 lists no venue. I did not re-verify the journal page online. | Cite the CMAME entry, with the arXiv ID alongside. |
| 8 | Zeng et al., J. Lightwave Technol. 44(2), 665 (2026), conditional FNO | Could not be found in searches. | Marked **not verified**. It is not cited in the report unless the students confirm it. |
| 9 | Jiang et al. JLT 42(5), 1381 (2024) | A reference list gives J. Lightw. Technol. 42, 1381–1394 (2024), "Predicting Ultrafast Nonlinear Dynamics in Fiber Optics by Enhanced PINN". Issue number not verified. | Cite without the issue number. |
| 10 | Starter repo "six presets fig3 to fig8 with errors 0.48%–7.72%" | Confirmed (README). The license is MIT, (c) 2026 Xie Yulai. | None. |
