# Physics-Informed vs Data-Driven Learning for Nonlinear Ultrashort Pulse Propagation: A Controlled Extrapolation Study

Ultrafast Optics course project (IIT Bombay). We test whether physics-informed neural networks (PINNs) extrapolate
better than data-driven models (MLP, conditional FNO) when ultrashort pulses enter propagation regimes not seen in
training: higher soliton order N, or longer propagation distance ξ. We also test whether spectral shift and
conservation-law drift predict where each model fails.

Foundation: Jiang et al., *Laser Photonics Rev.* 16, 2100483 (2022). Their parametric PINN was tested by
interpolation only; this repository tests **extrapolation** under a pre-registered protocol
(`configs/splits.yaml`, committed before any training).

Status and results: `PROGRESS.md` (phase log), `DECISIONS.md` (every design decision), `report/`.

## Setup

```bash
# Python 3.12
python -m venv .venv
# Windows: .venv\Scripts\activate    macOS/Linux: source .venv/bin/activate
pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu128   # or plain `pip install torch==2.7.1` for CPU
pip install -r requirements.txt
git submodule update --init       # starter repo (only for Phase 2 reproduction and one cross-check test)
python scripts/00_env_check.py
pytest
```

## Conventions

Normalized NLSE (Agrawal): `i ∂U/∂ξ = (s/2) ∂²U/∂τ² − N² |U|² U`, with U = A/√P0, ξ = z/L_D, τ = T/T0,
s = sgn(β2) (+1 normal, −1 anomalous), L_D = T0²/|β2|, L_NL = 1/(γP0), N² = L_D/L_NL = γP0T0²/|β2|.
SSFM linear step: Û ← exp(i s ω² h/2) Û. PINN residuals for U = u + iv:
`f = −v_ξ − (s/2)u_ττ + N²(u²+v²)u`, `g = u_ξ − (s/2)v_ττ + N²(u²+v²)v`.
Hamiltonian `H = ∫[(s/2)|U_τ|² + (N²/2)|U|⁴]dτ`. All are verified numerically in `tests/`.
Physical scale for the report: SMF-28 at 1550 nm, β2 ≈ −21.7 ps²/km, γ ≈ 1.3 W⁻¹km⁻¹.

## Information available to each model

| Model | Sees labels (SSFM fields)? | Sees PDE? | Collocation N-range | Notes |
|---|---|---|---|---|
| PINN-strict | IC only (analytic, N-independent) | yes | train N range | pure extrapolation |
| PINN-extended | IC only | yes | train + test N range (no labels) | uses physics at test conditions (its legitimate advantage); N-extrapolation only |
| PINN+ | IC only | yes | train N range | periodic Fourier features in τ, causal weights in ξ, adaptive loss weights |
| DD-MLP | full fields at train N | no | n/a | same trunk, optimizer schedule and batch size as the PINN |
| cFNO | full fields at train N | no | n/a | (U0, N, ξ) → U(ξ); can be queried at any ξ |
| cFNO-AR | full fields at train N | no | n/a | step operator U(ξ) → U(ξ+0.02 … ξ+0.1), rolled out from U0 |
| SSFM | n/a | exact | n/a | reference; wall-clock at matched accuracy (`scripts/05_ssfm_cost.py`) |

Fairness: every model gets the same number of optimizer steps (wall-clock is logged too). Hyperparameters are
tuned only on the interpolation split. In z-extrapolation, every model, including every PINN, sees only ξ ≤ 2.

**Causal training note:** PINN+ uses the causal residual weighting of Wang, Sankaran & Perdikaris (CMAME 2024). That
method is patented for commercial use. Academic use, as here, is fine.

## Reproduce

| Step | Command | Approx. runtime (RTX 5070 Laptop) |
|---|---|---|
| Environment report and benchmarks | `python scripts/00_env_check.py`, `python scripts/01_benchmark.py` | 2 min |
| Starter-repo reproduction | `python scripts/02_reproduce_repo.py --presets fig3 fig7 --seeds 0 1 2` | ~1 h |
| SSFM validation, grid, gallery | `python scripts/03_validate_ssfm.py` | 15 min (CPU) |
| Datasets | `python scripts/04_generate_data.py` | 2 min (CPU, 6 workers) |
| SSFM cost ladder | `python scripts/05_ssfm_cost.py` | ~10 min (CPU) |
| One experiment | `python scripts/run_experiment.py --model pinn_strict --regime A --exp n_extrap --seed 0` | see `PROGRESS.md` |
| Main sweep | `python scripts/run_sweep.py --blocks a b c d` | see `PROGRESS.md` |
| Tables | `python scripts/make_tables.py --aggregate` | seconds |
| Figures | `python scripts/make_figures.py` | ~1 min |

Every script supports `--smoke` (under 60 s) where it trains or simulates.
