# Physics-Informed vs Data-Driven Learning for Nonlinear Ultrashort Pulse Propagation: A Controlled Extrapolation Study

Ultrafast Optics course project (IIT Bombay). We test whether physics-informed neural networks (PINNs) extrapolate
better than data-driven models (MLP, conditional FNO) when ultrashort pulses enter propagation regimes not seen in
training: higher soliton order N, or longer propagation distance xi. We also test whether spectral shift and
conservation-law drift predict where each model fails.

*Status: under construction. See `PROGRESS.md`.*

## Setup

```bash
# Python 3.12
python -m venv .venv
# Windows: .venv\Scripts\activate    macOS/Linux: source .venv/bin/activate
pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu128   # or plain `pip install torch==2.7.1` for CPU
pip install -r requirements.txt
python scripts/00_env_check.py
pytest
```

## Conventions

Normalized NLSE (Agrawal): `i dU/dxi = (s/2) d2U/dtau2 - N^2 |U|^2 U`, with U = A/sqrt(P0), xi = z/L_D, tau = T/T0,
s = sgn(beta2) (+1 normal, -1 anomalous), L_D = T0^2/|beta2|, L_NL = 1/(gamma P0), N^2 = L_D/L_NL.
All conventions are verified numerically in `tests/`.
