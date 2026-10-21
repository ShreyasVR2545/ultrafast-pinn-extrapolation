# CLAUDE.md — persistent rules (condensed from PROJECT_BRIEF.md; the brief is authoritative)

Resume a session: read this file, `PROGRESS.md`, `DECISIONS.md`, then continue the next phase in `PROJECT_BRIEF.md`.

## Setup answers (2026-10-22, do not ask again)
Repo `ultrafast-pinn-extrapolation`, private, owner ShreyasVR2545. venv (`.venv`, Python 3.12 from uv). Teammate: skipped.
Budget 12 GPU-h target, Colab allowed. Real deadline: presentation ~2026-10-26, so all compute must finish within ~1-2 days
wall-clock; the user asked not to cut scope unless necessary (see DECISIONS.md D-003).

## Operating rules
- Plan each phase briefly (goal, files, verification, compute) before acting. Log every decision in DECISIONS.md, every phase in PROGRESS.md.
- Every training script has `--smoke` (<60 s). Smoke first, then scale. Long jobs in background with logs on disk.
- Verify against physics (energy, analytic limits). Too-good results = leakage/bug until ruled out.
- Never fabricate numbers. Tables/figures are generated from `runs/` and `results/` by scripts only.
- Never weaken a test to pass. Seed everything; save config, git hash, versions, hardware in `meta.json` per run.
- Plain PyTorch/NumPy/SciPy/Matplotlib/PyYAML/neuraloperator. pathlib, Python scripts (Windows box).
- Splits in `configs/splits.yaml` are pre-registered and committed before training; never changed after seeing results.
- Hyperparameters tuned on the interpolation split only. PINN and DD-MLP share architecture and optimizer schedule.
- Hard stops only: human action needed (auth/admin), destructive git/file actions, core Regime A cannot fit budget,
  SSFM cannot be validated, research-question-changing finding. Everything else: decide, log, continue.

## Git
Conventional Commits (<=72 char subject). One branch per phase, merge `--no-ff` to main, push. Tags: m1-ground-truth,
m2-in-distribution, m3-main-results, m4-analysis, final. Never commit data/, runs/, *.npz, *.pt, files >5 MB.
Never force-push or rewrite main. `git status` before every commit.

## Physics conventions (verified numerically, see tests/)
i U_xi = (s/2) U_tautau - N^2 |U|^2 U ; s=+1 normal, s=-1 anomalous; L_D = T0^2/|beta2|, N^2 = gamma P0 T0^2/|beta2|.
Linear Fourier step: U^ *= exp(i s omega^2 h / 2). Residuals: f = -v_xi - (s/2)u_tt + N^2|U|^2 u ; g = u_xi - (s/2)v_tt + N^2|U|^2 v.
GVD Gaussian: U = (1 - i s xi)^(-1/2) exp(-tau^2/(2(1 - i s xi))). H = int[(s/2)|U_tau|^2 + (N^2/2)|U|^4].
Python: `.venv/Scripts/python` (Windows). Code lives in `src/nlse_ml` (add `src` to sys.path or `pip install -e .`).
