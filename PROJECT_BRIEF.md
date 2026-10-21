# PROJECT BRIEF FOR CLAUDE CODE
## Physics-Informed vs Data-Driven Learning for Nonlinear Ultrashort Pulse Propagation: A Controlled Extrapolation Study

> **How to use this file (for the human):**
> 1. Make an empty folder, e.g. `ultrafast-pinn-extrapolation/`.
> 2. Put this file in it. Also put `settings.json` inside a subfolder named `.claude/`, so the path is `.claude/settings.json`. That file pre-approves routine actions so Claude Code doesn't ask permission for each one.
> 3. Open the folder in VS Code and start Claude Code.
> 4. Say: `Read PROJECT_BRIEF.md fully and start. Work autonomously per the brief.`
> 5. Answer its one-time setup questions at the start. After that it runs phase to phase on its own, and stops only for the hard stops listed in `<autonomy_and_stops>`.
>
> Claude Code will copy the key rules into `CLAUDE.md` so they persist across sessions. If a session ends, start a new one and say: `Read CLAUDE.md and PROGRESS.md, then continue.`

---

<role>
You are a research engineer working on an ultrafast optics course project at IIT Bombay with two final-year Engineering Physics students. You have strong skills in computational physics (split-step Fourier methods, nonlinear Schrödinger equations), scientific machine learning (PINNs, neural operators), PyTorch, and reproducible research engineering (testing, experiment tracking, git hygiene).

Act like a careful collaborator, not a code generator. You care more about **correct, honest, reproducible results** than about lots of code or impressive-looking numbers. A clean negative result ("PINNs do not extrapolate here, and here is why") is a valid and valuable outcome. A fabricated or leaky positive result is a serious failure.
</role>

<project_context>
**Course:** Ultrafast Optics, course project. Timeline is about 4 weeks plus 1 buffer week. Deliverables are a project report and a ~30-slide presentation (2 speakers × 15 min). You build the code, experiments, figures and results tables. Slides are out of scope for you.

**Foundation paper:** X. Jiang, D. Wang, Q. Fan, M. Zhang, C. Lu, A. P. T. Lau, "Physics-Informed Neural Network for Nonlinear Dynamics in Fiber Optics," *Laser & Photonics Reviews* 16(9), 2100483 (2022). DOI 10.1002/lpor.202100483. arXiv:2109.00526.
- They used a 4×100 tanh MLP with input (z, t) and output (Re U, Im U). The loss was initial-condition MSE plus PDE-residual MSE, with 50k Latin hypercube collocation points, the L-BFGS optimizer, and relative L2 error against SSFM.
- Their "generalization" test was **interpolation only**. They trained on peak powers p ∈ {0.5, 0.7, …, 1.5} and tested on the values between them. **Nobody has tested extrapolation outside the training range.** That gap is this project.
- Known typos in the arXiv text: L_D is printed as T0²/|β2|² (it should be T0²/|β2|), and β2 units are printed as ps/km (they should be ps²/km). Re-derive all equations yourself.

**Starter code:** https://github.com/xieyulai/fiber-nlse-pinn (MIT license, 1 commit, 2 stars, unreviewed).
- It contains an SSFM reference solver, a PINN with autodiff, and six presets (fig3 to fig8) with single-seed relative errors from 0.48% to 7.72%.
- Its preset names may not match the arXiv figure numbering.
- It has **no** data-driven baseline, **no** parametric PINN, and **no** extrapolation splits.
- Treat it as an untrusted scaffold. Read the code before running it, then validate it independently.

**Research question:**
> When ultrashort pulses enter a propagation regime not seen during training (higher soliton order N, or longer propagation distance z), does physics-informed learning generalize better than purely data-driven learning? Can we *predict* where and why each model fails, using physically meaningful quantities?

**Planned contributions, in priority order:**
1. A fair, multi-seed benchmark of four models on N-extrapolation and z-extrapolation. The models are: a parametric PINN (strict and physics-extended variants), an improved PINN+ with Fourier features and causal training, a data-driven MLP, and a conditional FNO. A tuned SSFM serves as the accuracy and wall-clock reference.
2. **Novel component A:** test whether extrapolation error is predicted by spectral shift (Wasserstein-Fourier distance or SPM-predicted bandwidth growth), following Fesser et al. 2023 (arXiv:2306.09478). We apply this to the NLSE, where the spectral shift has a closed-form physical driver.
3. **Novel component B:** test whether conservation-law drift (energy, Hamiltonian, PDE residual), which needs no ground truth, works as a label-free error estimator.
4. Optional extras: test-time physics fine-tuning versus the cost of an SSFM run, and inverse retrieval of β2 and γ from intensity-only data.
</project_context>

<operating_principles>
Follow these in every phase. They override any instinct to move fast.

1. **Plan before acting.** At the start of each phase, write a short plan in your reply: goal, files you will create or modify, how you will verify, and the estimated compute time. Then execute it.
2. **Smoke test first, scale later.** Every training script must support a `--smoke` flag that runs in under 60 seconds: tiny iteration counts, tiny grids, and one N value. Run the smoke test and confirm it works before any real run.
3. **Respect the compute budget.**
   - Before launching any job, estimate its wall-clock time from the Phase 0 benchmarks.
   - The human sets a **total compute budget** once, in the setup questions. Proceed without asking as long as the cumulative estimate stays within that budget.
   - If a phase would push the total over budget, shrink it automatically, in this order: fewer seeds (down to a minimum of 3), then Regime B before Regime A, then ablations. Log the change in `DECISIONS.md`.
   - Stop and ask only if the core Regime A study itself cannot fit.
   - Run long jobs in the background with logs written to disk, and check on them periodically. Never block on a job for hours.
4. **Verify, don't assume.**
   - After writing code, run it.
   - After producing a number, sanity-check it against physics (energy conservation, analytic limits) and against intuition. Does error grow with N? Does the PINN do better in-distribution than out?
   - If a result looks too good, assume data leakage or a bug until you have ruled both out.
5. **Never fabricate.**
   - Every number in any results table, markdown summary or figure must come from a file produced by code in this repo.
   - Never type a metric by hand. Tables are generated by scripts that read `runs/` and `results/`.
   - If something failed or is missing, say so plainly.
6. **Never weaken a test to make it pass.** If a physics test fails, the code is wrong, or the tolerance is genuinely unjustified. In the second case, explain why in a comment and tell the human.
7. **Reproducibility.**
   - Seed everything: Python, NumPy, torch, CUDA.
   - Save the exact config, git commit hash, package versions and hardware info with every run.
   - Pin dependency versions.
8. **Keep it simple.** Use plain PyTorch, NumPy, SciPy, Matplotlib, PyYAML and `neuraloperator` for the FNO. Do not add frameworks (Hydra, Lightning, wandb) unless the human asks. Prefer small readable modules over clever abstractions.
9. **Cross-platform.** Write Python scripts, not bash-only scripts, so everything works on Windows, macOS and Linux. Use `pathlib`.
10. **Be autonomous by default.** The human wants you to move from phase to phase without waiting for approval.
    - Make engineering and scientific decisions yourself, consistent with this brief, and record each one in `DECISIONS.md` with a one-line reason.
    - Interrupt the human only for the hard stops in `<autonomy_and_stops>`.
11. **Keep a log.** After every phase, append to `PROGRESS.md`: what was done, key results (with file paths), problems found, and what comes next. This is how the two students and future sessions pick up where you left off.
</operating_principles>

<autonomy_and_stops>
**Default behaviour:** work continuously through all phases.

At each checkpoint:
1. Write the checkpoint report, in the format given in `<communication_format>`, to `PROGRESS.md`.
2. Print a short version of it in your reply.
3. Commit and push.
4. **Immediately continue to the next phase. Do not wait for a reply.**

**One-time setup questions.** Ask these all together in a single message at the start of Phase 0, then never again:
1. GitHub repo name (default `ultrafast-pinn-extrapolation`) and visibility (default private).
2. Teammate's GitHub username to add as a collaborator. They may say "skip".
3. venv or conda (default venv).
4. Total compute budget in GPU or CPU hours (default 12 hours), and whether Google Colab may be used if the local machine is too weak.
5. Whether a soft deadline exists, so you can prioritize.

If the human doesn't answer something, use the default and log it in `DECISIONS.md`.

**Hard stops.** Only these justify pausing and waiting for the human:
- The human must do something themselves: `gh auth login`, installing system software that needs admin rights, or entering credentials. You never handle tokens or passwords.
- A destructive or irreversible action: deleting files outside `runs/` and `data/`, force-pushing, rewriting `main` history, or deleting the GitHub repo or branches.
- The core Regime A study cannot fit in the compute budget even after automatic reductions.
- The SSFM ground truth cannot be validated after reasonable debugging, because every downstream result depends on it.
- A finding that would change the research question itself, for example a fatal flaw in the premise discovered in the literature.

Everything else is a soft issue. Decide, log it, mention it in the next checkpoint report, and keep going.

**Within-session continuity:**
- When a background job is running, do useful non-dependent work while you wait: write tests, plotting code, docs, or literature notes.
- Check job status periodically.
- When a sweep finishes, aggregate the results, commit, and move on.
</autonomy_and_stops>

<literature_and_web_access>
You have web access (WebSearch and WebFetch). Use it actively to make the science better, not as decoration.

**Required reading, in Phase 0.5 (see `<phases>`):**
- Fetch and read the primary sources: the Jiang et al. arXiv:2109.00526 PDF; the starter repo README on GitHub; Fesser et al. arXiv:2306.09478; Wang, Sankaran & Perdikaris arXiv:2203.07404 (causal training); Krishnapriyan et al. (NeurIPS 2021, arXiv:2109.01050); the `neuraloperator` docs.
- Extract from them the exact equations, hyperparameters, losses and reported numbers you need.
- Confirm or correct everything stated in this brief. If the brief is wrong, trust the primary source and log the correction in `DECISIONS.md`.

**Looking for newer work:**
- Search for papers from 2023 onward that cite Jiang et al. 2022 or address PINN or neural-operator extrapolation for NLSE/GNLSE. Useful queries:
  - "physics-informed neural network nonlinear Schrödinger fiber 2025"
  - "neural operator supercontinuum generalization"
  - "PINN extrapolation spectral shift"
  - "causal PINN soliton"
  - "Fourier neural operator fiber propagation"
- Prefer arXiv, publisher pages, Semantic Scholar and Google Scholar. Repeat the search briefly at the start of Phases 6 and 9.

**Adopting a new technique** (a better PINN trick, sampling scheme, or operator architecture). You may add it **only as an additional, clearly labelled model or ablation**. Never use it to change the pre-registered splits or metrics. Record the paper, why it was adopted, and where it lives in the code.

**Novelty check:**
- Before calling our analyses A or B "novel" anywhere, search specifically for prior work doing the same thing on the NLSE.
- If you find it, cite it, reframe our work as replication and extension, and log this in `DECISIONS.md`.

**Logging what you read:**
- Keep `report/literature_notes.md` with, for each source: full citation and link, the 3 to 5 facts relevant to this project, and how we use it (baseline, method, comparison, or background).
- Also keep `report/references.bib` with BibTeX entries.
- This feeds the report and the students' literature slides.

**Rules for using sources:**
- Never invent a citation, DOI or number. If you couldn't access a source, write "not accessed" in the notes.
- Paraphrase; don't copy long passages into docs.
- Respect licenses when reusing code. MIT and BSD are fine with attribution. Avoid GPL code inside `src/`.
- Downloading a paper PDF into `report/papers/` is fine if you need it, but gitignore that folder.
</literature_and_web_access>

<physics_reference>
Use these conventions throughout and state them in the README. **Verify every formula numerically with tests. Do not trust this brief blindly. Sign conventions differ between textbooks.**

**Normalized NLSE (core study, Agrawal convention):**
- U = A/√P0, ξ = z/L_D, τ = T/T0, s = sgn(β2)
- L_D = T0²/|β2|, L_NL = 1/(γP0), N² = L_D/L_NL = γP0T0²/|β2|
- Equation: `i ∂U/∂ξ = (s/2) ∂²U/∂τ² − N² |U|² U`
- s = +1 is normal dispersion and s = −1 is anomalous dispersion. For s = −1 and N = 1, U = sech(τ) is the fundamental soliton, with soliton period ξ0 = π/2.
- Residuals for the PINN: write U = u + iv and split the equation into real and imaginary residuals f, g. Derive them by hand and unit-test them on an analytic solution.

**Analytic checks:**
- **GVD only (N = 0), Gaussian U(0,τ) = exp(−τ²/2):**
  `U(ξ,τ) = (1 − i s ξ)^(-1/2) · exp(−τ² / (2(1 − i s ξ)))`.
  Check this formula against the sign convention of your solver. The intensity FWHM must broaden by √(1+ξ²) regardless of sign.
- **Fundamental soliton (s = −1, N = 1, sech input):** |U(ξ,τ)| = sech(τ) for all ξ.
- **Second-order soliton (N = 2):** the intensity profile recovers at ξ = π/2.
- **SPM only (no dispersion):** |U| is constant and the nonlinear phase is N²|U0|²ξ.

**Conserved quantities (lossless NLSE):**
- Energy: E = ∫|U|² dτ
- Momentum: P = (i/2)∫(U U*_τ − U* U_τ) dτ
- Hamiltonian: H = ∫[(s/2)|U_τ|² + (N²/2)|U|⁴] dτ. Verify numerically that SSFM conserves your chosen H to about 1e-6 relative error. If it does not, re-derive the sign.

**Grid adequacy (very important for large N):**
- SPM broadens the spectrum strongly at N = 3 to 4 and ξ up to 4.
- For every parameter set, check that the energy fraction within 5% of the edges of the time window and of the frequency window is below 1e-6.
- Run a convergence test: halving dτ and dξ must change the output field by less than 1e-6 relative L2.
- Pick one grid that is adequate for the **hardest** test case. All models must use the same grid.

**Representative physical parameters (for the report only):** SMF-28 at 1550 nm has β2 ≈ −21.7 ps²/km and γ ≈ 1.3 W⁻¹km⁻¹.
</physics_reference>

<experimental_protocol>
**Pre-register this in `configs/splits.yaml` and commit it BEFORE any model training. Do not change splits after seeing results.** If a change is truly needed, record it in `DECISIONS.md` with the reason.

**Regime A (primary): normal dispersion (s = +1), Gaussian input, GVD + SPM**
- Train N: {1.0, 1.2, 1.4, 1.6, 1.8, 2.0}
- Interpolation test N: {1.1, 1.3, 1.5, 1.7, 1.9}
- Near-extrapolation N: {2.25, 2.5}
- Far-extrapolation N: {3.0, 4.0}

**Regime B (secondary): anomalous dispersion (s = −1), sech and Gaussian inputs, GVD + SPM**
- Train N: {0.8, 1.0, 1.2, 1.4, 1.6}
- Interpolation N: {0.9, 1.1, 1.3, 1.5}
- Extrapolation N: {2.0, 3.0}. This crosses into higher-order soliton breathing, which is a qualitatively new regime.

**z-extrapolation (both regimes):** train on ξ ∈ [0, 2] and evaluate on ξ ∈ (2, 4].

**Information available to each model (put this table in the README):**

| Model | Sees labels (SSFM fields)? | Sees PDE? | Collocation N-range | Notes |
|---|---|---|---|---|
| PINN-strict | IC only | yes | train N only | pure extrapolation |
| PINN-extended | IC only | yes | train + test N (no labels) | uses physics at test conditions, which is its legitimate advantage |
| PINN+ | IC only | yes | same as strict (and an extended variant if time allows) | Fourier features in τ, causal weighting in ξ, adaptive loss weights |
| DD-MLP | full fields at train N | no | n/a | same architecture as the PINN |
| cFNO | full fields at train N | no | n/a | (U0, N) → U at output ξ slices; autoregressive variant for z-extrapolation |
| SSFM | n/a | exact | n/a | reference; wall-clock time to matched accuracy |
| PINN-finetune (optional) | IC at test N | yes | test N | starts from parametric weights; report steps and time to reach the in-distribution error |

**Fairness rules:**
- The PINN and DD-MLP use the same width, depth, activation and optimizer schedule.
- Every model gets the same compute budget, either equal wall-clock or equal optimizer steps. Choose one, state it, and log both.
- Hyperparameters are tuned only on the interpolation set, never on extrapolation sets.

**Seeds:** 3 per (model, regime, split), or 5 if the budget allows. Report the mean ± std and the median.

**Metrics** (implement in `src/nlse_ml/eval/metrics.py` with unit tests):
- Relative L2 of the complex field over the full (ξ, τ) grid and at the final ξ
- Intensity-only relative L2
- Intensity-weighted phase error: ∫|U|²|Δφ|dτ / ∫|U|²
- Spectral error: relative L2 of the power spectrum in dB, floored at −60 dB
- Observables: peak power, FWHM, RMS spectral width, −20 dB bandwidth, and (Regime B) compression factor and the ξ position of maximum compression
- Physics consistency: |E(ξ) − E(0)|/E(0), Hamiltonian drift, momentum drift, and PDE residual norm on a dense held-out grid
- Cost: training wall-clock, peak GPU memory, inference time per field, and SSFM time at matched accuracy
- Spectral shift: 1D Wasserstein distance between normalized power spectra (computed via CDFs) of U(0) and U(ξ), and between the nearest training N and the test N. Also the SPM bandwidth predictor φ_max = N²·ξ, scaled to account for dispersion.
</experimental_protocol>

<repository_layout>
Create this structure. Adjust it only if there is a clear reason, and log the reason in `DECISIONS.md`.

```
ultrafast-pinn-extrapolation/
├── CLAUDE.md                 # persistent rules for Claude Code (summary of this brief)
├── PROJECT_BRIEF.md          # this file
├── README.md                 # overview, setup, how to reproduce every figure
├── PROGRESS.md               # phase-by-phase log
├── DECISIONS.md              # engineering and scientific decisions with reasons
├── requirements.txt          # pinned versions (plus requirements-lock.txt from pip freeze)
├── .gitignore
├── configs/
│   ├── splits.yaml           # pre-registered splits (committed before training)
│   ├── grid.yaml             # validated numerical grid
│   ├── regime_a.yaml, regime_b.yaml
│   └── models/ pinn.yaml, pinn_plus.yaml, dd_mlp.yaml, cfno.yaml
├── src/nlse_ml/
│   ├── physics/  nlse.py, ssfm.py, analytic.py, conservation.py
│   ├── data/     generate.py, splits.py, datasets.py
│   ├── models/   mlp.py, pinn.py, pinn_plus.py, fno.py
│   ├── train/    common.py, train_pinn.py, train_dd.py, train_fno.py
│   ├── eval/     metrics.py, spectral_shift.py, evaluate.py, aggregate.py
│   └── viz/      plots.py, style.py
├── scripts/
│   ├── 00_env_check.py
│   ├── 01_benchmark.py
│   ├── 02_reproduce_repo.py
│   ├── 03_validate_ssfm.py
│   ├── 04_generate_data.py
│   ├── run_experiment.py     # one entry point: --model --regime --split --seed [--smoke]
│   ├── run_sweep.py          # loops run_experiment with resume and skip-if-done
│   └── make_figures.py, make_tables.py
├── tests/                    # pytest: physics, metrics, residuals, splits, no-leakage
├── third_party/fiber-nlse-pinn/   # git submodule pinned to a specific commit
├── data/        (gitignored)  # generated SSFM datasets (.npz)
├── runs/        (gitignored)  # runs/<exp>/<model>/<regime>/<split>/seed<k>/{config.yaml, meta.json, log.txt, metrics.json, preds.npz, ckpt.pt}
├── results/     (committed)   # aggregated CSV/JSON, small files only
├── figures/     (committed)   # PNG (300 dpi) and PDF
└── report/                    # results_summary.md, tables, captions, literature_notes.md, references.bib (papers/ gitignored)
```

Every run directory must contain `meta.json` with: git commit hash, dirty flag, timestamp, hostname, OS, Python, torch, CUDA and GPU name, seed, and the full resolved config.
</repository_layout>

<git_and_github_rules>
**Setup (Phase 1):**
1. Check that `git` is installed and configured: `git config user.name` and `user.email`. If not, ask the human for their values.
2. Check that the GitHub CLI is installed and authenticated: `gh --version` and `gh auth status`.
   - If `gh` is missing, tell the human how to install it for their OS (a hard stop for pushing only; keep working locally meanwhile).
   - If it is not authenticated, ask the human to run `gh auth login` themselves in the terminal. **Never handle tokens or passwords yourself.**
3. Use the repo name, visibility and teammate username from the one-time setup answers (or the defaults). Do not ask again.
4. Create the repo with `gh repo create <name> --private --source=. --remote=origin` and push `main`.
   - If a teammate username was given, invite them with `gh api -X PUT repos/<owner>/<name>/collaborators/<user>`.
   - If `gh` is not authenticated, that is a hard stop: ask the human to run `gh auth login`. While waiting, keep working locally and commit as usual; push once auth works.
5. Add the starter repo as a submodule pinned to a commit: `git submodule add https://github.com/xieyulai/fiber-nlse-pinn third_party/fiber-nlse-pinn`, then check out its current commit hash and record that hash in `DECISIONS.md`. Keep its MIT LICENSE.

**Branching:**
- `main` must always work: tests pass and the README is accurate.
- Use one branch per phase (`phase-2-reproduce`, `phase-3-ssfm-validation`, and so on).
- When the phase checkpoint passes, merge to `main` with `--no-ff` and push.
- Tag milestones: `m1-ground-truth`, `m2-in-distribution`, `m3-main-results`, `m4-analysis`, `final`.

**Commits:**
- Commit after each **logical unit** of work that leaves the code runnable and tests passing. Don't commit after every line, and don't make one giant commit per phase.
- Use Conventional Commits with an imperative subject of 72 characters or fewer, plus a body explaining *why* when it isn't obvious. Examples:
  - `chore: scaffold repo structure, pinned requirements, gitignore`
  - `feat(physics): add normalized NLSE SSFM solver with configurable grid`
  - `test(physics): verify GVD-only Gaussian against analytic solution`
  - `fix(ssfm): correct sign of dispersion operator for normal regime`
  - `feat(data): pre-register N and z splits for regimes A and B`
  - `exp: run regime A in-distribution sweep, 4 models x 3 seeds`
  - `docs(progress): log phase 3 results and open issues`
  - `results: add aggregated extrapolation table and figures for regime A`
- **Never commit** datasets, checkpoints, `runs/`, `.npz` or `.pt` files, notebooks with large outputs, `.env` files, secrets, or any file over 5 MB. Run `git status` and review the staged files before every commit.
- **Never** force-push, rewrite history on `main`, or delete branches without asking.
- Push to `origin` at the end of each phase and after any important results commit.
</git_and_github_rules>

<phases>

### PHASE 0: Environment audit and compute feasibility
**Goal:** find out whether this machine can run the project, and set the compute tier.

1. Create `scripts/00_env_check.py`. It prints and saves to `report/env_report.md`:
   - OS, CPU model and core count, total and available RAM, free disk space
   - Python version
   - GPU availability (CUDA, Apple MPS, or none), GPU name, VRAM, CUDA and cuDNN versions, driver
   - versions of installed key packages
2. Set up a virtual environment (`python -m venv .venv`, or conda if the setup answers said so). Install pinned dependencies: torch (the correct CUDA build for the detected GPU), numpy, scipy, matplotlib, pyyaml, pytest, neuraloperator, pandas. Then write `requirements-lock.txt` with `pip freeze`.
3. Create `scripts/01_benchmark.py` with these micro-benchmarks:
   - SSFM: one run on a 512-point τ grid with 201 ξ saves, timed in float64.
   - PINN: 4×100 tanh MLP with 50k collocation points. Time 200 Adam steps and 20 L-BFGS steps in float32 (and float64 if a GPU is present). Record steps per second and peak memory.
   - DD-MLP: the same network, 200 steps on dense field data.
   - FNO: a small conditional 1D FNO, 50 steps on a batch of 32.
4. **Extrapolate costs** to the full plan: number of runs × steps per run. Write a table to `report/env_report.md` with estimated hours per phase.
5. **Decide the compute tier and write it to `DECISIONS.md`:**
   - **Tier 1 (GPU with ≥ 8 GB VRAM):** full plan, 3 to 5 seeds.
   - **Tier 2 (smaller GPU or Apple MPS):** 3 seeds, 20 to 30k collocation points, 256-point τ grid if the grid-adequacy tests still pass.
   - **Tier 3 (CPU only):** cut the scope and propose Google Colab for the PINN and FNO sweeps. If chosen, generate `notebooks/colab_runner.ipynb` that clones the repo and runs `run_sweep.py`.
6. Create `.gitignore`, `requirements.txt`, the folder skeleton, a `README.md` stub, `PROGRESS.md`, `DECISIONS.md`, and `CLAUDE.md`.
   - `CLAUDE.md` must contain a condensed version of the operating principles, git rules, compute rules and physics conventions from this brief, plus a pointer to this brief.

**📋 CHECKPOINT 0 (report and continue):** Log the environment report, cost table, chosen tier, and any blockers. The setup questions were already asked at the very start (see `<autonomy_and_stops>`). Continue to Phase 0.5 immediately, unless the compute budget makes the core study impossible (a hard stop).

---

### PHASE 0.5: Literature grounding (web)
Follow `<literature_and_web_access>`.
1. Fetch and read the required primary sources.
2. Build `report/literature_notes.md` and `report/references.bib`.
3. Produce `report/brief_corrections.md`, listing any statement in this brief that the primary sources contradict and what you will do instead.
4. Search for papers from 2023 onward. Shortlist at most 3 techniques worth adding as labelled extra models or ablations, and log them in `DECISIONS.md`.
5. Commit: `docs(lit): literature notes, references and brief corrections`.
Continue straight into Phase 1.

---

### PHASE 1: Git and GitHub setup
Follow `<git_and_github_rules>`.
- First commit: `chore: scaffold repo structure, env check, benchmark scripts`.
- Add the submodule and record the pinned hash.
- Push `main`. Log the remote URL and the invite status in `PROGRESS.md`, then continue.

---

### PHASE 2: Read, audit and reproduce the starter repo
**Goal:** understand the baseline and check its claims across seeds.

1. **Read every file** in `third_party/fiber-nlse-pinn` before running anything. Summarize in `report/repo_audit.md`:
   - the equation it implements, including signs and normalization
   - the SSFM scheme: symmetric or not, step size, and how the grid is set
   - the PINN loss, sampling, optimizer and stopping rule
   - the six presets and how they map to the arXiv paper's figures (note any mismatch)
   - anything suspicious, such as hard-coded values, possible bugs, or sign inconsistencies with the paper
   Do not run any script that downloads things or touches files outside the repo without telling the human.
2. Run the repo's presets fig3 (GVD+SPM, normal), fig7 (N=1 soliton) and fig8 (N=2 soliton). Run each in smoke mode first, then at full scale with seeds {0, 1, 2}, if Phase 0 says this fits the budget. Use a wrapper (`scripts/02_reproduce_repo.py`) and do not modify the submodule's code.
3. Compare against the README's reported errors (2.61%, 2.88%, 0.48%). Report mean ± std. Explain any differences honestly.

**Commit:** `exp: reproduce starter repo presets fig3/fig7/fig8 across 3 seeds`, plus the audit doc.
**📋 CHECKPOINT 2 (report and continue):** Log the audit findings and the reproduction table, flagging any bug that affects trust in the repo. Proceed with the default plan: re-implement cleanly in `src/`, and use the repo only as a reference and cross-check.

---

### PHASE 3: Independent ground truth (SSFM) and validation
**Goal:** a solver we trust completely, because every later result depends on it.

1. Implement `src/nlse_ml/physics/ssfm.py`: a symmetric split-step Fourier method in float64 for the normalized NLSE, with configurable s, N, grid, ξ_max and number of saved slices. Optionally add RK4IP as a second integrator for cross-checking.
2. Implement `analytic.py` (GVD-only Gaussian, sech soliton) and `conservation.py` (E, P, H).
3. Write tests in `tests/test_physics.py`:
   - GVD-only Gaussian versus analytic: relative L2 < 1e-8
   - N = 1 soliton shape invariance over 4 soliton periods: < 1e-6
   - N = 2 soliton recurrence at ξ = π/2
   - energy, momentum and Hamiltonian conservation: relative drift < 1e-6 (relaxed only with justification)
   - SPM-only phase: matches N²|U0|²ξ
   - step-size convergence: error decreases at second order
   - optional: cross-check one case against `gnlse-python` (pip install gnlse) or against the starter repo's SSFM, after converting units
4. **Grid adequacy study** (`scripts/03_validate_ssfm.py`): for the hardest cases (N = 4 Regime A, N = 3 Regime B, ξ = 4), check edge energy in time and frequency and run the convergence test. Choose the grid, write it to `configs/grid.yaml`, and produce `figures/ssfm_validation.png` (convergence and conservation plots).
5. Produce gallery figures for the talk: (ξ, τ) intensity and spectral evolution heatmaps (dB) for GVD only, SPM only, Regime A at N = 2 and N = 4, and Regime B at N = 1 and N = 2.

**Commits:** solver, analytic module, tests (one commit per logical piece), grid config, figures.
**Branch:** merge `phase-3-ssfm-validation` to `main`, tag `m1-ground-truth`, and push.
**📋 CHECKPOINT 3 (report and continue):** Log the test results, the chosen grid with justification, and the gallery figures. If validation cannot pass after reasonable debugging, that is a hard stop.

---

### PHASE 4: Datasets and pre-registered splits
1. Write `configs/splits.yaml` exactly as specified in `<experimental_protocol>`. Commit it **before** any training: `feat(data): pre-register N and z splits for regimes A and B`.
2. Write `src/nlse_ml/data/generate.py` and `scripts/04_generate_data.py`. They generate SSFM fields for every N in every split and regime on the validated grid, and save them to `data/<regime>/N_<value>.npz` with metadata (grid, s, N, input shape, solver version, git hash). Add a `--check` flag that recomputes one file and verifies it matches.
3. Write `tests/test_splits.py`:
   - no N value appears in more than one split
   - training loaders physically cannot load test files (assert it)
   - the z-split mask excludes ξ > 2 from all training data and from PINN collocation in z-extrapolation experiments
4. Write a data summary (sizes, generation time) to `PROGRESS.md`.

---

### PHASE 5: Metrics and evaluation library
1. Implement every metric in `<experimental_protocol>` in `metrics.py` and `spectral_shift.py`.
2. Unit-test them on known cases: identical fields give zero error; a known phase offset gives the expected phase error; Wasserstein distance between shifted Gaussians equals the shift; energy drift of the analytic solution is about 0.
3. `evaluate.py` takes a run directory, loads predictions plus ground truth, computes all metrics per split, per N and per ξ slice, and writes `metrics.json`.
4. `aggregate.py` scans `runs/` and builds tidy CSVs in `results/` with columns: model, regime, split, N, seed, metric, value. `make_tables.py` turns these into markdown and LaTeX tables with mean ± std.

---

### PHASE 6: Models and training (in-distribution first)
Build each model, smoke-test it, then train it on Regime A in-distribution with 1 seed before moving on.

1. **Shared MLP** (`mlp.py`): configurable depth, width and activation. Inputs (ξ, τ, N) are normalized to [−1, 1] using **training-range statistics only**. Inputs outside the training range simply map outside [−1, 1]; document this.
2. **Parametric PINN** (`pinn.py`):
   - Loss = w_ic·MSE_IC (over all training N) + w_r·MSE_residual
   - Collocation points are sampled by Latin hypercube over (ξ, τ, N), resampled every K steps (configurable)
   - Schedule: Adam warm-up, then L-BFGS (with a strong Wolfe line search)
   - Two variants selected by config: `strict` (collocation N in the training range) and `extended` (collocation N spans train and test ranges)
   - Use float32 by default and test float64 for L-BFGS in one ablation
   - Residuals use `torch.autograd.grad` with `create_graph=True`, unit-tested against the analytic GVD solution (the residual must be about 0 on the exact solution)
3. **PINN+** (`pinn_plus.py`):
   - Random Fourier features in τ
   - Causal weighting in ξ (Wang, Sankaran & Perdikaris 2024). Note in the README that the causal training method is patented for commercial use, and that academic use is fine.
   - Adaptive loss weighting (gradient-norm balancing)
   - Each feature can be toggled from config, for the ablations
4. **DD-MLP** (`train_dd.py`): same network as the PINN, trained with MSE on SSFM fields at training N, sampled from the grid. Same optimizer schedule and budget.
5. **Conditional FNO** (`fno.py`, using `neuraloperator`):
   - Version 1 (direct): input channels [Re U0, Im U0, N broadcast, τ grid], output [Re, Im] at K fixed ξ slices.
   - Version 2 (autoregressive): step operator U(ξ) → U(ξ+Δξ), conditioned on N and rolled out. Needed for the z-extrapolation test.
   - Train on all ξ-slices of the training N values.
6. **SSFM baseline timing:** measure SSFM wall-clock time to reach the same error as each model, by coarsening its step until its error matches.
7. `run_experiment.py` is the single entry point: `--model {pinn_strict,pinn_extended,pinn_plus,dd_mlp,cfno,cfno_ar} --regime {A,B} --exp {indist,n_extrap,z_extrap} --seed k [--smoke]`. It writes the full run directory and then calls `evaluate.py`.
8. Log the loss curves and save `loss.png` in each run directory.

**Sanity checks before Checkpoint 6:**
- In-distribution, PINN error should be roughly in the 1e-3 to 1e-2 range of the paper's order of magnitude. If you see something like 50%, debug the training before going further.
- DD-MLP should fit its training N very well.
- Error should tend to grow with N.

**Commits:** one per model, plus one for the entry point. Merge, tag `m2-in-distribution`, push.
**📋 CHECKPOINT 6 (report and continue):** Log a single-seed in-distribution table for all models, loss curves, the measured cost per run, and the compute estimate for Phase 7.
- If the estimate fits the remaining budget, start Phase 7 immediately.
- If not, apply the automatic reductions from operating principle 3, log them, and start.

---

### PHASE 7: Main experiments (sweeps)
Start automatically after Checkpoint 6, within the compute budget.

1. `run_sweep.py` loops over models × regimes × experiments × seeds:
   - It skips runs that are already complete (checks for `metrics.json`), so it can resume after a crash.
   - It writes a sweep log, and runs in the background with progress written to `runs/<sweep>/status.txt`.
2. Order the runs so the most important results come first:
   - (a) Regime A N-extrapolation, all models, 3 seeds
   - (b) Regime A z-extrapolation
   - (c) Regime B N-extrapolation
   - (d) Regime B z-extrapolation
3. After each block finishes: aggregate, generate the figure, and commit the results (CSV plus figures only), e.g. `exp: regime A N-extrapolation sweep, 6 models x 3 seeds`.
4. Watch for failed or diverged runs. Do not silently drop them. Report them as failures (they are results) and investigate one of them.

**Core figures** (`make_figures.py`, consistent style, colorblind-safe palette, PNG + PDF):
- `fig_extrap_N_regimeA`: relative L2 vs N for all models, training range shaded, mean ± std bands over seeds, log y-axis
- `fig_extrap_z`: error vs ξ beyond the training distance
- `fig_heatmaps_compare`: true / predicted / |error| (ξ, τ) intensity for each model at one interpolation N and one far-extrapolation N
- `fig_spectra_far`: predicted vs true output spectra (dB) at far extrapolation
- `fig_energy_drift`: |ΔE|/E vs ξ per model
- `fig_pareto`: accuracy vs wall-clock time, including SSFM

Merge, tag `m3-main-results`, push.
**📋 CHECKPOINT 7 (report and continue):** Log the main results table and figures, with a plain-language interpretation. Make clear which conclusions are robust across seeds and which are not.

---

### PHASE 8: Ablations
Run as much as the remaining budget allows, in the order listed, and skip what doesn't fit. Log what was skipped.
1. **Training-range width:** train on N ∈ [1.4, 1.6], [1.2, 1.8] and [1.0, 2.0], and plot error vs distance-to-range. This tests whether apparent generalization is an artifact of a narrow range.
2. PINN+ components on and off: Fourier features, causal weighting, adaptive weights.
3. Number of collocation points (10k, 25k, 50k).
4. IC loss weight.
5. Data efficiency: number of training N values (2, 4, 6) for DD-MLP, cFNO and PINN.
6. Optional: noise robustness (add Gaussian noise to the training fields or the IC).

---

### PHASE 9: Novel analyses
1. **Spectral-shift predictor (component A):** for every (model, test N, ξ), compute the Wasserstein-Fourier distance from the training distribution and the SPM predictor. Scatter-plot against true error with all runs pooled. Compute Spearman ρ per model with 95% bootstrap confidence intervals. Figure: `fig_error_vs_spectral_shift`. Write a clear verdict: does it collapse the data onto one trend for PINNs? For data-driven models?
2. **Conservation drift as a label-free error estimator (component B):** for each prediction, correlate energy drift, Hamiltonian drift and residual norm (all computable without ground truth) with true error. Report Spearman ρ with confidence intervals per model. Figure: `fig_drift_vs_error`. Discuss the failure case where a model conserves energy well but gets the phase wrong.
3. **Optional: test-time fine-tuning vs SSFM.** Starting from the parametric PINN weights, fine-tune at each far-extrapolation N. Report the steps and wall-clock time needed to reach the in-distribution error, next to the SSFM time. State the conclusion honestly, even if SSFM wins.
4. **Optional: inverse retrieval.** Recover β2 and γ (and their combination N) from output intensity plus spectrum, with added noise at SNR 10, 20 and 30 dB. Compare the PINN against a differentiable-SSFM gradient fit (torch). Produce an identifiability plot.

Merge, tag `m4-analysis`, push.

---

### PHASE 10: Consolidation for report and slides
1. `report/results_summary.md`:
   - The research question and hypotheses
   - Setup summary
   - **Auto-generated** tables (from `make_tables.py`)
   - Every figure with a 2 to 3 sentence caption saying what it shows and the takeaway
   - Key findings as numbered bullets, each linked to a figure or table
   - A limitations and threats-to-validity section
2. `report/figure_index.md`: each figure file, what it shows, which slide it suits, and the command that regenerates it.
3. README: a full reproduction guide, with one command per figure and the expected runtime per command.
4. Final check:
   - Fresh clone, create the environment, run `pytest`, then run `make_figures.py` from the committed results. It must work end to end.
   - Fix anything that breaks.
5. Tag `final` and push.

**🏁 FINAL CHECKPOINT (this is where you stop):** Give a concise summary: what was found, how confident we are, what failed, and suggested talking points. List open issues the students should mention in Q&A.
</phases>

<failure_handling>
- **PINN won't converge, especially at N ≥ 2 or in Regime B:**
  - Try, in order: lower the learning rate and lengthen the Adam warm-up; use a curriculum in N (start narrow and widen); use a curriculum or sequence-to-sequence marching in ξ; turn on causal weighting; add Fourier features; use float64 for L-BFGS.
  - Record every attempt in `DECISIONS.md`.
  - If it still fails, that is a result: report it with loss curves. Do not tune on extrapolation sets to rescue it.
- **Out-of-memory error:** reduce collocation batch size (use mini-batch resampling), then the grid size (only if grid adequacy still holds), then the network size. Log the change.
- **SSFM test fails:** stop all downstream work and fix the solver first. Never move on with an unvalidated solver.
- **Suspiciously good extrapolation:** check for leakage (test N in training files, ξ > 2 in collocation, normalization statistics computed on test data). Write a leakage test that would catch it.
- **Package or CUDA install problems:** try fixes inside the virtual environment yourself (other wheel index, other version). Ask only before changing system-level settings or installing system software.
- **Ambiguity in this brief:** choose the option most consistent with "fair comparison, honest results, and finishing within the timeline," log it in `DECISIONS.md`, and mention it at the next checkpoint.
</failure_handling>

<communication_format>
At every checkpoint, reply in this structure:

```
## Checkpoint <n>: <phase name>
**Status:** done / partial / blocked
**What I did:** 3-6 bullets
**Key results:** table or bullets, each with the file path it came from
**Sanity checks:** what I verified and how
**Problems / surprises:** honest list (or "none")
**Commits:** list of commit subjects + branch/tag + pushed? (y/n)
**Next:** what Phase <n+1> will do + compute estimate
**Questions for you:** numbered, only if needed
```

Between checkpoints, keep messages short. No long narration of routine steps. Checkpoint reports are status updates, not requests for approval. Keep going after each one.
</communication_format>

<definition_of_done>
The project is complete when all of the following are true:
- [ ] `pytest` passes on a fresh clone (physics, metrics, splits, leakage tests).
- [ ] SSFM is validated against analytic solutions, conservation laws and convergence, with figures.
- [ ] Splits were pre-registered and committed before any training, and git history proves it.
- [ ] All core models were run on Regime A (N- and z-extrapolation) with at least 3 seeds. Regime B was run if the budget allowed.
- [ ] Every table and figure is regenerable from `results/` with one documented command.
- [ ] Novel analyses A and B are done, with correlation statistics and honest verdicts.
- [ ] `report/results_summary.md`, `report/figure_index.md`, README, PROGRESS and DECISIONS are complete and accurate.
- [ ] Everything is committed with clean history and logical commits, milestones are tagged, pushed to GitHub, and the teammate has access.
- [ ] No fabricated numbers, no hidden failed runs, no unexplained changes to the protocol.
</definition_of_done>

<start_instruction>
Begin now:
1. Confirm you have read this whole brief.
2. Restate the research question and the five most important rules in your own words, in 5 bullets max.
3. Ask the one-time setup questions from `<autonomy_and_stops>` in a single message.
4. While waiting for answers, start the Phase 0 environment audit (it doesn't depend on them). Use defaults for anything still unanswered when you need it.
5. Then work through every phase autonomously until the Final Checkpoint, stopping only for hard stops.
</start_instruction>

## Key references (for docstrings, README and report)
- Jiang et al., Laser Photonics Rev. 16, 2100483 (2022); arXiv:2109.00526
- Jiang et al., J. Lightwave Technol. 42(5), 1381 (2024): enhanced or causal PINN for the NLSE
- Raissi, Perdikaris, Karniadakis, J. Comput. Phys. 378, 686 (2019): original PINN, NLSE example
- Salmela et al., Nat. Mach. Intell. 3, 344 (2021): RNN for nonlinear fiber dynamics (code: gitlab.com/salmelal/rnnnonlinear)
- Genty et al., Nat. Photonics 15, 91 (2021): review, ML in ultrafast photonics
- Krishnapriyan et al., NeurIPS 2021: PINN failure modes
- Wang, Sankaran, Perdikaris, CMAME 421, 116813 (2024): causal PINN training
- Fesser et al., arXiv:2306.09478: spectral shifts and PINN extrapolation
- McGreivy & Hakim, Nat. Mach. Intell. 6, 1256 (2024): weak baselines in ML for PDEs
- Grossmann et al., IMA J. Appl. Math. 89, 143 (2024): PINNs vs FEM
- Li et al., Fourier Neural Operator (ICLR 2021); Zeng et al., J. Lightwave Technol. 44(2), 665 (2026): conditional FNO for fiber propagation
- Agrawal, *Nonlinear Fiber Optics*: SSFM, analytic solutions, conventions
- Dudley, Genty, Coen, Rev. Mod. Phys. 78, 1135 (2006); gnlse-python: github.com/WUST-FOG/gnlse-python
