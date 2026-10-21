# Environment report

## Hardware and software

| Item | Value |
|---|---|
| OS | Windows-11-10.0.26200-SP0 |
| CPU | Intel(R) Core(TM) Ultra 9 275HX |
| Logical cores | 24 |
| RAM total (GB) | 33.8 |
| RAM available (GB) | 8.0 |
| Free disk (GB) | 395 |
| Python | 3.12.13 |
| Python executable | C:\Users\Shreyas\ultrafast-pinn-extrapolation\.venv\Scripts\python.exe |
| torch | 2.7.1+cu128 |
| CUDA available | True |
| MPS available | False |
| GPU | NVIDIA GeForce RTX 5070 Laptop GPU |
| VRAM (GB) | 8.5 |
| Compute capability | 12.0 |
| CUDA (torch build) | 12.8 |
| cuDNN | 90701 |
| NVIDIA driver | 592.07 |
| pkg:numpy | 2.2.6 |
| pkg:scipy | 1.15.3 |
| pkg:matplotlib | 3.10.3 |
| pkg:yaml | 6.0.2 |
| pkg:pandas | 2.3.1 |
| pkg:pytest | 8.4.1 |
| pkg:neuralop | 1.0.2 |

## Benchmarks

Device: cuda. Raw numbers in `results/benchmark.json`.

| Benchmark | Result |
|---|---|
| SSFM, 512 pts, 201 saves, dxi=1e-3, xi in [0,4] | 0.15 s |
| PINN 4x100 tanh, 50000 colloc, f32 Adam | 18.1 it/s |
| PINN f32 L-BFGS (strong Wolfe, max_iter=1) | 11.0 it/s |
| PINN f32 peak memory | 1.79 GB |
| PINN f64 Adam / L-BFGS | 1.5 / 1.0 it/s |
| PINN f32 Taylor-mode derivatives, Adam / L-BFGS | 25.3 / 15.4 it/s |
| DD-MLP Adam (50k batch) | 232.0 it/s |
| FNO 1D (64 modes, 64 ch, 4 layers), batch 32 | 72.0 it/s |

### Cost extrapolation (assumes 20000 Adam + 5000 L-BFGS steps per PINN/MLP run, 20k FNO steps, 3 seeds, sequential, Taylor-mode PINN; NOTE: naive, see DECISIONS.md for the actual plan)

Per run: PINN 0.31 h, DD-MLP 0.03 h, FNO 0.08 h.

| Phase | Est. hours |
|---|---|
| Phase 2: repo presets (3 presets x 3 seeds) | 1.4 |
| Phase 3/4: SSFM validation + data (~40 fields) | 0.1 |
| Phase 6: in-distribution, 6 models x 1 seed | 1.1 |
| Phase 7a/b: regime A N- & z-extrap | 6.7 |
| Phase 7c/d: regime B N- & z-extrap | 6.7 |
| Phase 8: ablations (~20 PINN-sized runs) | 6.2 |
| **Total** | **22.2** |
