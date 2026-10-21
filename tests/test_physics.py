"""Validation of the reference solver against analytic solutions, conservation laws and convergence."""
import sys
from pathlib import Path

import numpy as np
import pytest

from nlse_ml.physics import analytic, conservation as C
from nlse_ml.physics.nlse import gaussian, sech, tau_grid
from nlse_ml.physics.ssfm import rk4ip, ssfm


def rel_l2(a, b):
    return np.linalg.norm(a - b) / np.linalg.norm(b)


@pytest.mark.parametrize("s", [1, -1])
def test_gvd_gaussian_matches_analytic(s):
    tau, _ = tau_grid(1024, 40.0)
    xi, U = ssfm(gaussian(tau), tau, s, 0.0, 4.0, 41, 0.1)
    assert rel_l2(U, analytic.gvd_gaussian(xi, tau, s)) < 1e-8


@pytest.mark.parametrize("s", [1, -1])
def test_gvd_fwhm_broadening_independent_of_sign(s):
    tau, _ = tau_grid(4096, 40.0)
    xi, U = ssfm(gaussian(tau), tau, s, 0.0, 3.0, 4, 0.1)
    I = np.abs(U) ** 2
    for k in range(len(xi)):
        above = tau[I[k] >= I[k].max() / 2]
        fwhm = above[-1] - above[0]
        fwhm0 = 2 * np.sqrt(np.log(2)) * 2 * np.sqrt(1 + xi[k] ** 2) / 2  # intensity FWHM = 2 sqrt(ln2) T(xi)
        assert abs(fwhm - fwhm0) < 2 * (tau[1] - tau[0])


def test_fundamental_soliton_shape_invariant_4_periods():
    tau, _ = tau_grid(1024, 30.0)
    xi, U = ssfm(sech(tau), tau, -1, 1.0, 4 * np.pi / 2, 21, 1e-3)
    assert rel_l2(np.abs(U), np.abs(analytic.fundamental_soliton(xi, tau))) < 1e-6
    # full complex field including the exp(i xi/2) phase
    assert rel_l2(U, analytic.fundamental_soliton(xi, tau)) < 1e-6


def test_second_order_soliton_recurrence():
    tau, _ = tau_grid(2048, 30.0)
    xi, U = ssfm(sech(tau), tau, -1, 2.0, np.pi / 2, 3, 1e-5)
    I0, Imid, I1 = np.abs(U) ** 2
    assert rel_l2(I1, I0) < 1e-6
    assert Imid.max() > 3.0 * I0.max()  # strong compression at xi = pi/4 (peak ~ 4.7 in |U|^2 units for N=2)


def test_spm_only_phase():
    tau, _ = tau_grid(512, 20.0)
    u0 = gaussian(tau)
    xi, U = ssfm(u0, tau, 0, 1.7, 2.0, 11, 0.05)
    assert rel_l2(U, analytic.spm_only(xi, u0, 1.7)) < 1e-12
    np.testing.assert_allclose(np.abs(U), np.abs(u0)[None, :].repeat(len(xi), 0), atol=1e-12)


@pytest.mark.parametrize("shape,s,N", [("gaussian", 1, 2.0), ("sech", -1, 1.5), ("gaussian", -1, 1.6)])
def test_conservation(shape, s, N):
    tau, _ = tau_grid(1024, 40.0)
    u0 = gaussian(tau) if shape == "gaussian" else sech(tau)
    xi, U = ssfm(u0, tau, s, N, 4.0, 41, 2e-4)
    assert C.relative_drift(C.energy(U, tau)).max() < 1e-10  # unitary scheme: round-off only
    assert C.relative_drift(C.hamiltonian(U, tau, s, N)).max() < 1e-6
    # momentum is 0 for a real symmetric input; check absolute drift relative to sqrt(E*|H|) scale
    P = C.momentum(U, tau)
    assert np.abs(P - P[0]).max() < 1e-10


def test_momentum_conserved_for_chirped_shifted_pulse():
    """Non-trivial momentum: frequency-shifted pulse U0 = sech(tau) exp(-i 0.5 tau)."""
    tau, _ = tau_grid(2048, 40.0)
    u0 = sech(tau) * np.exp(-0.5j * tau)
    xi, U = ssfm(u0, tau, -1, 1.3, 2.0, 21, 2e-4)
    P = C.momentum(U, tau)
    assert abs(P[0]) > 0.5
    assert C.relative_drift(P).max() < 1e-8


def test_ssfm_second_order_convergence():
    tau, _ = tau_grid(512, 30.0)
    u0 = gaussian(tau)
    _, ref = rk4ip(u0, tau, 1, 2.0, 1.0, 2, 1e-4)
    errs = []
    for h in [4e-2, 2e-2, 1e-2]:
        _, U = ssfm(u0, tau, 1, 2.0, 1.0, 2, h)
        errs.append(rel_l2(U[-1], ref[-1]))
    orders = np.log2(np.array(errs[:-1]) / np.array(errs[1:]))
    assert np.all(np.abs(orders - 2.0) < 0.15), (errs, orders)


def test_ssfm_matches_rk4ip():
    tau, _ = tau_grid(1024, 30.0)
    u0 = sech(tau)
    # Step sizes chosen so each integrator's own truncation error is < 1e-6 (SSFM err at h=1e-4
    # is 5e-6 and falls exactly 4x per halving; RK4IP at h=1e-3 is 7e-7). Tolerance unchanged.
    _, U1 = ssfm(u0, tau, -1, 2.5, 2.0, 5, 2.5e-5)
    _, U2 = rk4ip(u0, tau, -1, 2.5, 2.0, 5, 2.5e-4)
    assert rel_l2(U1, U2) < 1e-6


def test_matches_starter_repo_ssfm():
    """Cross-check against third_party/fiber-nlse-pinn (fig3 preset: normal GVD+SPM, N^2 = 1.3)."""
    repo = Path(__file__).resolve().parents[1] / "third_party" / "fiber-nlse-pinn"
    if not (repo / "ssfm.py").exists():
        pytest.skip("submodule not checked out")
    pytest.importorskip("torch")
    sys.path.insert(0, str(repo))
    try:
        from nlse import NLSEParams
        from ssfm import run_ssfm
    finally:
        sys.path.remove(str(repo))
    p = NLSEParams.fig3(n_t=1024, n_z=2000)
    _, _, A = run_ssfm(p, device="cpu", save_every=500)
    tau = np.linspace(-p.t_window_t0, p.t_window_t0, p.n_t, endpoint=False)
    xi, U = ssfm(gaussian(tau), tau, 1, np.sqrt(p.N_sq), p.z_max_ld, A.shape[0], 1e-4)
    # their scheme uses dz = 2e-3 (second order), so agreement is limited by their step size
    assert rel_l2(A, U) < 1e-4
