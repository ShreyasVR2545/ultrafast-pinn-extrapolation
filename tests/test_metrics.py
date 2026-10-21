"""Metric unit tests on cases with known answers."""
import numpy as np
import pytest

from nlse_ml.eval import metrics as M
from nlse_ml.eval.spectral_shift import spm_phase, wasserstein_fourier, wf_from_input
from nlse_ml.physics import analytic
from nlse_ml.physics.nlse import gaussian, sech, tau_grid
from nlse_ml.physics.ssfm import ssfm


@pytest.fixture(scope="module")
def field():
    tau, _ = tau_grid(512, 25.0)
    xi, U = ssfm(gaussian(tau), tau, 1, 1.5, 2.0, 41, 1e-3)
    return xi, tau, U


def test_identical_fields_zero_error(field):
    xi, tau, U = field
    m = M.field_metrics(U, U, xi, tau, 1, 1.5)
    for k in ["rel_l2", "rel_l2_final", "rel_l2_intensity", "phase_err", "spectral_err"]:
        assert m[k] == pytest.approx(0.0, abs=1e-14), k
    assert m["peak_power_final_relerr"] == 0.0


def test_constant_phase_offset(field):
    xi, tau, U = field
    for phi in [0.1, 1.0, 3.0]:
        P = U * np.exp(1j * phi)
        assert M.phase_err(P, U) == pytest.approx(phi, rel=1e-10)
        assert M.rel_l2_intensity(P, U) == pytest.approx(0.0, abs=1e-13)
        assert M.rel_l2(P, U) == pytest.approx(2 * np.sin(phi / 2), rel=1e-10)  # |e^{i phi} - 1|
    # wrapping: offset of 2pi - 0.2 counts as 0.2
    assert M.phase_err(U * np.exp(1j * (2 * np.pi - 0.2)), U) == pytest.approx(0.2, rel=1e-10)


def test_scaled_field_rel_l2(field):
    xi, tau, U = field
    assert M.rel_l2(1.1 * U, U) == pytest.approx(0.1, rel=1e-12)
    assert M.rel_l2_intensity(1.1 * U, U) == pytest.approx(0.21, rel=1e-12)


def test_wasserstein_shifted_gaussians_equals_shift():
    w = np.linspace(-40, 40, 8001)
    for d in [0.5, 2.0, 5.0]:
        S1 = np.exp(-w**2)
        S2 = np.exp(-(w - d) ** 2)
        assert wasserstein_fourier(S1, S2, w) == pytest.approx(d, rel=1e-3)
    assert wasserstein_fourier(np.exp(-w**2), 3 * np.exp(-w**2), w) == pytest.approx(0.0, abs=1e-12)


def test_wf_frequency_shifted_pulse():
    tau, _ = tau_grid(2048, 40.0)
    U0 = sech(tau)
    U = np.stack([U0, U0 * np.exp(-1j * 1.5 * tau)])  # numpy FFT: e^{-i d tau} shifts omega by -d
    assert wf_from_input(U, tau)[1] == pytest.approx(1.5, rel=1e-2)


def test_energy_drift_of_analytic_solution_is_zero():
    tau, _ = tau_grid(1024, 40.0)
    xi = np.linspace(0, 4, 21)
    for s in (1, -1):
        U = analytic.gvd_gaussian(xi, tau, s)
        assert M.energy_drift(U, tau).max() < 1e-12
        assert M.hamiltonian_drift(U, tau, s, 0.0).max() < 1e-10
    U = analytic.fundamental_soliton(xi, tau)
    assert M.hamiltonian_drift(U, tau, -1, 1.0).max() < 1e-12
    assert M.momentum_drift(U, tau).max() < 1e-12


def test_grid_residual_small_on_truth_large_on_wrong_physics():
    # Use the production slice spacing (201 slices over xi in [0, 4], d_xi = 0.02): the 4th-order
    # FD floor scales as d_xi^4 (measured 1e-3 at 0.05, 2.8e-5 at 0.02).
    tau, _ = tau_grid(512, 25.0)
    xi, U = ssfm(gaussian(tau), tau, 1, 1.5, 4.0, 201, 1e-4)
    r_true = M.grid_residual(U, xi, tau, 1, 1.5)
    r_wrongN = M.grid_residual(U, xi, tau, 1, 2.0)
    assert r_true < 1e-4 and r_wrongN > 100 * r_true


def test_observables_on_gaussian():
    tau, _ = tau_grid(4096, 40.0)
    xi = np.array([0.0, 1.0, 2.0])
    U = analytic.gvd_gaussian(xi, tau, 1)
    np.testing.assert_allclose(M.fwhm(U, tau), 2 * np.sqrt(np.log(2)) * np.sqrt(1 + xi**2), rtol=1e-4)
    np.testing.assert_allclose(M.peak_power(U), 1 / np.sqrt(1 + xi**2), rtol=1e-6)
    # Gaussian |U|^2 = exp(-tau^2) has power spectrum ~ exp(-omega^2): RMS width 1/sqrt(2), unchanged by GVD
    np.testing.assert_allclose(M.rms_spectral_width(U, tau), 1 / np.sqrt(2), rtol=1e-6)
    # -20 dB full width of exp(-omega^2): 2 sqrt(ln 100)
    np.testing.assert_allclose(M.bandwidth_db(U, tau), 2 * np.sqrt(np.log(100)), rtol=2e-3)


def test_spm_phase_predictor():
    assert spm_phase(2.0, 1.0, dispersion_scaled=False) == pytest.approx(4.0)
    assert spm_phase(2.0, 1.0) == pytest.approx(4 * np.arcsinh(1.0))


def test_spectral_err_detects_broadening(field):
    xi, tau, U = field
    assert M.spectral_err(U[-1:], U[:1]) > 0.05
