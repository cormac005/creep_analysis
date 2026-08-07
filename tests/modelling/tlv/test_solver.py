import pytest
import numpy as np
from unittest.mock import MagicMock, patch

from creep_model.modelling.tlv.parameters import T_20_KELVIN, T_30_KELVIN
from creep_model.modelling.tlv.solver import (
    solve_tlv,
    solve_tlv_prepared,
    prepare_test_data,
    _solve_tlv_numba_kernel,
    SolverConvergenceError,
)


@pytest.fixture
def mock_creep_test():
    test = MagicMock()
    test.time_series = np.array([0.0, 1.0, 2.0], dtype=np.float64)
    test.interpolate_temperature.return_value = np.array([20.0, 20.0, 20.0], dtype=np.float64)
    test.applied_stress_MPa = 50.0
    test.strain_series = np.array([0.001, 0.002, 0.003], dtype=np.float64)
    test.temp_time_series = np.array([0.0, 2.0], dtype=np.float64)

    mock_poly = MagicMock()
    mock_poly.deriv.return_value = lambda t: np.zeros_like(t) if isinstance(t, np.ndarray) else 0.0
    test.temperature_polynomial.return_value = mock_poly
    return test


@pytest.fixture
def mock_params():
    params = MagicMock()
    # Mock to_array() returning 10 parameter values:
    # [A20, A30, n20, n30, m20, m30, Ee20, Ee30, Ev20, Ev30]
    params.to_array.return_value = np.array([
        1e-5, 1e-5,        # A
        1.0, 1.0,          # n
        0.0, 0.0,          # m
        10000.0, 10000.0,  # Ee
        5000.0, 5000.0,    # Ev
    ], dtype=np.float64)
    return params


def test_prepare_test_data(mock_creep_test):
    prep = prepare_test_data(mock_creep_test)
    assert prep.applied_stress_MPa == 50.0
    assert prep.eps_measured_0 == 0.001
    assert len(prep.time) == 3
    assert len(prep.dt) == 2


def test_solve_tlv_integration(mock_creep_test, mock_params):
    """Integration test running JIT compiled kernel end-to-end."""
    strain_measured = solve_tlv(mock_creep_test, mock_params, use_measured_initial_condition=True)
    assert len(strain_measured) == 3
    assert np.all(np.isfinite(strain_measured))

    strain_calculated = solve_tlv(mock_creep_test, mock_params, use_measured_initial_condition=False)
    assert len(strain_calculated) == 3
    assert np.all(np.isfinite(strain_calculated))


@pytest.mark.parametrize(
    "err_code, match_str",
    [
        (-2, "stalled"),
        (-3, "diverged"),
        (-4, "did not converge"),
        (-99, "Unknown solver convergence error"),
    ],
)
@patch("creep_model.modelling.tlv.solver._solve_tlv_numba_kernel")
def test_solve_tlv_convergence_error_handling(mock_kernel, err_code, match_str, mock_creep_test, mock_params):
    mock_kernel.return_value = (np.array([]), err_code)
    with pytest.raises(SolverConvergenceError, match=match_str):
        solve_tlv(mock_creep_test, mock_params)


# --- Direct Pure-Python Kernel Line-Coverage Tests via `.py_func` ---

@pytest.fixture
def kernel_inputs():
    """Default valid inputs for _solve_tlv_numba_kernel."""
    p_arr = np.array([1e-5, 1e-5, 1.0, 1.0, 0.0, 0.0, 10000.0, 10000.0, 5000.0, 5000.0], dtype=np.float64)
    time = np.array([0.0, 1.0, 2.0], dtype=np.float64)
    T = np.array([293.15, 293.15, 293.15], dtype=np.float64)
    dt = np.array([1.0, 1.0], dtype=np.float64)
    t_mid = np.array([0.5, 1.5], dtype=np.float64)
    T_mid = np.array([293.15, 293.15], dtype=np.float64)
    T_dot_mid = np.array([0.0, 0.0], dtype=np.float64)
    sigma = 50.0
    eps_0 = 0.001
    return p_arr, time, T, dt, t_mid, T_mid, T_dot_mid, sigma, eps_0


def test_numba_kernel_py_func_branches(kernel_inputs):
    """Executes .py_func to hit initial conditions and stress_diff <= 0 branch."""
    p_arr, time, T, dt, t_mid, T_mid, T_dot_mid, sigma, eps_0 = kernel_inputs

    # Measured IC branch
    strain, err = _solve_tlv_numba_kernel.py_func(
        p_arr, time, T, dt, t_mid, T_mid, T_dot_mid, sigma, eps_0, True, 1e-8, 100, T_20_KELVIN, T_30_KELVIN
    )
    assert err == 0
    assert len(strain) == 3

    # Calculated IC branch
    strain_calc, err_calc = _solve_tlv_numba_kernel.py_func(
        p_arr, time, T, dt, t_mid, T_mid, T_dot_mid, sigma, eps_0, False, 1e-8, 100, T_20_KELVIN, T_30_KELVIN
    )
    assert err_calc == 0
    assert len(strain_calc) == 3

    # Negative stress / stress_diff <= 0 branch (tests norton_hoff_deriv = 0.0)
    strain_neg, err_neg = _solve_tlv_numba_kernel.py_func(
        p_arr, time, T, dt, t_mid, T_mid, T_dot_mid, -100.0, eps_0, True, 1e-8, 100, T_20_KELVIN, T_30_KELVIN
    )
    assert err_neg == 0


def test_numba_kernel_py_func_errors(kernel_inputs):
    """Executes .py_func error exit codes (-2, -3, -4) for 100% line coverage."""
    p_arr, time, T, dt, t_mid, T_mid, T_dot_mid, sigma, eps_0 = kernel_inputs

    # Max iter error (-4)
    _, err_max = _solve_tlv_numba_kernel.py_func(
        p_arr, time, T, dt, t_mid, T_mid, T_dot_mid, sigma, eps_0, True, 1e-15, 0, T_20_KELVIN, T_30_KELVIN
    )
    assert err_max == -4

    # Suppress NumPy runtime warnings for intentional divide-by-zero / overflow test cases
    with np.errstate(all="ignore"):
        # Stalled / Zero or NaN derivative error (-2)
        p_arr_zero = np.zeros(10, dtype=np.float64)
        _, err_stalled = _solve_tlv_numba_kernel.py_func(
            p_arr_zero, time, T, dt, t_mid, T_mid, T_dot_mid, sigma, eps_0, True, 1e-8, 100, T_20_KELVIN, T_30_KELVIN
        )
        assert err_stalled == -2

        # Non-finite divergence error (-3)
        p_arr_inf = np.array([1e300, 1e300, 100.0, 100.0, 10.0, 10.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float64)
        _, err_inf = _solve_tlv_numba_kernel.py_func(
            p_arr_inf, time, T, dt, t_mid, T_mid, T_dot_mid, sigma, eps_0, True, 1e-8, 100, T_20_KELVIN, T_30_KELVIN
        )
        assert err_inf == -3