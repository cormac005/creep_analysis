import pytest
import numpy as np
from unittest.mock import MagicMock, patch
from creep_model.modelling.tlv.solver import _newton_raphson_step, solve_tlv, SolverConvergenceError


@patch("creep_model.modelling.tlv.solver.residual")
@patch("creep_model.modelling.tlv.solver.residual_derivative")
def test_newton_raphson_step_success(mock_dr, mock_r):
    # Iteration 1: residual is 2.0, derivative is 4.0 -> steps by -0.5 (guess goes from 10 -> 9.5)
    # Iteration 2: residual is 0.0 -> converges and returns 9.5
    mock_r.side_effect = [2.0, 0.0]
    mock_dr.return_value = 4.0

    # Added T_dot=0.0 as the 8th positional argument
    res = _newton_raphson_step(10.0, 5.0, 298.0, 298.0, 0.0, 1.0, MagicMock(), 0.0, tol=1e-8)
    assert res == 9.5


@patch("creep_model.modelling.tlv.solver.residual")
@patch("creep_model.modelling.tlv.solver.residual_derivative")
def test_newton_raphson_step_zero_derivative(mock_dr, mock_r):
    mock_r.return_value = 2.0
    mock_dr.return_value = 0.0  # Stall

    with pytest.raises(SolverConvergenceError, match="stalled"):
        _newton_raphson_step(10.0, 5.0, 298.0, 298.0, 0.0, 1.0, MagicMock(), 0.0)


@patch("creep_model.modelling.tlv.solver.residual")
@patch("creep_model.modelling.tlv.solver.residual_derivative")
def test_newton_raphson_step_max_iter(mock_dr, mock_r):
    mock_r.return_value = 2.0
    mock_dr.return_value = 1.0

    with pytest.raises(SolverConvergenceError, match="did not converge"):
        _newton_raphson_step(10.0, 5.0, 298.0, 298.0, 0.0, 1.0, MagicMock(), 0.0, max_iter=3)


@patch("creep_model.modelling.tlv.solver.sigma_ep_0")
@patch("creep_model.modelling.tlv.solver._newton_raphson_step")
def test_solve_tlv(mock_nr_step, mock_sigma_0):
    mock_test = MagicMock()
    mock_test.time_series = np.array([0.0, 1.0, 2.0])
    mock_test.interpolate_temperature.return_value = np.array([26.85, 27.85, 28.85])  # + 273.15 -> [300, 301, 302]
    mock_test.applied_stress_MPa = 50.0

    # Mocks required for polynomial T_dot calculation in solve_tlv
    mock_poly = MagicMock()
    mock_poly.deriv.return_value = lambda t: 0.05
    mock_test.temperature_polynomial.return_value = mock_poly
    mock_test.temp_time_series = np.array([0.0, 2.0])

    mock_params = MagicMock()
    mock_params.at_temperature.return_value = {"Ee": 2.0}

    mock_sigma_0.return_value = 10.0
    mock_nr_step.side_effect = [12.0, 14.0]

    # Explicitly pass use_measured_initial_condition=False so it uses sigma_ep_0
    strain = solve_tlv(mock_test, mock_params, use_measured_initial_condition=False)

    # Expected sigma_ep array: [10.0, 12.0, 14.0], Ee = 2.0 -> strain = sigma_ep / Ee = [5.0, 6.0, 7.0]
    np.testing.assert_array_equal(strain, np.array([5.0, 6.0, 7.0]))
    assert mock_nr_step.call_count == 2


@patch("creep_model.modelling.tlv.solver.residual")
def test_newton_raphson_step_translates_value_error(mock_r):
    mock_r.side_effect = ValueError("Invalid state: sigma_ep_mid > sigma")
    with pytest.raises(SolverConvergenceError, match="unphysical state"):
        _newton_raphson_step(10.0, 5.0, 298.0, 298.0, 0.0, 1.0, MagicMock(), 0.0)