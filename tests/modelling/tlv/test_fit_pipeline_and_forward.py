import pytest
import numpy as np
from unittest.mock import MagicMock, patch
from creep_model.modelling.tlv.fit_pipeline import (
    _group_mse, _group_mse_prepared, _de_objective, _lm_residuals, fit_group, _NON_CONVERGENCE_PENALTY
)
from creep_model.modelling.tlv.solver import SolverConvergenceError
from creep_model.modelling.tlv.forward_model import TLVCreepModel

@patch("creep_model.modelling.tlv.fit_pipeline.solve_tlv")
def test_group_mse(mock_solve):
    mock_solve.return_value = np.array([2.0, 2.0])
    test = MagicMock(strain_series=np.array([0.0, 0.0]))
    assert _group_mse(MagicMock(), [test]) == 4.0

@patch("creep_model.modelling.tlv.fit_pipeline.solve_tlv_prepared")
def test_group_mse_prepared(mock_solve_prep):
    mock_solve_prep.return_value = np.array([2.0, 2.0])
    prep_mock = MagicMock()
    strain_series = np.array([0.0, 0.0])
    params_mock = MagicMock()
    assert _group_mse_prepared(params_mock, [(prep_mock, strain_series)]) == 4.0

@patch("creep_model.modelling.tlv.fit_pipeline._group_mse")
def test_de_objective(mock_mse):
    mock_mse.return_value = 0.5
    mock_bounds = MagicMock()
    mock_bounds.denormalize.return_value = MagicMock()
    
    res = _de_objective(np.ones(10), [], mock_bounds)
    assert res == 0.5
    
    mock_mse.side_effect = SolverConvergenceError("failed")
    res2 = _de_objective(np.ones(10), [], mock_bounds)
    assert res2 == _NON_CONVERGENCE_PENALTY

@patch("creep_model.modelling.tlv.fit_pipeline._group_mse_prepared")
def test_de_objective_prepared_tuple(mock_mse_prep):
    mock_mse_prep.return_value = 0.3
    mock_bounds = MagicMock()
    mock_bounds.denormalize.return_value = MagicMock()
    
    item = (MagicMock(), np.array([1.0, 2.0]))
    res = _de_objective(np.ones(10), [item], mock_bounds)
    assert res == 0.3

@patch("creep_model.modelling.tlv.fit_pipeline.solve_tlv")
@patch("creep_model.modelling.tlv.fit_pipeline.unscale")
@patch("creep_model.modelling.tlv.parameters.TLVParameters.from_array")
def test_lm_residuals(mock_from_array, mock_unscale, mock_solve):
    mock_unscale.return_value = np.ones(10)
    mock_solve.return_value = np.array([5.0, 6.0])
    
    test = MagicMock(strain_series=np.array([5.0, 5.0]), test_id="T1")
    res = _lm_residuals(np.ones(10), [test], np.ones(10))
    np.testing.assert_array_equal(res, np.array([0.0, 1.0]))

@patch("creep_model.modelling.tlv.fit_pipeline.solve_tlv")
@patch("creep_model.modelling.tlv.fit_pipeline.unscale")
@patch("creep_model.modelling.tlv.parameters.TLVParameters.from_array")
def test_lm_residuals_convergence_error(mock_from_array, mock_unscale, mock_solve):
    mock_unscale.return_value = np.ones(10)
    mock_solve.side_effect = SolverConvergenceError("solver failed")
    
    test = MagicMock(strain_series=np.array([5.0, 5.0]), test_id="T1")
    res = _lm_residuals(np.ones(10), [test], np.ones(10))
    expected_penalty = np.sqrt(_NON_CONVERGENCE_PENALTY)
    np.testing.assert_array_equal(res, np.array([expected_penalty, expected_penalty]))

@patch("creep_model.modelling.tlv.fit_pipeline.solve_tlv_prepared")
@patch("creep_model.modelling.tlv.fit_pipeline.unscale")
@patch("creep_model.modelling.tlv.parameters.TLVParameters.from_array")
def test_lm_residuals_prepared_tuple(mock_from_array, mock_unscale, mock_solve_prep):
    mock_unscale.return_value = np.ones(10)
    mock_solve_prep.return_value = np.array([5.0, 6.0])
    
    prep_mock = MagicMock(spec=["test_id"])
    prep_mock.test_id = "T_prep"
    item = (prep_mock, np.array([5.0, 5.0]))
    
    res = _lm_residuals(np.ones(10), [item], np.ones(10))
    np.testing.assert_array_equal(res, np.array([0.0, 1.0]))
    
    mock_solve_prep.side_effect = SolverConvergenceError("prep solver failed")
    res_err = _lm_residuals(np.ones(10), [item], np.ones(10))
    expected_penalty = np.sqrt(_NON_CONVERGENCE_PENALTY)
    np.testing.assert_array_equal(res_err, np.array([expected_penalty, expected_penalty]))

@patch("creep_model.modelling.tlv.fit_pipeline.differential_evolution")
@patch("creep_model.modelling.tlv.fit_pipeline.least_squares")
@patch("creep_model.modelling.tlv.fit_pipeline.compute_scale_factors")
@patch("creep_model.modelling.tlv.fit_pipeline.prepare_test_data")
def test_fit_group(mock_prep, mock_scale_factors, mock_least_squares, mock_de, capsys):
    mock_bounds = MagicMock()
    mock_bounds.as_unit_bounds.return_value = []
    
    mock_params = MagicMock()
    mock_params.to_array.return_value = np.ones(10)
    mock_bounds.denormalize.return_value = mock_params
    
    mock_de_res = MagicMock(success=False, message="failed", fun=_NON_CONVERGENCE_PENALTY + 1.0, x=np.ones(10))
    mock_de.return_value = mock_de_res
    
    mock_ls_res = MagicMock(success=False, message="failed", x=np.ones(10))
    mock_least_squares.return_value = mock_ls_res
    mock_scale_factors.return_value = np.ones(10)
    
    res = fit_group([MagicMock(strain_series=np.array([1.0, 2.0]))], mock_bounds, lm_kwargs={"method": "trf"})
    assert res is not None

    captured = capsys.readouterr()
    assert "Warning: DE did not report success" in captured.out
    assert "Warning: DE's best candidate never converged" in captured.out
    assert "Warning: Local refinement solver did not report success" in captured.out

def test_tlv_creep_model():
    model = TLVCreepModel()
    with pytest.raises(NotImplementedError):
        model.fit(None, None)
    with pytest.raises(NotImplementedError):
        model._predict(None)
    with pytest.raises(RuntimeError, match="has no fitted parameters"):
        model.predict_test(MagicMock())

@patch("creep_model.modelling.tlv.forward_model.solve_tlv")
def test_tlv_creep_model_predict_test(mock_solve):
    mock_params = MagicMock()
    mock_params.to_array.return_value = np.ones(10)
    
    model = TLVCreepModel(params=mock_params)
    assert (model.fitted_params_ == np.ones(10)).all()
    
    model.predict_test(MagicMock())
    mock_solve.assert_called_once()