import pytest
import numpy as np
from unittest.mock import patch, MagicMock
from creep_model.modelling.empirical import (
    findley_law, modified_findley_law, global_creep_law,
    LocalFindleyModel, QuantizedFindleyModel, LocalModifiedFindleyModel, 
    GlobalMSEModel
)

def test_math_laws():
    assert findley_law(2.0, 1.0, 2.0, 3.0) == 17.0
    assert modified_findley_law(2.0, 1.0, 2.0, 3.0, 2.0) == 17.0
    X = np.array([[2.0], [5.0]])
    res = global_creep_law(X, 2.0, 3.0, 2.0, 3.0)
    assert res[0] == 610.0

@patch("creep_model.modelling.empirical.curve_fit")
def test_local_findley_model(mock_cf):
    mock_cf.return_value = ([0.1, 0.2, 0.3], None)
    model = LocalFindleyModel()
    model.fit(np.array([0, 1]), np.array([1, 2]))
    assert list(model.fitted_params_) == [0.1, 0.2, 0.3]
    
    with patch("creep_model.modelling.empirical.findley_law") as mock_law:
        model.predict(np.array([1]))
        mock_law.assert_called_once()
        
    unfitted_model = LocalFindleyModel()
    with pytest.raises(ValueError, match="must be fitted"):
        unfitted_model.predict(np.array([1]))

@patch("creep_model.modelling.empirical.minimize")
def test_quantized_findley_model(mock_min):
    mock_res = MagicMock(success=False, message="struggled", x=np.array([1, 2, 3]))
    mock_min.return_value = mock_res
    
    model = QuantizedFindleyModel()
    model.fit(np.array([1, 2]), np.array([1, 2]))
    assert (model.fitted_params_ == np.array([1, 2, 3])).all()
    
    loss = model._custom_loss([0, 0, 0], np.array([1]), np.array([10]))
    assert loss > 81.0

    with pytest.raises(ValueError, match="not fitted"):
        QuantizedFindleyModel().predict(np.array([1.0]))

@patch("creep_model.modelling.empirical.minimize")
def test_local_modified_findley_model(mock_min):
    mock_res = MagicMock(success=True, fun=1.0, x=np.array([1, 2, 3, 4]))
    mock_min.return_value = mock_res
    
    model = LocalModifiedFindleyModel(n_starts=2)
    model.fit(np.array([1, 2]), np.array([1, 2]))
    assert mock_min.call_count == 2
    
    with patch("creep_model.modelling.empirical.modified_findley_law") as mock_law:
        model.predict(np.array([1]))
        mock_law.assert_called_once()
        
    mock_min.return_value = MagicMock(success=False)
    bad_model = LocalModifiedFindleyModel(n_starts=1)
    with pytest.raises(RuntimeError, match="Optimization failed"):
        bad_model.fit(np.array([1, 2]), np.array([1, 2]))

    with pytest.raises(ValueError, match="not fitted"):
        LocalModifiedFindleyModel().predict(np.array([1.0]))
        
def test_local_modified_findley_model_mle_loss():
    model = LocalModifiedFindleyModel()
    params = [0.1, 0.1, 0.1, 0.1]
    time = np.array([1.0])
    y_obs = np.array([1.0])
    loss = model._default_rounded_mle_loss(params, time, y_obs)
    assert isinstance(loss, float)

@patch("creep_model.modelling.empirical.curve_fit")
def test_global_mse_model(mock_cf):
    mock_cf.return_value = ([1, 2, 3, 4], None)
    model = GlobalMSEModel()
    
    X_T = np.array([[1, 2, 3], [4, 5, 6]])
    model.fit(X_T, np.array([1, 2, 3]))
    assert list(model.fitted_params_) == [1, 2, 3, 4]
    
    X = X_T.T
    model.fit(X, np.array([1, 2, 3]))
    
    with patch("creep_model.modelling.empirical.global_creep_law") as mock_law:
        model.predict(X_T)
        model.predict(X)
        assert mock_law.call_count == 2
        
    with pytest.raises(ValueError, match="Unexpected X shape"):
        model._prepare_input(np.array([1, 2, 3]))
        
    with pytest.raises(ValueError, match="not fitted"):
        GlobalMSEModel().predict(X)