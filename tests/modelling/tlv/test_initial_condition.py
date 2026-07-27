import pytest
from unittest.mock import MagicMock
from creep_model.modelling.tlv.initial_conditions import f_ratio, sigma_ep_0

def test_f_ratio():
    mock_params = MagicMock()
    # At Ev=300 and Ee=100, f_ratio should be 300 / 400 = 0.75
    mock_params.at_temperature.return_value = {"Ev": 300, "Ee": 100}
    
    result = f_ratio(T_kelvin=300.0, params=mock_params)
    assert result == 0.75
    mock_params.at_temperature.assert_called_once_with(300.0)

def test_sigma_ep_0():
    mock_params = MagicMock()
    # At Ev=100 and Ee=300, f_ratio = 100 / 400 = 0.25
    mock_params.at_temperature.return_value = {"Ev": 100, "Ee": 300}
    
    # sigma_ep_0 = (1 - 0.25) * 10.0 = 7.5
    result = sigma_ep_0(applied_stress_MPa=10.0, T0_kelvin=293.15, params=mock_params)
    assert result == 7.5