import pytest
from unittest.mock import MagicMock
from creep_model.modelling.tlv.initial_conditions import f_ratio, sigma_ep_0, sigma_ep_0_from_measurement

def test_f_ratio():
    mock_params = MagicMock()
    mock_params.at_temperature.return_value = {"Ev": 300, "Ee": 100}
    
    result = f_ratio(T_kelvin=300.0, params=mock_params)
    assert result == 0.75
    mock_params.at_temperature.assert_called_once_with(300.0)

def test_sigma_ep_0():
    mock_params = MagicMock()
    mock_params.at_temperature.return_value = {"Ev": 100, "Ee": 300}
    
    result = sigma_ep_0(applied_stress_MPa=10.0, T0_kelvin=293.15, params=mock_params)
    assert result == 7.5

def test_sigma_ep_0_from_measurement():
    mock_params = MagicMock()
    mock_params.at_temperature.return_value = {"Ee": 150}
    
    result = sigma_ep_0_from_measurement(0.02, 300.0, mock_params)
    assert result == 3.0
    mock_params.at_temperature.assert_called_once_with(300.0)