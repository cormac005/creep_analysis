import pytest
import numpy as np
from unittest.mock import MagicMock, patch

from creep_model.modelling.tlv.trimming import trim_tertiary, trim_and_partition


@patch("creep_model.modelling.tlv.trimming.classify_stages")
def test_trim_tertiary_no_trimming_needed(mock_classify):
    """
    If no stage bounds are detected and temperature data covers the entire time 
    series, the test should be returned unchanged.
    """
    mock_classify.return_value = MagicMock(primary_end_idx=None, secondary_end_idx=None)
    
    test = MagicMock()
    test.time_series = np.array([0, 10, 20])
    test.strain_series = np.array([0.0, 0.1, 0.2])
    test.temp_time_series = np.array([0, 10, 30]) # Max temp time (30) > Max time (20)
    
    result = trim_tertiary(test, 10, 20)
    assert result is test


@patch("creep_model.modelling.tlv.trimming.classify_stages")
@patch("creep_model.modelling.tlv.trimming.replace")
def test_trim_tertiary_trims_at_primary_end(mock_replace, mock_classify):
    """
    Should trim at primary_end_idx, removing both secondary and tertiary creep.
    """
    mock_classify.return_value = MagicMock(primary_end_idx=5, secondary_end_idx=10)
    
    test = MagicMock()
    test.time_series = np.arange(20)
    test.strain_series = np.arange(20)
    test.temp_time_series = None # Ignore temperature trimming
    
    mock_replace.return_value = "trimmed_test"
    result = trim_tertiary(test, 10, 20)
    
    assert result == "trimmed_test"
    mock_replace.assert_called_once()
    
    # Verify it was trimmed to index 5 (length 6)
    called_time_series = mock_replace.call_args[1]["time_series"]
    assert len(called_time_series) == 6


@patch("creep_model.modelling.tlv.trimming.classify_stages")
@patch("creep_model.modelling.tlv.trimming.replace")
def test_trim_tertiary_fallback_to_secondary_end(mock_replace, mock_classify):
    """
    Should fallback to trimming at secondary_end_idx if primary_end_idx is not found.
    """
    mock_classify.return_value = MagicMock(primary_end_idx=None, secondary_end_idx=10)
    
    test = MagicMock()
    test.time_series = np.arange(20)
    test.strain_series = np.arange(20)
    test.temp_time_series = None
    
    trim_tertiary(test, 10, 20)
    
    # Verify it was trimmed to index 10 (length 11)
    called_time_series = mock_replace.call_args[1]["time_series"]
    assert len(called_time_series) == 11


@patch("creep_model.modelling.tlv.trimming.classify_stages")
@patch("creep_model.modelling.tlv.trimming.replace")
def test_trim_tertiary_temperature_cutoff(mock_replace, mock_classify):
    """
    Should trim data after the last temperature reading, even if no creep stages 
    triggered a trim.
    """
    mock_classify.return_value = MagicMock(primary_end_idx=None, secondary_end_idx=None)
    
    test = MagicMock()
    test.time_series = np.array([0, 10, 20, 30, 40])
    test.strain_series = np.array([0.0, 0.1, 0.2, 0.3, 0.4])
    test.temp_time_series = np.array([0, 15, 25]) # Max temp time is 25
    
    trim_tertiary(test, 10, 20)
    
    # Valid time indices <= 25 are 0 (0s), 1 (10s), 2 (20s). So cutoff index is 2. Length is 3.
    called_time_series = mock_replace.call_args[1]["time_series"]
    assert len(called_time_series) == 3


@patch("creep_model.modelling.tlv.trimming.classify_stages")
@patch("creep_model.modelling.tlv.trimming.replace")
def test_trim_tertiary_min_of_stage_and_temp(mock_replace, mock_classify):
    """
    Should take the strictest (minimum) cutoff index between stage trimming 
    and temperature trimming.
    """
    mock_classify.return_value = MagicMock(primary_end_idx=3, secondary_end_idx=None)
    
    test = MagicMock()
    test.time_series = np.array([0, 10, 20, 30, 40])
    test.strain_series = np.array([0.0, 0.1, 0.2, 0.3, 0.4])
    test.temp_time_series = np.array([0, 15]) # Max temp time is 15
    
    trim_tertiary(test, 10, 20)
    
    # Stage cutoff = 3. Temp cutoff = 1 (since 10 <= 15). Min is 1 (length 2).
    called_time_series = mock_replace.call_args[1]["time_series"]
    assert len(called_time_series) == 2


@patch("creep_model.modelling.tlv.trimming.trim_tertiary")
def test_trim_and_partition(mock_trim):
    mock_exp = MagicMock()
    
    # Setup tests: 1 empty, 1 fails trim, 1 High, 1 Standard
    t_empty = MagicMock(is_empty=True)
    t_fail = MagicMock(is_empty=False, test_id="FailT")
    t_high = MagicMock(is_empty=False, print_quality="High")
    t_std = MagicMock(is_empty=False, print_quality="Standard")
    
    mock_exp.tests = {"1": t_empty, "2": t_fail, "3": t_high, "4": t_std}
    
    # Mock behavior of trim_tertiary
    def mock_trim_side_effect(test, k1, k2):
        if test is t_fail:
            raise ValueError()
        return test
    mock_trim.side_effect = mock_trim_side_effect
    
    groups = trim_and_partition(mock_exp, 10, 20)
    
    assert len(groups["High"]) == 1
    assert len(groups["Standard"]) == 1
    assert groups["High"][0] is t_high
    assert groups["Standard"][0] is t_std