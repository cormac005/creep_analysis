import pytest
from unittest.mock import MagicMock, patch
from creep_model.modelling.tlv.trimming import trim_tertiary, trim_and_partition

@patch("creep_model.modelling.tlv.trimming.classify_stages")
def test_trim_tertiary_primary_only_returns_unchanged(mock_classify):
    """primary_end_idx=None means the test never left primary creep --
    tertiary creep is therefore impossible, so the test should be
    returned unchanged, not skipped/raised."""
    mock_classify.return_value = MagicMock(primary_end_idx=None)
    test = MagicMock(test_id="T1")

    result = trim_tertiary(test, 10, 20)
    assert result is test

@patch("creep_model.modelling.tlv.trimming.classify_stages")
def test_trim_tertiary_no_tertiary(mock_classify):
    mock_classify.return_value = MagicMock(primary_end_idx=5, secondary_end_idx=None)
    test = MagicMock(test_id="T1")
    
    result = trim_tertiary(test, 10, 20)
    assert result is test  # Unchanged

@patch("creep_model.modelling.tlv.trimming.classify_stages")
@patch("creep_model.modelling.tlv.trimming.replace")
def test_trim_tertiary_trims(mock_replace, mock_classify):
    mock_classify.return_value = MagicMock(primary_end_idx=5, secondary_end_idx=10)
    test = MagicMock(time_series=[0]*20, strain_series=[0]*20)
    
    mock_replace.return_value = "trimmed_test"
    result = trim_tertiary(test, 10, 20)
    
    assert result == "trimmed_test"
    mock_replace.assert_called_once()

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