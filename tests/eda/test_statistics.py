"""
Tests for creep_model.eda.statistics.
"""
import numpy as np
import pytest

from creep_model.eda.stage_classification import StageClassification, Plateau
from creep_model.eda.statistics import (
    first_valid_strain,
    steady_state_strain_rate,
    build_eda_dataframe,
)


def _classification_for(test_id, primary_end_idx, secondary_end_idx, k1=2, k2=2):
    """Minimal StageClassification builder -- plateaus list is irrelevant to
    steady_state_strain_rate, so an empty list is fine here."""
    return StageClassification(
        test_id=test_id,
        plateaus=[],
        primary_end_idx=primary_end_idx,
        secondary_end_idx=secondary_end_idx,
        k1=k1,
        k2=k2,
    )


class TestFirstValidStrain:
    def test_returns_first_strain_reading(self, make_test):
        test = make_test(strain_series=[0.001, 0.002, 0.003])
        assert first_valid_strain(test) == pytest.approx(0.001)


class TestSteadyStateStrainRate:
    def test_exact_known_slope(self, make_test):
        """
        Strain is EXACTLY linear (0.001 + 0.0002*t) over the whole test, so
        the secondary-creep region [start, end] should recover the slope
        0.0002 to floating-point precision, and mean_temp should match the
        mean of temperature_readings over the same slice.

        NOTE: temperature is put on the SAME time base as time_series here
        deliberately, to isolate testing the slope/mean-temp logic from the
        time-base bug covered separately below.
        """
        time = np.arange(20, dtype=np.float64)
        strain = 0.001 + 0.0002 * time
        temps = np.linspace(20.0, 23.5, num=20)  # matches time_series length
        test = make_test(
            strain_series=strain,
            time_series=time,
            temp_time_series=time.copy(),
            temperature_readings=temps,
        )
        classification = _classification_for(test.test_id, primary_end_idx=5, secondary_end_idx=15)

        slope, mean_temp = steady_state_strain_rate(test, classification)

        assert slope == pytest.approx(0.0002, rel=1e-6)
        assert mean_temp == pytest.approx(np.mean(temps[5:16]))

    def test_no_secondary_creep_returns_nan(self, make_test):
        test = make_test(strain_series=[0.001, 0.002, 0.003])
        classification = _classification_for(test.test_id, primary_end_idx=None, secondary_end_idx=None)

        slope, mean_temp = steady_state_strain_rate(test, classification)

        assert np.isnan(slope)
        assert np.isnan(mean_temp)

    def test_region_too_short_returns_nan(self, make_test):
        """Only a single point in [start, end] -- not enough to fit a line."""
        test = make_test(strain_series=[0.001, 0.002, 0.003, 0.004])
        classification = _classification_for(test.test_id, primary_end_idx=2, secondary_end_idx=2)

        slope, mean_temp = steady_state_strain_rate(test, classification)

        assert np.isnan(slope)
        assert np.isnan(mean_temp)

    def test_secondary_end_idx_none_falls_back_to_test_end(self, make_test):
        """secondary_end_idx=None means the test ended mid-secondary-creep --
        should use the last index of time_series, not crash."""
        time = np.arange(10, dtype=np.float64)
        strain = 0.001 + 0.0002 * time
        temps = np.linspace(20.0, 21.0, num=10)
        test = make_test(
            strain_series=strain,
            time_series=time,
            temp_time_series=time.copy(),
            temperature_readings=temps,
        )
        classification = _classification_for(test.test_id, primary_end_idx=3, secondary_end_idx=None)

        slope, mean_temp = steady_state_strain_rate(test, classification)

        assert slope == pytest.approx(0.0002, rel=1e-6)
        assert mean_temp == pytest.approx(np.mean(temps[3:10]))

    def test_steady_state_strain_rate_temperature_time_base_bug(self, make_test):
        """
        time_series has 20 points; temperature is recorded on a much coarser
        5-point temp_time_series (realistic: CreepData.xlsx temperature is
        sampled far less often than strain). Indexing
        `temperature_readings[start:end+1]` with strain-series indices
        (up to 19) against a 5-element array either raises or silently
        returns a near-empty/misaligned slice -- either way, NOT the
        temperature actually recorded near [start, end] in real time.
        """
        time = np.arange(20, dtype=np.float64)
        strain = 0.001 + 0.0002 * time
        temp_time = np.linspace(0, 19, num=5)  # coarse: only 5 readings
        temp_vals = np.array([20.0, 21.0, 22.0, 23.0, 24.0])
        test = make_test(
            strain_series=strain,
            time_series=time,
            temp_time_series=temp_time,
            temperature_readings=temp_vals,
        )
        classification = _classification_for(test.test_id, primary_end_idx=5, secondary_end_idx=15)

        _, mean_temp = steady_state_strain_rate(test, classification)

        # What it SHOULD be, using the properly interpolated temperature
        # onto the time_series basis:
        expected_mean_temp = np.mean(test.interpolate_temperature()[5:16])
        assert mean_temp == pytest.approx(expected_mean_temp)


class TestBuildEdaDataframe:
    def test_builds_expected_columns_and_values(self, make_test):
        time = np.arange(10, dtype=np.float64)
        strain_a = 0.001 + 0.0002 * time
        strain_b = 0.0015 + 0.0001 * time
        temps = np.linspace(20.0, 21.0, num=10)

        test_a = make_test(
            strain_series=strain_a, time_series=time, temp_time_series=time.copy(),
            temperature_readings=temps, test_id="A", applied_stress_MPa=10.0,
            age_days=3, print_quality="Standard",
        )
        test_b = make_test(
            strain_series=strain_b, time_series=time, temp_time_series=time.copy(),
            temperature_readings=temps, test_id="B", applied_stress_MPa=20.0,
            age_days=4, print_quality="High",
        )

        tests = {"A": test_a, "B": test_b}
        classifications = {
            "A": _classification_for("A", primary_end_idx=2, secondary_end_idx=8),
            "B": _classification_for("B", primary_end_idx=2, secondary_end_idx=8),
        }

        df = build_eda_dataframe(tests, classifications)

        expected_columns = {
            "Test_ID", "Applied_Stress_MPa", "Age_Days", "Print_Quality",
            "Mean_Temp_C_Secondary_Creep", "Eps_Tilde_0", "Eps_Dot_Ss",
        }
        assert set(df.columns) == expected_columns
        assert len(df) == 2
        assert set(df["Test_ID"]) == {"A", "B"}

        row_a = df.loc[df["Test_ID"] == "A"].iloc[0]
        assert row_a["Eps_Tilde_0"] == pytest.approx(strain_a[0])
        assert row_a["Eps_Dot_Ss"] == pytest.approx(0.0002, rel=1e-6)

    def test_missing_classification_raises(self, make_test):
        test_a = make_test(strain_series=[0.001, 0.002], test_id="A")
        with pytest.raises(ValueError, match="No classification found"):
            build_eda_dataframe({"A": test_a}, classifications={})