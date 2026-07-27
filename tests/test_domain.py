"""
Tests for creep_model.domain -- CreepTest and CreepExperiment.
"""
import numpy as np
import pytest

from creep_model.domain import CreepExperiment


class TestCreepTestBasics:
    def test_is_empty_true_for_no_data(self, make_test):
        test = make_test(strain_series=[])
        assert test.is_empty is True

    def test_is_empty_false_for_data(self, make_test):
        test = make_test(strain_series=[0.001, 0.002])
        assert test.is_empty is False

    def test_final_time_returns_last_time_point(self, make_test):
        test = make_test(strain_series=[0.001, 0.002, 0.003], time_series=[0.0, 5.0, 10.0])
        assert test.final_time == pytest.approx(10.0)

    def test_final_time_zero_when_empty(self, make_test):
        test = make_test(strain_series=[])
        assert test.final_time == 0.0

    def test_eps_0_returns_first_strain_reading(self, make_test):
        test = make_test(strain_series=[0.0012, 0.002, 0.003])
        assert test.eps_0 == pytest.approx(0.0012)

    def test_eps_0_nan_when_empty(self, make_test):
        test = make_test(strain_series=[])
        assert np.isnan(test.eps_0)

    def test_mean_temperature(self, make_test):
        test = make_test(
            strain_series=[0.001, 0.002, 0.003],
            temperature_readings=[20.0, 22.0, 24.0],
        )
        assert test.mean_temperature == pytest.approx(22.0)

    def test_mean_temperature_nan_when_empty(self, make_test):
        test = make_test(strain_series=[])
        assert np.isnan(test.mean_temperature)


class TestGetStrainAtTime:
    def test_interpolates_between_points(self, make_test):
        test = make_test(
            strain_series=[0.0, 0.002, 0.004],
            time_series=[0.0, 10.0, 20.0],
        )
        # Halfway between t=0 (strain 0.0) and t=10 (strain 0.002)
        assert test.get_strain_at_time(5.0) == pytest.approx(0.001)

    def test_exact_match_at_recorded_time(self, make_test):
        test = make_test(
            strain_series=[0.0, 0.002, 0.004],
            time_series=[0.0, 10.0, 20.0],
        )
        assert test.get_strain_at_time(10.0) == pytest.approx(0.002)

    def test_nan_when_empty(self, make_test):
        test = make_test(strain_series=[])
        assert np.isnan(test.get_strain_at_time(5.0))


class TestInterpolateTemperature:
    def test_matches_time_series_length(self, make_test):
        """interpolate_temperature() must resample onto time_series's basis,
        regardless of how coarse temp_time_series is (Thesis: temperature is
        recorded far less often than strain -- see docs/methodology.md)."""
        test = make_test(
            strain_series=np.arange(10, dtype=np.float64),
            time_series=np.arange(10, dtype=np.float64),
            temp_time_series=np.array([0.0, 4.5, 9.0]),
            temperature_readings=np.array([20.0, 22.0, 24.0]),
        )
        interp = test.interpolate_temperature()
        assert len(interp) == len(test.time_series)

    def test_linear_interpolation_values(self, make_test):
        test = make_test(
            strain_series=np.arange(5, dtype=np.float64),
            time_series=np.array([0.0, 1.0, 2.0, 3.0, 4.0]),
            temp_time_series=np.array([0.0, 4.0]),
            temperature_readings=np.array([20.0, 24.0]),
        )
        interp = test.interpolate_temperature()
        # Linear ramp from 20 to 24 over t=[0,4] -> +1 degree per second
        np.testing.assert_allclose(interp, [20.0, 21.0, 22.0, 23.0, 24.0])

    def test_nan_in_temperature_readings_is_not_filtered(self, make_test):
        """
        KNOWN BUG: interpolate_temperature() computes `valid_mask =
        ~np.isnan(self.temperature_readings)` but never uses it -- dead
        code. np.interp has no NaN-awareness, so a NaN anywhere in
        temperature_readings poisons every interpolated point that falls
        between the NaN's neighbouring temp_time_series entries (not just
        the exact NaN timestamp).

        In practice this doesn't currently bite real data, because
        io/parser.py already drops NaN temperature rows via
        `df_temp.dropna()` before a CreepTest is ever constructed -- so a
        CreepTest built by the real pipeline never has NaNs here. It WOULD
        bite any future code path that constructs CreepTest directly from
        partially-missing sensor data (e.g. a live/ongoing-test ingestion
        path) without the same upstream dropna(). This test pins the
        CURRENT (buggy) behaviour so a silent "fix" doesn't regress
        unnoticed, and flags it for a real fix (filter using valid_mask
        before interpolating) if that use case ever appears.
        """
        test = make_test(
            strain_series=np.arange(5, dtype=np.float64),
            time_series=np.array([0.0, 1.0, 2.0, 3.0, 4.0]),
            temp_time_series=np.array([0.0, 2.0, 4.0]),
            temperature_readings=np.array([20.0, np.nan, 24.0]),
        )
        interp = test.interpolate_temperature()
        # np.interp returns the exact fp value at points that exactly match
        # an xp entry (t=0.0 -> 20.0, t=4.0 -> 24.0), so those two endpoints
        # stay finite. Every OTHER point (t=1,2,3) falls strictly between
        # two xp entries that bracket the NaN and gets poisoned to NaN --
        # this is the actual shape of the "unused valid_mask" bug: not a
        # total wipeout, but silent NaN contamination of every interior
        # point near a missing temperature reading.
        assert not np.isnan(interp[0])
        assert not np.isnan(interp[-1])
        assert np.isnan(interp[1:-1]).all()


class TestCreepExperiment:
    def test_shortest_test_duration_ignores_empty_tests(self, make_test):
        completed_short = make_test(
            strain_series=[0.001, 0.002], time_series=[0.0, 5.0], test_id="short"
        )
        completed_long = make_test(
            strain_series=[0.001, 0.002, 0.003], time_series=[0.0, 5.0, 20.0], test_id="long"
        )
        empty = make_test(strain_series=[], test_id="empty")

        experiment = CreepExperiment(
            tests={"short": completed_short, "long": completed_long, "empty": empty}
        )
        assert experiment.shortest_test_duration == pytest.approx(5.0)

    def test_shortest_test_duration_raises_when_no_completed_tests(self, make_test):
        empty = make_test(strain_series=[], test_id="empty")
        experiment = CreepExperiment(tests={"empty": empty})
        with pytest.raises(ValueError, match="No valid completed tests"):
            experiment.shortest_test_duration