"""
Tests for creep_model.modelling.tlv.parameters.TLVParameters.
"""
import numpy as np
import pytest

from creep_model.modelling.tlv.parameters import (
    TLVParameters,
    T_20_KELVIN,
    T_30_KELVIN,
)


def _make_params(**overrides):
    defaults = dict(
        A20=1e-6, A30=2e-6,
        n20=1.5, n30=1.8,
        m20=-0.3, m30=-0.2,
        Ee20=500.0, Ee30=400.0,
        Ev20=800.0, Ev30=700.0,
    )
    defaults.update(overrides)
    return TLVParameters(**defaults)


class TestAtTemperature:
    def test_returns_exact_anchor_values_at_20C(self):
        params = _make_params()
        result = params.at_temperature(T_20_KELVIN)
        assert result["A"] == pytest.approx(params.A20)
        assert result["n"] == pytest.approx(params.n20)
        assert result["m"] == pytest.approx(params.m20)
        assert result["Ee"] == pytest.approx(params.Ee20)
        assert result["Ev"] == pytest.approx(params.Ev20)

    def test_returns_exact_anchor_values_at_30C(self):
        params = _make_params()
        result = params.at_temperature(T_30_KELVIN)
        assert result["A"] == pytest.approx(params.A30)
        assert result["n"] == pytest.approx(params.n30)
        assert result["m"] == pytest.approx(params.m30)
        assert result["Ee"] == pytest.approx(params.Ee30)
        assert result["Ev"] == pytest.approx(params.Ev30)

    def test_linear_interpolation_at_midpoint(self):
        params = _make_params(Ee20=400.0, Ee30=600.0)
        T_mid = (T_20_KELVIN + T_30_KELVIN) / 2.0
        result = params.at_temperature(T_mid)
        assert result["Ee"] == pytest.approx(500.0)

    def test_extrapolates_outside_20_30_range(self):
        """Eq. 1.3 is a straight-line fit between the two anchors -- nothing
        in the formula clamps T to [20C, 30C], so a value outside that
        range should extrapolate linearly rather than clip or raise. This
        matters because thesis notes mild ambient fluctuation can exceed
        the two anchor points slightly."""
        params = _make_params(Ee20=400.0, Ee30=600.0)
        T_above_30 = T_30_KELVIN + (T_30_KELVIN - T_20_KELVIN)  # i.e. "40C"
        result = params.at_temperature(T_above_30)
        assert result["Ee"] == pytest.approx(800.0)

    def test_accepts_array_input(self):
        params = _make_params(Ee20=400.0, Ee30=600.0)
        T_array = np.array([T_20_KELVIN, T_30_KELVIN])
        result = params.at_temperature(T_array)
        np.testing.assert_allclose(result["Ee"], [400.0, 600.0])


class TestTemperatureDerivatives:
    def test_dEe_dT_matches_slope(self):
        params = _make_params(Ee20=400.0, Ee30=600.0)
        expected_slope = (600.0 - 400.0) / (T_30_KELVIN - T_20_KELVIN)
        assert params.dEe_dT() == pytest.approx(expected_slope)

    def test_dEv_dT_matches_slope(self):
        params = _make_params(Ev20=800.0, Ev30=700.0)
        expected_slope = (700.0 - 800.0) / (T_30_KELVIN - T_20_KELVIN)
        assert params.dEv_dT() == pytest.approx(expected_slope)

    def test_zero_slope_when_20_and_30_values_equal(self):
        params = _make_params(Ee20=500.0, Ee30=500.0)
        assert params.dEe_dT() == pytest.approx(0.0)


class TestArrayRoundTrip:
    def test_to_array_from_array_round_trip(self):
        params = _make_params()
        arr = params.to_array()
        assert arr.shape == (10,)
        reconstructed = TLVParameters.from_array(arr)
        assert reconstructed == params

    def test_to_array_field_order(self):
        """Field order is load-bearing -- TLVBounds and the scaling module
        both assume this exact ordering (see their module docstrings)."""
        params = _make_params(
            A20=1, A30=2, n20=3, n30=4, m20=5, m30=6,
            Ee20=7, Ee30=8, Ev20=9, Ev30=10,
        )
        np.testing.assert_allclose(
            params.to_array(), [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        )