"""
Tests for creep_model.modelling.optimisation.bounds.TLVBounds.
"""
import numpy as np
import pytest

from creep_model.modelling.optimisation.bounds import TLVBounds
from creep_model.modelling.tlv.parameters import TLVParameters


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


class TestFromGroupData:
    def test_ee_ev_upper_bound_is_max_stress_over_strain(self, make_test):
        """Table 1.1: Ee/Ev upper bounds = max(sigma/eps) across the group
        being fit -- NOT left at +inf like the dataclass defaults."""
        test_a = make_test(
            strain_series=[0.01, 0.02], applied_stress_MPa=20.0, test_id="A"
        )  # max ratio = 20/0.02 = 1000
        test_b = make_test(
            strain_series=[0.05], applied_stress_MPa=10.0, test_id="B"
        )  # max ratio = 10/0.05 = 200

        bounds = TLVBounds.from_group_data([test_a, test_b])

        assert bounds.Ee_upper == pytest.approx(1000.0)
        assert bounds.Ev_upper == pytest.approx(1000.0)

    def test_skips_empty_tests(self, make_test):
        completed = make_test(
            strain_series=[0.02], applied_stress_MPa=20.0, test_id="A"
        )  # ratio = 1000
        empty = make_test(strain_series=[], test_id="B")

        bounds = TLVBounds.from_group_data([completed, empty])
        assert bounds.Ee_upper == pytest.approx(1000.0)

    def test_a_upper_defaults_but_is_overridable(self, make_test):
        test = make_test(strain_series=[0.01], applied_stress_MPa=10.0, test_id="A")

        default_bounds = TLVBounds.from_group_data([test])
        assert default_bounds.A_upper == pytest.approx(1e-5)

        custom_bounds = TLVBounds.from_group_data([test], A_upper=1e-3)
        assert custom_bounds.A_upper == pytest.approx(1e-3)

    def test_zero_strain_does_not_cause_division_by_zero(self, make_test):
        """from_group_data guards eps.max() with a floor of 1e-12, so a test
        whose strain never leaves exactly 0.0 shouldn't raise ZeroDivisionError
        or produce inf."""
        test = make_test(strain_series=[0.0, 0.0], applied_stress_MPa=10.0, test_id="A")
        bounds = TLVBounds.from_group_data([test])
        assert np.isfinite(bounds.Ee_upper)


class TestLowerUpperArrayOrdering:
    def test_array_order_matches_tlv_parameters_field_order(self):
        """Field order MUST match TLVParameters.to_array()/from_array()
        exactly -- A, n, m, Ee, Ev (each as 20/30 pairs) -- since fit_pipeline
        round-trips between normalized arrays and TLVParameters via this
        ordering."""
        bounds = TLVBounds(
            A_lower=0.0, A_upper=1.0,
            n_lower=2.0, n_upper=3.0,
            m_lower=-4.0, m_upper=4.0,
            Ee_lower=5.0, Ee_upper=6.0,
            Ev_lower=7.0, Ev_upper=8.0,
        )
        np.testing.assert_allclose(
            bounds.lower_array(), [0.0, 0.0, 2.0, 2.0, -4.0, -4.0, 5.0, 5.0, 7.0, 7.0]
        )
        np.testing.assert_allclose(
            bounds.upper_array(), [1.0, 1.0, 3.0, 3.0, 4.0, 4.0, 6.0, 6.0, 8.0, 8.0]
        )

    def test_as_unit_bounds_is_ten_unit_intervals(self):
        bounds = TLVBounds()
        unit_bounds = bounds.as_unit_bounds()
        assert unit_bounds == [(0.0, 1.0)] * 10


class TestNormalizeDenormalizeRoundTrip:
    def test_denormalize_of_normalize_recovers_original_params(self):
        bounds = TLVBounds(
            A_lower=0.0, A_upper=1e-5,
            n_lower=0.0, n_upper=5.0,
            m_lower=-5.0, m_upper=5.0,
            Ee_lower=0.0, Ee_upper=1000.0,
            Ev_lower=0.0, Ev_upper=1000.0,
        )
        params = _make_params()

        x_normalized = bounds.normalize(params)
        recovered = bounds.denormalize(x_normalized)

        np.testing.assert_allclose(recovered.to_array(), params.to_array(), rtol=1e-9)

    def test_normalize_maps_lower_bound_to_zero_and_upper_to_one(self):
        bounds = TLVBounds(
            A_lower=0.0, A_upper=10.0,
            n_lower=0.0, n_upper=10.0,
            m_lower=-10.0, m_upper=10.0,
            Ee_lower=0.0, Ee_upper=10.0,
            Ev_lower=0.0, Ev_upper=10.0,
        )
        params_at_lower = TLVParameters(
            A20=0.0, A30=0.0, n20=0.0, n30=0.0, m20=-10.0, m30=-10.0,
            Ee20=0.0, Ee30=0.0, Ev20=0.0, Ev30=0.0,
        )
        x = bounds.normalize(params_at_lower)
        np.testing.assert_allclose(x, np.zeros(10))

    def test_denormalize_at_unit_midpoint_gives_bounds_midpoint(self):
        bounds = TLVBounds(
            A_lower=0.0, A_upper=10.0,
            n_lower=0.0, n_upper=10.0,
            m_lower=-10.0, m_upper=10.0,
            Ee_lower=0.0, Ee_upper=10.0,
            Ev_lower=0.0, Ev_upper=10.0,
        )
        x_mid = np.full(10, 0.5)
        params = bounds.denormalize(x_mid)
        # A, n, Ee, Ev midpoint of [0,10] = 5.0; m midpoint of [-10,10] = 0.0
        assert params.A20 == pytest.approx(5.0)
        assert params.m20 == pytest.approx(0.0)