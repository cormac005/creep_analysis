"""
Tests for creep_model.modelling.optimisation.scaling.

NOTE: src/creep_model/modelling/optimisation/ has no __init__.py (unlike
every other subpackage in the repo, e.g. modelling/tlv/__init__.py) --
it currently works only because Python 3.3+ implicit namespace packages
paper over the gap for an editable install. Worth adding the missing
__init__.py for consistency with the rest of the package and to avoid
surprises if this is ever packaged as a real wheel (hatchling's default
src-layout discovery expects explicit packages). Not blocking these tests,
but flagged here since it's easy to miss.
"""
import numpy as np
import pytest

from creep_model.modelling.optimisation.scaling import (
    compute_scale_factors,
    scale,
    unscale,
)


class TestComputeScaleFactors:
    def test_powers_of_ten_are_identified_correctly(self):
        x = np.array([1e-5, 1.0, 1e3])
        factors = compute_scale_factors(x)
        np.testing.assert_allclose(factors, [1e-5, 1.0, 1e3])

    def test_values_within_an_order_of_magnitude_share_the_floor(self):
        """5e2 and 9e2 both belong to the 1e2 order of magnitude --
        np.floor(log10(x)) should give the same scale factor for both."""
        x = np.array([5e2, 9e2])
        factors = compute_scale_factors(x)
        np.testing.assert_allclose(factors, [1e2, 1e2])

    def test_negative_values_use_absolute_magnitude(self):
        x = np.array([-3e4, 3e4])
        factors = compute_scale_factors(x)
        np.testing.assert_allclose(factors, [1e4, 1e4])

    def test_zero_value_does_not_raise_or_produce_nan(self):
        """log10(0) is -inf; compute_scale_factors guards this with
        `np.where(np.abs(x) > 0, np.abs(x), 1e-12)` so a literal 0.0
        parameter still gets a finite (very small) scale factor rather
        than blowing up log10 or dividing by zero downstream."""
        x = np.array([0.0, 100.0])
        factors = compute_scale_factors(x)
        assert np.isfinite(factors).all()
        assert factors[0] == pytest.approx(1e-12)


class TestScaleUnscaleRoundTrip:
    def test_scale_then_unscale_recovers_original(self):
        x_physical = np.array([1e-6, 250.0, -3.5, 800.0])
        factors = compute_scale_factors(x_physical)
        x_scaled = scale(x_physical, factors)
        x_recovered = unscale(x_scaled, factors)
        np.testing.assert_allclose(x_recovered, x_physical)

    def test_scale_brings_values_to_order_1(self):
        """The whole point of scaling: wildly different native magnitudes
        (A ~ 1e-5 vs Ee ~ 1e2-1e3, per docs/methodology.md Sec 3.3) should
        all land at roughly the same order of magnitude after scaling."""
        x_physical = np.array([3.2e-5, 4.7e2, 8.1e-6, 9.9e3])
        factors = compute_scale_factors(x_physical)
        x_scaled = scale(x_physical, factors)
        # Every scaled value should be in [1, 10) in magnitude
        assert np.all(np.abs(x_scaled) >= 1.0)
        assert np.all(np.abs(x_scaled) < 10.0)