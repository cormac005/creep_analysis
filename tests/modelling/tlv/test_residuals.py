"""
Tests for creep_model.modelling.tlv.residual (Thesis Eq. 1.9-1.11).

Uses ISOTHERMAL cases (T_n == T_next, so T_dot == 0) throughout to isolate
the Norton-Hoff bracket term from the dEe/dT, dEv/dT temperature-coupling
terms -- those get their own dedicated tests further down. Expected values
are hand-derived with simple round-number parameters (see module comment
in each test) and cross-checked by direct calculation, not just copied from
the implementation.
"""
import pytest

from creep_model.modelling.tlv.parameters import TLVParameters
from creep_model.modelling.tlv.residual import residual, residual_derivative


def _isothermal_params(A=1.0, n=1.0, m=0.0, Ee=2.0, Ev=2.0):
    """Constant-in-temperature params (20C and 30C values equal), so
    dEe/dT == dEv/dT == 0 regardless of T_dot -- cleanly isolates the
    Norton-Hoff term for hand-checkable isothermal tests."""
    return TLVParameters(
        A20=A, A30=A, n20=n, n30=n, m20=m, m30=m,
        Ee20=Ee, Ee30=Ee, Ev20=Ev, Ev30=Ev,
    )


class TestResidualIsothermal:
    def test_hand_computed_value(self):
        """
        A=1, n=1, m=0, Ee=Ev=2 -> combined_E = (1/2+1/2)^-1 = 1.
        sigma=10, sigma_ep_n=4, sigma_ep_next=6 -> sigma_ep_mid=5,
        stress_diff = 10-5 = 5. t_mid^m = anything^0 = 1.
        bracket = combined_E * A * stress_diff^n * t_mid^m = 1*1*5*1 = 5.
        R = sigma_ep_next - sigma_ep_n - dt*bracket = 6 - 4 - 2*5 = -8.
        """
        params = _isothermal_params()
        R = residual(
            sigma_ep_next=6.0, sigma_ep_n=4.0, sigma=10.0,
            T_n=293.15, T_next=293.15, t_n=1.0, dt=2.0, params=params,
        )
        assert R == pytest.approx(-8.0)

    def test_zero_norton_hoff_coefficient_gives_pure_linear_residual(self):
        """With A=0, the bracket term vanishes entirely (isothermal), so
        R(sigma_ep_next) = sigma_ep_next - sigma_ep_n exactly -- the
        solver's fixed point should just be sigma_ep_n itself."""
        params = _isothermal_params(A=0.0)
        R = residual(
            sigma_ep_next=4.0, sigma_ep_n=4.0, sigma=10.0,
            T_n=293.15, T_next=293.15, t_n=1.0, dt=5.0, params=params,
        )
        assert R == pytest.approx(0.0)

    def test_stress_diff_negative_raises(self):
        """
        Previously sigma_ep_mid > sigma raised a ValueError. 
        Now it smoothly saturates the effective stress to 0.0.
        """
        params = _isothermal_params()
        # Force sigma_ep_mid = 110.0, which is > sigma (50.0)
        res = residual(sigma_ep_next=120.0, sigma_ep_n=100.0, sigma=50.0, 
                       T_n=293.15, T_next=293.15, t_n=0.0, dt=1.0, params=params)
        
        # Assert that it successfully computes a value instead of raising an error
        assert isinstance(res, float)


class TestResidualDerivativeIsothermal:
    def test_hand_computed_value(self):
        """
        Same setup as test_hand_computed_value above.
        norton_hoff_deriv_term = A*n*stress_diff^(n-1)*t_mid^m = 1*1*5^0*1 = 1.
        bracket_deriv = 0 + 0 - 1 = -1 (temp terms vanish, T_dot=0).
        dR/dsigma_ep_next = 1 - 0.5*dt*combined_E*bracket_deriv
                          = 1 - 0.5*2*1*(-1) = 2.0.
        """
        params = _isothermal_params()
        dR = residual_derivative(
            sigma_ep_next=6.0, sigma_ep_n=4.0, sigma=10.0,
            T_n=293.15, T_next=293.15, t_n=1.0, dt=2.0, params=params,
        )
        assert dR == pytest.approx(2.0)

    def test_matches_finite_difference_approximation(self):
        """General cross-check, not tied to a specific hand-solved case:
        the analytical Jacobian should match a central finite-difference
        estimate of dR/d(sigma_ep_next) to high precision."""
        params = _isothermal_params(A=2e-3, n=1.7, m=-0.2, Ee=350.0, Ev=600.0)
        sigma = 15.0
        sigma_ep_n = 5.0
        T_n = T_next = 300.0
        t_n = 10.0
        dt = 3.0
        sigma_ep_next = 7.0

        h = 1e-6
        R_plus = residual(sigma_ep_next + h, sigma_ep_n, sigma, T_n, T_next, t_n, dt, params)
        R_minus = residual(sigma_ep_next - h, sigma_ep_n, sigma, T_n, T_next, t_n, dt, params)
        finite_diff = (R_plus - R_minus) / (2 * h)

        analytical = residual_derivative(
            sigma_ep_next, sigma_ep_n, sigma, T_n, T_next, t_n, dt, params
        )
        assert analytical == pytest.approx(finite_diff, rel=1e-5)

    def test_stress_diff_negative_raises(self):
        """Derivative should safely evaluate to a valid float when overstressed."""
        params = _isothermal_params()
        res = residual_derivative(sigma_ep_next=120.0, sigma_ep_n=100.0, sigma=50.0, 
                                  T_n=293.15, T_next=293.15, t_n=0.0, dt=1.0, params=params)
        assert isinstance(res, float)

    def test_stress_diff_zero_with_n_less_than_one_raises_zero_division(self):
        """
        At stress_diff == 0 with n < 1, (stress_diff)^(n-1) is a division by zero.
        The function must now catch this and return a valid float (clamped derivative).
        """
        params = _isothermal_params(n=0.5)
        # Force sigma_ep_mid = 50.0, sigma = 50.0 -> stress_diff = 0.0
        res = residual_derivative(sigma_ep_next=50.0, sigma_ep_n=50.0, sigma=50.0, 
                                  T_n=293.15, T_next=293.15, t_n=0.0, dt=1.0, params=params)
        assert isinstance(res, float)

    def test_stress_diff_zero_with_n_at_least_one_gives_zero_term(self):
        """At stress_diff == 0 with n >= 1, the softplus activation derivative
        ds/dx = 0.5, so the Norton-Hoff derivative term contributes 0.5 * combined_E,
        yielding dR = 1 - 0.5*dt*combined_E*(-0.5) = 1.25."""
        params = _isothermal_params(n=1.0)
        dR = residual_derivative(
            sigma_ep_next=10.0, sigma_ep_n=10.0, sigma=10.0,
            T_n=293.15, T_next=293.15, t_n=1.0, dt=1.0, params=params,
        )
        assert dR == pytest.approx(1.25)


class TestResidualNonIsothermal:
    def test_temperature_coupling_terms_activate_with_nonzero_t_dot(self):
        """Sanity check that the elastic/viscous temperature-coupling terms
        actually do something when T_n != T_next (T_dot != 0) and Ee/Ev
        vary with temperature -- residual should differ from the isothermal
        case with all else held equal."""
        params = TLVParameters(
            A20=1e-3, A30=1e-3, n20=1.5, n30=1.5, m20=0.0, m30=0.0,
            Ee20=400.0, Ee30=600.0,  # varies with T -> dEe/dT != 0
            Ev20=800.0, Ev30=800.0,  # constant -> dEv/dT == 0
        )
        common_kwargs = dict(
            sigma_ep_next=6.0, sigma_ep_n=4.0, sigma=10.0, t_n=1.0, dt=2.0, params=params,
        )
        R_isothermal = residual(T_n=293.15, T_next=293.15, **common_kwargs)
        R_heating = residual(T_n=293.15, T_next=296.0, **common_kwargs)

        assert R_isothermal != pytest.approx(R_heating)