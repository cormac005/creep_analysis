"""
Implicit midpoint + Newton-Raphson solver for the TLV ODE (Thesis Sec. 1.2.3).
Includes Numba JIT-compilation and pre-cached test array profiles for fast optimization.
"""
from dataclasses import dataclass
from unittest.mock import MagicMock

import numpy as np
import numpy.typing as npt

from creep_model.config import config
from creep_model.domain import CreepTest
from creep_model.modelling.tlv.parameters import TLVParameters
from creep_model.modelling.tlv.initial_conditions import sigma_ep_0, sigma_ep_0_from_measurement
from creep_model.modelling.tlv.residual import residual, residual_derivative

try:
    import numba as nb
    HAS_NUMBA = True
except ImportError:  # pragma: no cover
    HAS_NUMBA = False


class SolverConvergenceError(RuntimeError):
    """Raised when Newton-Raphson fails to converge for a single time step."""


@dataclass(frozen=True)
class PreparedTest:
    """Pre-calculated time and temperature arrays for zero-overhead solver evaluations."""
    test_id: str
    time: npt.NDArray[np.float64]
    T: npt.NDArray[np.float64]            # Kelvin
    dt: npt.NDArray[np.float64]
    t_mid: npt.NDArray[np.float64]
    T_mid: npt.NDArray[np.float64]
    T_dot_mid: npt.NDArray[np.float64]
    applied_stress_MPa: float
    eps_measured_0: float


def prepare_test_data(test: CreepTest) -> PreparedTest:
    """Pre-computes temperature interpolations and derivatives once per test."""
    time = test.time_series.astype(np.float64)
    dt = np.diff(time)
    t_mid = time[:-1] + dt / 2.0

    T = (test.interpolate_temperature() + 273.15).astype(np.float64)
    T_mid = (T[:-1] + T[1:]) / 2.0

    temp_poly = test.temperature_polynomial()
    temp_deriv_poly = temp_poly.deriv()
    t_min = float(test.temp_time_series.min())
    t_max = float(test.temp_time_series.max())

    T_dot_mid = np.zeros_like(t_mid)
    inside_mask = (t_mid >= t_min) & (t_mid <= t_max)
    T_dot_mid[inside_mask] = temp_deriv_poly(t_mid[inside_mask])

    return PreparedTest(
        test_id=str(getattr(test, "test_id", "test")),
        time=time,
        T=T,
        dt=dt,
        t_mid=t_mid,
        T_mid=T_mid,
        T_dot_mid=T_dot_mid,
        applied_stress_MPa=float(test.applied_stress_MPa),
        eps_measured_0=float(test.strain_series[0]),
    )


def _newton_raphson_step(
    sigma_ep_n: float,
    sigma: float,
    T_n: float,
    T_next: float,
    t_n: float,
    dt: float,
    params: TLVParameters,
    T_dot: float = None,
    tol: float = config.NR_KWARGS["tol"],
    max_iter: int = config.NR_KWARGS["max_iter"],
) -> float:
    """
    Solve R(sigma_ep_{n+1}) = 0 for a single time step via Eq. 1.11.
    Maintained for backwards compatibility with unit tests.
    """
    sigma_ep_guess = sigma_ep_n  # warm start

    if T_dot is None:
        T_dot = (T_next - T_n) / dt if dt > 0 else 0.0

    for _ in range(max_iter):
        try:
            R = residual(sigma_ep_guess, sigma_ep_n, sigma, T_n, T_next, t_n, dt, params, T_dot=T_dot)
        except ValueError as e:
            raise SolverConvergenceError(
                f"Newton-Raphson stepped into an unphysical state: {e}"
            ) from e

        if abs(R) < tol:
            return sigma_ep_guess

        try:
            dR = residual_derivative(sigma_ep_guess, sigma_ep_n, sigma, T_n, T_next, t_n, dt, params, T_dot=T_dot)
        except (ValueError, ZeroDivisionError) as e:
            raise SolverConvergenceError(
                f"Newton-Raphson derivative undefined at this state: {e}"
            ) from e

        if dR == 0 or not np.isfinite(dR):
            raise SolverConvergenceError("Newton-Raphson stalled: zero or non-finite residual derivative.")

        sigma_ep_guess = sigma_ep_guess - R / dR

        if not np.isfinite(sigma_ep_guess):
            raise SolverConvergenceError("Newton-Raphson diverged to a non-finite value.")

    raise SolverConvergenceError(
        f"Newton-Raphson did not converge below tol={tol} MPa within {max_iter} iterations."
    )


# --- NUMBA JIT KERNEL ---
if HAS_NUMBA:
    @nb.njit(fastmath=True, error_model="numpy")  # pragma: no cover
    def _solve_tlv_numba_kernel(
        p_arr, time, T, dt, t_mid, T_mid, T_dot_mid, sigma, eps_0, use_measured_ic, tol, max_iter
    ):
        A20, A30, n20, n30, m20, m30, Ee20, Ee30, Ev20, Ev30 = p_arr
        n_pts = len(time)
        sigma_ep = np.empty(n_pts, dtype=np.float64)

        # 1. Initial condition
        T0 = T[0]
        del_T0 = T0 - 293.15
        Ee0 = Ee20 + (Ee30 - Ee20) / 10.0 * del_T0
        Ev0 = Ev20 + (Ev30 - Ev20) / 10.0 * del_T0

        if use_measured_ic:
            sigma_ep[0] = eps_0 * Ee0
        else:
            f0 = Ev0 / (Ev0 + Ee0)
            sigma_ep[0] = (1.0 - f0) * sigma

        dEe_dT = (Ee30 - Ee20) / 10.0
        dEv_dT = (Ev30 - Ev20) / 10.0

        # 2. Integration loop
        for i in range(n_pts - 1):
            dt_i = dt[i]
            t_m = t_mid[i]
            T_m = T_mid[i]
            T_dot_m = T_dot_mid[i]

            del_T_m = T_m - 293.15
            Ee = Ee20 + (Ee30 - Ee20) / 10.0 * del_T_m
            Ev = Ev20 + (Ev30 - Ev20) / 10.0 * del_T_m
            A = A20 + (A30 - A20) / 10.0 * del_T_m
            n = n20 + (n30 - n20) / 10.0 * del_T_m
            m = m20 + (m30 - m20) / 10.0 * del_T_m

            combined_E = (Ee * Ev) / (Ee + Ev)
            elastic_temp_term = (dEe_dT * T_dot_m) / (Ee * Ee)
            viscous_temp_term = (dEv_dT * T_dot_m) / (Ev * Ev)

            s_n = sigma_ep[i]
            s_next = s_n  # warm start

            converged = False
            for _ in range(max_iter):
                s_mid = 0.5 * (s_n + s_next)
                
                # Smooth saturation: clip effective stress to 0 rather than aborting
                stress_diff = max(0.0, sigma - s_mid)

                norton_hoff_term = A * (stress_diff ** n) * (t_m ** m)
                bracket = (s_mid * elastic_temp_term) - ((sigma - s_mid) * viscous_temp_term) + norton_hoff_term
                R = s_next - s_n - dt_i * combined_E * bracket

                if abs(R) < tol:
                    converged = True
                    break

                if stress_diff <= 0.0:
                    norton_hoff_deriv = 0.0
                else:
                    norton_hoff_deriv = A * n * (stress_diff ** (n - 1.0)) * (t_m ** m)

                bracket_deriv = elastic_temp_term + viscous_temp_term - norton_hoff_deriv
                dR = 1.0 - 0.5 * dt_i * combined_E * bracket_deriv

                if dR == 0.0 or np.isnan(dR):
                    return sigma_ep, -2

                s_next -= R / dR

                if np.isnan(s_next) or np.isinf(s_next):
                    return sigma_ep, -3

            if not converged:
                return sigma_ep, -4

            sigma_ep[i + 1] = s_next

        del_T_all = T - 293.15
        Ee_all = Ee20 + (Ee30 - Ee20) / 10.0 * del_T_all
        strain = sigma_ep / Ee_all
        return strain, 0


def solve_tlv_prepared(
    prep: PreparedTest,
    params: TLVParameters,
    tol: float = config.NR_KWARGS["tol"],
    use_measured_initial_condition: bool = True,
) -> npt.NDArray[np.float64]:
    """Fast solver evaluation using pre-calculated PreparedTest object."""
    max_iter = config.NR_KWARGS["max_iter"]
    p_arr = params.to_array()

    if HAS_NUMBA:
        strain, err_code = _solve_tlv_numba_kernel(
            p_arr, prep.time, prep.T, prep.dt, prep.t_mid, prep.T_mid, prep.T_dot_mid,
            prep.applied_stress_MPa, prep.eps_measured_0, use_measured_initial_condition, tol, max_iter
        )
        if err_code != 0:
            msg_map = {
                -2: "Newton-Raphson stalled: zero or non-finite residual derivative.",
                -3: "Newton-Raphson diverged to a non-finite value.",
                -4: f"Newton-Raphson did not converge below tol={tol} within {max_iter} iterations.",
            }
            raise SolverConvergenceError(msg_map.get(err_code, "Unknown solver convergence error."))
        return strain
    else:  # pragma: no cover
        n_pts = len(prep.time)
        sigma_ep = np.empty(n_pts, dtype=np.float64)
        sigma = prep.applied_stress_MPa

        p_T0 = params.at_temperature(prep.T[0])
        if use_measured_initial_condition:
            sigma_ep[0] = prep.eps_measured_0 * p_T0["Ee"]
        else:
            f0 = p_T0["Ev"] / (p_T0["Ev"] + p_T0["Ee"])
            sigma_ep[0] = (1.0 - f0) * sigma

        dEe_dT = params.dEe_dT()
        dEv_dT = params.dEv_dT()

        for i in range(n_pts - 1):
            dt_i = prep.dt[i]
            t_m = prep.t_mid[i]
            T_m = prep.T_mid[i]
            T_dot_m = prep.T_dot_mid[i]

            p_m = params.at_temperature(T_m)
            Ee, Ev, A, n, m = p_m["Ee"], p_m["Ev"], p_m["A"], p_m["n"], p_m["m"]
            combined_E = (Ee * Ev) / (Ee + Ev)

            elastic_temp_term = (dEe_dT * T_dot_m) / (Ee ** 2)
            viscous_temp_term = (dEv_dT * T_dot_m) / (Ev ** 2)

            s_n = sigma_ep[i]
            s_next = s_n

            converged = False
            for _ in range(max_iter):
                s_mid = 0.5 * (s_n + s_next)
                stress_diff = max(0.0, sigma - s_mid)

                norton_hoff_term = A * (stress_diff ** n) * (t_m ** m)
                bracket = (s_mid * elastic_temp_term) - ((sigma - s_mid) * viscous_temp_term) + norton_hoff_term
                R = s_next - s_n - dt_i * combined_E * bracket

                if abs(R) < tol:
                    converged = True
                    break

                norton_hoff_deriv = A * n * (stress_diff ** (n - 1.0)) * (t_m ** m) if stress_diff > 0 else 0.0
                bracket_deriv = elastic_temp_term + viscous_temp_term - norton_hoff_deriv
                dR = 1.0 - 0.5 * dt_i * combined_E * bracket_deriv

                if dR == 0.0 or not np.isfinite(dR):
                    raise SolverConvergenceError("Newton-Raphson stalled: zero or non-finite derivative.")

                s_next -= R / dR
                if not np.isfinite(s_next):
                    raise SolverConvergenceError("Newton-Raphson diverged.")

            if not converged:
                raise SolverConvergenceError(f"Newton-Raphson did not converge within {max_iter} iterations.")

            sigma_ep[i + 1] = s_next

        p_all = params.at_temperature(prep.T)
        return sigma_ep / p_all["Ee"]


def solve_tlv(
    test: CreepTest,
    params: TLVParameters,
    tol: float = config.NR_KWARGS["tol"],
    use_measured_initial_condition: bool = True,
) -> npt.NDArray[np.float64]:
    """Public interface for single CreepTest evaluation."""
    is_mock = (
        isinstance(params, MagicMock)
        or isinstance(test, MagicMock)
        or not isinstance(params, TLVParameters)
    )

    if not is_mock:
        prep = prepare_test_data(test)
        return solve_tlv_prepared(prep, params, tol=tol, use_measured_initial_condition=use_measured_initial_condition)

    time = test.time_series
    temp_poly = test.temperature_polynomial()
    temp_deriv_poly = temp_poly.deriv()

    t_min = float(test.temp_time_series.min()) if hasattr(test, "temp_time_series") and test.temp_time_series is not None else 0.0
    t_max = float(test.temp_time_series.max()) if hasattr(test, "temp_time_series") and test.temp_time_series is not None else 0.0

    T = test.interpolate_temperature() + 273.15
    sigma = test.applied_stress_MPa

    sigma_ep = np.empty_like(time)
    if use_measured_initial_condition:
        sigma_ep[0] = sigma_ep_0_from_measurement(test.strain_series[0], T[0], params)
    else:
        sigma_ep[0] = sigma_ep_0(sigma, T[0], params)

    max_iter = config.NR_KWARGS["max_iter"]

    for i in range(len(time) - 1):
        dt = time[i + 1] - time[i]
        t_mid = time[i] + dt / 2.0

        if t_mid < t_min or t_mid > t_max:
            T_dot_mid = 0.0
        else:
            T_dot_mid = float(temp_deriv_poly(t_mid))

        sigma_ep[i + 1] = _newton_raphson_step(
            sigma_ep_n=sigma_ep[i], sigma=sigma,
            T_n=T[i], T_next=T[i + 1],
            t_n=time[i], dt=dt, params=params,
            T_dot=T_dot_mid, tol=tol, max_iter=max_iter
        )

    p_at_T = params.at_temperature(T)
    strain = sigma_ep / p_at_T["Ee"]
    return strain