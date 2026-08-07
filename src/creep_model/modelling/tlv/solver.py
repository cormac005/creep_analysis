"""
Implicit midpoint + Newton-Raphson solver for the TLV ODE (Thesis Sec. 1.2.3).
Includes Numba JIT-compilation and pre-cached test array profiles for fast optimization.
"""
from dataclasses import dataclass

import numba as nb
import numpy as np
import numpy.typing as npt

from creep_model.config import config
from creep_model.domain import CreepTest
from creep_model.modelling.tlv.parameters import TLVParameters, T_20_KELVIN, T_30_KELVIN


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


# --- NUMBA JIT KERNEL ---
@nb.njit(fastmath=True, error_model="numpy")
def _solve_tlv_numba_kernel(
    p_arr,
    time,
    T,
    dt,
    t_mid,
    T_mid,
    T_dot_mid,
    sigma,
    eps_0,
    use_measured_ic,
    tol,
    max_iter,
    t20_k=T_20_KELVIN,
    t30_k=T_30_KELVIN,
):
    A20, A30, n20, n30, m20, m30, Ee20, Ee30, Ev20, Ev30 = p_arr
    delta_T_ref = t30_k - t20_k
    n_pts = len(time)
    sigma_ep = np.empty(n_pts, dtype=np.float64)

    # 1. Initial condition
    T0 = T[0]
    del_T0 = T0 - t20_k
    Ee0 = Ee20 + (Ee30 - Ee20) / delta_T_ref * del_T0
    Ev0 = Ev20 + (Ev30 - Ev20) / delta_T_ref * del_T0

    if use_measured_ic:
        sigma_ep[0] = eps_0 * Ee0
    else:
        f0 = Ev0 / (Ev0 + Ee0)
        sigma_ep[0] = (1.0 - f0) * sigma

    dEe_dT = (Ee30 - Ee20) / delta_T_ref
    dEv_dT = (Ev30 - Ev20) / delta_T_ref

    # 2. Integration loop
    for i in range(n_pts - 1):
        dt_i = dt[i]
        t_m = t_mid[i]
        T_m = T_mid[i]
        T_dot_m = T_dot_mid[i]

        del_T_m = T_m - t20_k
        Ee = Ee20 + (Ee30 - Ee20) / delta_T_ref * del_T_m
        Ev = Ev20 + (Ev30 - Ev20) / delta_T_ref * del_T_m
        A = A20 + (A30 - A20) / delta_T_ref * del_T_m
        n = n20 + (n30 - n20) / delta_T_ref * del_T_m
        m = m20 + (m30 - m20) / delta_T_ref * del_T_m

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

    del_T_all = T - t20_k
    Ee_all = Ee20 + (Ee30 - Ee20) / delta_T_ref * del_T_all
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

    strain, err_code = _solve_tlv_numba_kernel(
        p_arr,
        prep.time,
        prep.T,
        prep.dt,
        prep.t_mid,
        prep.T_mid,
        prep.T_dot_mid,
        prep.applied_stress_MPa,
        prep.eps_measured_0,
        use_measured_initial_condition,
        tol,
        max_iter,
        T_20_KELVIN,
        T_30_KELVIN,
    )
    if err_code != 0:
        msg_map = {
            -2: "Newton-Raphson stalled: zero or non-finite residual derivative.",
            -3: "Newton-Raphson diverged to a non-finite value.",
            -4: f"Newton-Raphson did not converge below tol={tol} within {max_iter} iterations.",
        }
        raise SolverConvergenceError(msg_map.get(err_code, "Unknown solver convergence error."))
    return strain


def solve_tlv(
    test: CreepTest,
    params: TLVParameters,
    tol: float = config.NR_KWARGS["tol"],
    use_measured_initial_condition: bool = True,
) -> npt.NDArray[np.float64]:
    """Public interface for single CreepTest evaluation."""
    prep = prepare_test_data(test)
    return solve_tlv_prepared(
        prep, params, tol=tol, use_measured_initial_condition=use_measured_initial_condition
    )