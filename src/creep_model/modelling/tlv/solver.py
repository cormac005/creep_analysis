"""
Implicit midpoint + Newton-Raphson solver for the TLV ODE (Thesis Sec. 1.2.3).
"""
import numpy as np
import numpy.typing as npt

from creep_model.domain import CreepTest
from creep_model.modelling.tlv.parameters import TLVParameters
from creep_model.modelling.tlv.initial_conditions import sigma_ep_0, sigma_ep_0_from_measurement
from creep_model.modelling.tlv.residual import residual, residual_derivative


class SolverConvergenceError(RuntimeError):
    """Raised when Newton-Raphson fails to converge for a single time step.

    Deliberately a distinct exception type (not a bare RuntimeError) so the
    optimisation objective (fit_pipeline.py) can catch it specifically and
    penalise the candidate parameter set, without accidentally swallowing
    unrelated bugs.
    """


def _newton_raphson_step(
    sigma_ep_n: float,
    sigma: float,
    T_n: float,
    T_next: float,
    t_n: float,
    dt: float,
    params: TLVParameters,
    tol: float = 1e-8,
    max_iter: int = 100,
) -> float:
    """
    Solve R(sigma_ep_{n+1}) = 0 for a single time step via Eq. 1.11.

    NOTE: candidate parameter sets explored during DE's global search
    routinely produce numerically unstable iterates (overflow in the
    Norton-Hoff power term, or an overshoot into the unphysical
    sigma_ep_mid > sigma region). residual()/residual_derivative()
    correctly raise ValueError/ZeroDivisionError for these states -- that
    contract is relied on by their own unit tests and must not change.
    Here, at the solver level, those exceptions are translated into
    SolverConvergenceError, which fit_pipeline._de_objective already
    catches and penalises. Without this translation, a single bad DE
    candidate crashes the entire optimisation run (as seen when this
    wasn't caught: the ValueError propagated through the multiprocessing
    pool and killed differential_evolution entirely).
    """
    sigma_ep_guess = sigma_ep_n  # warm start from the previous accepted value

    for _ in range(max_iter):
        try:
            R = residual(sigma_ep_guess, sigma_ep_n, sigma, T_n, T_next, t_n, dt, params)
        except ValueError as e:
            raise SolverConvergenceError(
                f"Newton-Raphson stepped into an unphysical state: {e}"
            ) from e

        if abs(R) < tol:
            return sigma_ep_guess

        try:
            dR = residual_derivative(sigma_ep_guess, sigma_ep_n, sigma, T_n, T_next, t_n, dt, params)
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


def solve_tlv(
    test: CreepTest,
    params: TLVParameters,
    tol: float = 1e-8,
    use_measured_initial_condition: bool = True,
) -> npt.NDArray[np.float64]:
    """
    ...
    Args:
        ...
        use_measured_initial_condition: if True (default), sigma_ep(0) is
            calibrated from the first measured strain reading via
            sigma_ep_0_from_measurement rather than derived theoretically
            (Eq. 1.4). Set False to use the original theoretical IC.
    """
    time = test.time_series

    # interpolate_temperature() returns Celsius; at_temperature()/its
    # anchors (T_20_KELVIN, T_30_KELVIN) expect Kelvin. BUG FIX: this
    # conversion was previously missing entirely -- every at_temperature()
    # call below was silently extrapolating ~27x beyond the intended
    # 20-30C window. Converting once here fixes every downstream call.
    T = test.interpolate_temperature() + 273.15

    sigma = test.applied_stress_MPa

    sigma_ep = np.empty_like(time)
    if use_measured_initial_condition:
        sigma_ep[0] = sigma_ep_0_from_measurement(test.strain_series[0], T[0], params)
    else:
        sigma_ep[0] = sigma_ep_0(sigma, T[0], params)

    for i in range(len(time) - 1):
        dt = time[i + 1] - time[i]
        sigma_ep[i + 1] = _newton_raphson_step(
            sigma_ep_n=sigma_ep[i], sigma=sigma,
            T_n=T[i], T_next=T[i + 1],
            t_n=time[i], dt=dt, params=params, tol=tol,
        )

    p_at_T = params.at_temperature(T)
    strain = sigma_ep / p_at_T["Ee"]
    return strain