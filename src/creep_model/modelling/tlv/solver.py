"""
Implicit midpoint + Newton-Raphson solver for the TLV ODE (Thesis Sec. 1.2.3).
"""
import numpy as np
import numpy.typing as npt

from creep_model.domain import CreepTest
from creep_model.modelling.tlv.parameters import TLVParameters
from creep_model.modelling.tlv.initial_conditions import sigma_ep_0
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
) -> npt.NDArray[np.float64]:
    """
    Integrate the TLV ODE (Eq. 1.2a) over a single CreepTest's time series,
    returning the predicted strain trace (Eq. 1.2b).

    Args:
        test: the CreepTest to predict on -- should already be TRIMMED of
              tertiary-creep data if being used for fitting (trimming is the
              caller's responsibility; see modeling/trimming.py).
        params: candidate or fitted TLVParameters.
        tol: Newton-Raphson residual tolerance in MPa.

    Returns:
        Predicted strain array, same length and time base as test.time_series.

    Raises:
        SolverConvergenceError if any time step fails to converge -- this is
        expected to happen for some candidate parameter sets during DE's
        global search; catch it at the call site (fit_pipeline.py) rather
        than here.
    """
    time = test.time_series
    # NOTE: interpolate_temperature() resamples temperature onto the STRAIN
    # time base at the RECORDED time points. The midpoint scheme needs T at
    # t_n, t_{n+1} (both on that grid -- fine) but does not need T at
    # arbitrary off-grid midpoints, since T_mid is computed here as the
    # AVERAGE of T_n and T_next (per Eq. 1.6 applied to T), not by a second,
    # independent interpolation call. Keep it that way for consistency with
    # how sigma_ep_mid is defined -- don't swap in np.interp(t_mid, ...) for
    # T_mid without also reconsidering whether Eq. 1.6 intends the same
    # midpoint convention for T.
    T = test.interpolate_temperature()
    sigma = test.applied_stress_MPa

    sigma_ep = np.empty_like(time)
    sigma_ep[0] = sigma_ep_0(sigma, T[0], params)

    for i in range(len(time) - 1):
        dt = time[i + 1] - time[i]
        sigma_ep[i + 1] = _newton_raphson_step(
            sigma_ep_n=sigma_ep[i],
            sigma=sigma,
            T_n=T[i],
            T_next=T[i + 1],
            t_n=time[i],
            dt=dt,
            params=params,
            tol=tol,
        )

    p_at_T = params.at_temperature(T)
    strain = sigma_ep / p_at_T["Ee"]  # Eq. 1.2b
    return strain
