"""
Implicit midpoint + Newton-Raphson solver for the TLV ODE (Thesis Sec. 1.2.3).
"""
import numpy as np
import numpy.typing as npt

from creep_model.domain import CreepTest
from creep_model.modeling.tlv.parameters import TLVParameters
from creep_model.modeling.tlv.initial_condition import sigma_ep_0
from creep_model.modeling.tlv.residual import residual, residual_derivative


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
    max_iter: int = 50,
) -> float:
    """
    Solve R(sigma_ep_{n+1}) = 0 for a single time step via Eq. 1.11.
    Iterates until |R| < tol (MPa), per the thesis's stated convergence
    criterion.
    """
    sigma_ep_guess = sigma_ep_n  # warm start from the previous accepted value

    for _ in range(max_iter):
        R = residual(sigma_ep_guess, sigma_ep_n, sigma, T_n, T_next, t_n, dt, params)
        if abs(R) < tol:
            return sigma_ep_guess
        dR = residual_derivative(sigma_ep_guess, sigma_ep_n, sigma, T_n, T_next, t_n, dt, params)
        if dR == 0:
            raise SolverConvergenceError("Newton-Raphson stalled: zero residual derivative.")
        sigma_ep_guess = sigma_ep_guess - R / dR

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
