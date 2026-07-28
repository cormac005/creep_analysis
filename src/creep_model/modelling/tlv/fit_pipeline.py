"""
Two-stage DE -> LM optimisation per print-quality group (Thesis Sec. 1.3.2).
"""
import numpy as np
import numpy.typing as npt
from scipy.optimize import differential_evolution, least_squares

from creep_model.domain import CreepTest
from creep_model.modelling.tlv.parameters import TLVParameters
from creep_model.modelling.tlv.solver import solve_tlv, SolverConvergenceError
from creep_model.modelling.optimisation.bounds import TLVBounds
from creep_model.modelling.optimisation.scaling import compute_scale_factors, scale, unscale

# Penalty returned to DE when a candidate parameter set fails to converge --
# large enough to be reliably rejected by the population-based search
# without being inf/nan (which some scipy internals handle poorly).
_NON_CONVERGENCE_PENALTY = 1e12


def _group_mse(params: TLVParameters, tests: list[CreepTest]) -> float:
    """Eq. 1.12, aggregated (mean) across every test in the group."""
    losses = []
    for test in tests:
        y_pred = solve_tlv(test, params)
        losses.append(np.mean((y_pred - test.strain_series) ** 2))
    return float(np.mean(losses))


def _de_objective(x_normalized: npt.NDArray[np.float64], tests: list[CreepTest], bounds: TLVBounds) -> float:
    params = bounds.denormalize(x_normalized)
    try:
        return _group_mse(params, tests)
    except SolverConvergenceError:
        return _NON_CONVERGENCE_PENALTY


def _lm_residuals(
    x_scaled: npt.NDArray[np.float64],
    tests: list[CreepTest],
    scale_factors: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """
    ...
    NOTE: earlier versions let SolverConvergenceError propagate uncaught
    here on the assumption DE's best candidate would always sit in a
    well-converging region. In practice, LM's finite-difference Jacobian
    evaluates points just outside x0 by construction, and can briefly
    probe just outside the stable region -- especially if DE itself
    didn't fully converge (see fit_group's "DE did not report success"
    warning). A single such point shouldn't crash the whole pipeline, so
    it's now penalised with a large-but-finite residual instead, with a
    printed warning. If this warning fires repeatedly for the same test
    across many LM iterations, that IS still a signal worth investigating
    (see original note) -- it's just no longer fatal on its own.
    """
    x_physical = unscale(x_scaled, scale_factors)
    params = TLVParameters.from_array(x_physical)

    all_residuals = []
    for test in tests:
        try:
            y_pred = solve_tlv(test, params)
        except SolverConvergenceError as e:
            print(f"Warning: solver did not converge during LM refinement "
                  f"for test {test.test_id} at a probed point ({e}); "
                  "penalising rather than crashing.")
            all_residuals.append(np.full_like(test.strain_series, 1e3))
            continue
        all_residuals.append(y_pred - test.strain_series)
    return np.concatenate(all_residuals)

def fit_group(
    tests: list[CreepTest],
    bounds: TLVBounds,
    de_kwargs: dict | None = None,
    lm_kwargs: dict | None = None,
) -> TLVParameters:
    """
    Fit TLVParameters to a single print-quality group via DE (global search)
    followed by LM (local refinement) -- Sec. 1.3.2.

    Args:
        tests: TRIMMED CreepTest objects (tertiary creep already removed via
               modeling.trimming.trim_tertiary), all of the same
               print_quality.
        bounds: TLVBounds for this group -- build via
                TLVBounds.from_group_data(tests) so Ee/Ev upper bounds
                reflect this group's actual data (Table 1.1).
        de_kwargs: passed through to scipy.optimize.differential_evolution,
                   e.g. {"workers": -1, "maxiter": 200, "seed": 42, "popsize": 20}.
                   `workers=-1` parallelises across population members --
                   worth using given each evaluation calls the sequential
                   Newton-Raphson solver across every test in the group.
        lm_kwargs: passed through to scipy.optimize.least_squares,
                   e.g. {"max_nfev": 5000}.

    Returns:
        Fitted TLVParameters for this group.
    """
    de_kwargs = dict(de_kwargs or {})
    lm_kwargs = dict(lm_kwargs or {})

    # --- Stage 1: Differential Evolution over normalised [0,1]^10 space ---
    de_result = differential_evolution(
        func=_de_objective,
        bounds=bounds.as_unit_bounds(),
        args=(tests, bounds),
        **de_kwargs,
    )
    if not de_result.success:
        print(f"Warning: DE did not report success: {de_result.message}")
    if de_result.fun >= _NON_CONVERGENCE_PENALTY:
        print("Warning: DE's best candidate never converged in the solver -- "
              "check bounds (Table 1.1) or tol before trusting this fit.")

    params_physical = bounds.denormalize(de_result.x)

    # --- Stage 2: order-of-magnitude scaling, then LM refinement ---
    x_physical = params_physical.to_array()
    scale_factors = compute_scale_factors(x_physical)
    x_scaled_0 = scale(x_physical, scale_factors)

    lm_method = lm_kwargs.pop("method", "lm")
    lm_result = least_squares(
        fun=_lm_residuals,
        x0=x_scaled_0,
        args=(tests, scale_factors),
        method=lm_method,
        **lm_kwargs,
    )
    if not lm_result.success:
        print(f"Warning: LM did not report success: {lm_result.message}")

    x_final_physical = unscale(lm_result.x, scale_factors)
    return TLVParameters.from_array(x_final_physical)
