"""
Two-stage DE -> LM optimisation per print-quality group (Thesis Sec. 1.3.2).
"""
import numpy as np
import numpy.typing as npt
from scipy.optimize import differential_evolution, least_squares

from creep_model.config import config
from creep_model.domain import CreepTest
from creep_model.modelling.tlv.parameters import TLVParameters
from creep_model.modelling.tlv.solver import (
    PreparedTest, prepare_test_data, solve_tlv, solve_tlv_prepared, SolverConvergenceError
)
from creep_model.modelling.optimisation.bounds import TLVBounds
from creep_model.modelling.optimisation.scaling import compute_scale_factors, scale, unscale

_NON_CONVERGENCE_PENALTY = config.NON_CONVERGENCE_PENALTY


def _group_mse_prepared(params: TLVParameters, prep_tests: list[tuple[PreparedTest, npt.NDArray[np.float64]]]) -> float:
    """Eq. 1.12, evaluated using pre-calculated PreparedTest array profiles."""
    losses = []
    for prep, strain_series in prep_tests:
        y_pred = solve_tlv_prepared(prep, params)
        losses.append(np.mean((y_pred - strain_series) ** 2))
    return float(np.mean(losses))


def _group_mse(params: TLVParameters, tests: list[CreepTest]) -> float:
    """
    Eq. 1.12, aggregated (mean) across every test in the group.
    Maintained for API backwards compatibility with unit tests.
    """
    losses = []
    for test in tests:
        y_pred = solve_tlv(test, params)
        losses.append(np.mean((y_pred - test.strain_series) ** 2))
    return float(np.mean(losses))


def _de_objective(
    x_normalized: npt.NDArray[np.float64],
    tests: list,
    bounds: TLVBounds
) -> float:
    params = bounds.denormalize(x_normalized)
    try:
        if tests and isinstance(tests[0], tuple):
            return _group_mse_prepared(params, tests)
        return _group_mse(params, tests)
    except SolverConvergenceError:
        return _NON_CONVERGENCE_PENALTY


def _lm_residuals(
    x_scaled: npt.NDArray[np.float64],
    tests: list,
    scale_factors: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    x_physical = unscale(x_scaled, scale_factors)
    params = TLVParameters.from_array(x_physical)

    # LM minimizes sum of squared residuals. 
    # To match the DE scalar MSE penalty, we take the square root of the penalty for the raw residual array.
    residual_penalty = np.sqrt(_NON_CONVERGENCE_PENALTY)

    all_residuals = []
    for item in tests:
        if isinstance(item, tuple):
            prep, strain_series = item
            test_id = getattr(prep, "test_id", "test")
            y_pred_func = lambda: solve_tlv_prepared(prep, params)
        else:
            strain_series = item.strain_series
            test_id = item.test_id
            y_pred_func = lambda: solve_tlv(item, params)

        try:
            y_pred = y_pred_func()
        except SolverConvergenceError:
            # Apply the consistent penalty to this test's residual block
            all_residuals.append(np.full_like(strain_series, residual_penalty))
            continue
            
        all_residuals.append(y_pred - strain_series)
        
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
    """
    merged_de_kwargs = {**config.DE_KWARGS, **(de_kwargs or {})}
    merged_lm_kwargs = {**config.LM_KWARGS, **(lm_kwargs or {})}

    prep_tests = [(prepare_test_data(t), t.strain_series.astype(np.float64)) for t in tests]

    # --- Stage 1: Differential Evolution over normalised [0,1]^10 space ---
    de_result = differential_evolution(
        func=_de_objective,
        bounds=bounds.as_unit_bounds(),
        args=(prep_tests, bounds),
        **merged_de_kwargs,
    )
    if not de_result.success:
        print(f"Warning: DE did not report success: {de_result.message}")
    if de_result.fun >= _NON_CONVERGENCE_PENALTY:
        print("Warning: DE's best candidate never converged in the solver -- check bounds or tol.")

    params_physical = bounds.denormalize(de_result.x)

    # --- Stage 2: order-of-magnitude scaling, then LM refinement ---
    x_physical = params_physical.to_array()
    scale_factors = compute_scale_factors(x_physical)
    x_scaled_0 = scale(x_physical, scale_factors)

    lm_method = merged_lm_kwargs.pop("method", "lm")
    lm_bounds = (0.0, np.inf) if lm_method == "trf" else (-np.inf, np.inf)

    lm_result = least_squares(
        fun=_lm_residuals,
        x0=x_scaled_0,
        args=(prep_tests, scale_factors),
        bounds=lm_bounds,
        method=lm_method,
        **merged_lm_kwargs,
    )
    if not lm_result.success:
        print(f"Warning: Local refinement solver did not report success: {lm_result.message}")

    x_final_physical = unscale(lm_result.x, scale_factors)
    return TLVParameters.from_array(x_final_physical)