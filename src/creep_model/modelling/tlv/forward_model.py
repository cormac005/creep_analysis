"""
sklearn-style wrapper around the TLV solver, conforming to BaseCreepModel.

NOTE: .fit() is intentionally NOT implemented here -- fitting requires
solving the ODE for every test in a print-quality group simultaneously (see
Eq. 1.12's group-level MSE), not a single (X, y) pair, so it doesn't fit
BaseCreepModel's per-instance fit(X, y) signature. Use
optimization.fit_pipeline.fit_group(tests, bounds) directly and construct
this class from its returned TLVParameters.
"""
import numpy.typing as npt

from creep_model.domain import CreepTest
from creep_model.modelling.tlv.parameters import TLVParameters
from creep_model.modelling.tlv.solver import solve_tlv
from creep_model.modelling.base import BaseCreepModel


class TLVCreepModel(BaseCreepModel):
    """Wraps a fitted TLVParameters set + solver as a BaseCreepModel, for
    consistency with viz/plots.py and the rest of the modeling/ API."""

    def __init__(self, params: TLVParameters | None = None):
        super().__init__()
        self._params = params
        if params is not None:
            self.fitted_params_ = params.to_array()

    def fit(self, X, y) -> "TLVCreepModel":
        raise NotImplementedError(
            "TLVCreepModel.fit() is not meaningful for a single (X, y) pair -- "
            "use modeling.optimization.fit_pipeline.fit_group(tests, bounds) "
            "and construct this class from its returned TLVParameters instead."
        )

    def predict_test(self, test: CreepTest) -> npt.NDArray:
        """Predict the full strain trace for a single CreepTest (preferred
        entry point -- needs time, temperature, and stress together, which a
        flat X matrix can't carry)."""
        if self._params is None:
            raise RuntimeError("TLVCreepModel has no fitted parameters.")
        return solve_tlv(test, self._params)

    def _predict(self, X) -> npt.NDArray:
        raise NotImplementedError(
            "TLVCreepModel predicts per-CreepTest, not from a flat X matrix -- "
            "use predict_test(test) instead."
        )
