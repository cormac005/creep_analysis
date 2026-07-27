"""
Tests for creep_model.modelling.base.BaseCreepModel.

BaseCreepModel is abstract (fit() is @abstractmethod), so tests exercise it
through a minimal concrete subclass rather than instantiating it directly.
"""
import numpy as np
import pytest

from creep_model.modelling.base import BaseCreepModel


class _EchoModel(BaseCreepModel):
    """Minimal concrete subclass: 'fits' by storing a constant, 'predicts'
    by returning it broadcast to X's shape."""

    def fit(self, X, y):
        self.fitted_params_ = np.array([float(np.mean(y))])
        return self

    def _predict(self, X):
        return np.full(X.shape[0], self.fitted_params_[0])


class TestBaseCreepModel:
    def test_cannot_instantiate_directly(self):
        """fit() is abstract -- BaseCreepModel itself must not be instantiable."""
        with pytest.raises(TypeError):
            BaseCreepModel()

    def test_is_fitted_false_before_fit(self):
        model = _EchoModel()
        assert model.is_fitted is False

    def test_is_fitted_true_after_fit(self):
        model = _EchoModel()
        model.fit(np.zeros((3, 1)), np.array([1.0, 2.0, 3.0]))
        assert model.is_fitted is True

    def test_predict_before_fit_raises(self):
        model = _EchoModel()
        with pytest.raises(RuntimeError, match="must be fitted"):
            model.predict(np.zeros((3, 1)))

    def test_predict_delegates_to_underscore_predict_after_fit(self):
        model = _EchoModel()
        model.fit(np.zeros((3, 1)), np.array([1.0, 2.0, 3.0]))
        result = model.predict(np.zeros((3, 1)))
        np.testing.assert_allclose(result, [2.0, 2.0, 2.0])

    def test_fit_returns_self_for_chaining(self):
        model = _EchoModel()
        result = model.fit(np.zeros((2, 1)), np.array([1.0, 1.0]))
        assert result is model