import numpy as np
import numpy.typing as npt
from scipy.optimize import curve_fit, minimize
from creep_model.modelling.base import BaseCreepModel
from typing import Callable, Optional

def findley_law(time: npt.NDArray[np.float64], eps0: float, m: float, n: float) -> npt.NDArray[np.float64]:
    return eps0 + m * np.power(time, n)

def modified_findley_law(time: npt.NDArray[np.float64], eps0: float, a: float, m: float, n: float) -> npt.NDArray[np.float64]:
    return eps0 + (a * time) + (m * np.power(time, n))

def global_creep_law(X_T: npt.NDArray[np.float64], A: float, B: float, n: float, m: float) -> npt.NDArray[np.float64]:
    time = X_T[0]
    stress = X_T[1]
    return (A * stress) + (B * np.power(stress, n) * np.power(time, m))

class LocalFindleyModel(BaseCreepModel):
    def __init__(self, p0: list[float] | None = None):
        super().__init__()
        self.p0 = p0 or [0.0, 0.001, 0.3] 

    def fit(self, X: npt.NDArray[np.float64], y: npt.NDArray[np.float64]) -> "LocalFindleyModel":
        time_flat = X.flatten()
        eps0_guess = y[0]
        delta_strain = y[-1] - y[0]
        m_guess = delta_strain / (time_flat[-1] ** 0.3) if time_flat[-1] > 0 else 0.001
        p0_dynamic = [eps0_guess, m_guess, 0.3]
        physical_bounds = ([0.0, 0.0, 0.01], [np.inf, np.inf, 0.99])
        popt, _ = curve_fit(findley_law, time_flat, y, p0=p0_dynamic, bounds=physical_bounds, maxfev=10000)
        self.fitted_params_ = popt
        return self
        
    def _predict(self, X: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        return findley_law(X.flatten(), *self.fitted_params_)
    
    def predict(self, X: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction.")
        return self._predict(X)

class QuantizedFindleyModel(BaseCreepModel):
    def __init__(self, sensor_strain_resolution: float = 5e-4):
        super().__init__()
        self.epsilon = sensor_strain_resolution / 2.0

    def _custom_loss(self, params: list[float], time_flat: npt.NDArray, y_true: npt.NDArray) -> float:
        eps0, m, n = params
        y_pred = findley_law(time_flat, eps0, m, n)
        abs_residual = np.abs(y_true - y_pred)
        violation = np.maximum(0, abs_residual - self.epsilon)
        loss = np.sum(violation ** 2) + 1e-6 * np.sum(abs_residual ** 2)
        return float(loss)

    def fit(self, X: npt.NDArray, y: npt.NDArray) -> "QuantizedFindleyModel":
        time_flat = X.flatten()
        eps0_guess = y[0]
        m_guess = (y[-1] - y[0]) / (time_flat[-1] ** 0.3) if time_flat[-1] > 0 else 0.001
        p0 = [eps0_guess, m_guess, 0.3]
        bounds = [(0.0, None), (0.0, None), (0.01, 0.99)]
        result = minimize(self._custom_loss, x0=p0, args=(time_flat, y), bounds=bounds, method='L-BFGS-B')
        if not result.success:
            print(f"Warning: Optimizer struggled: {result.message}")
        self.fitted_params_ = result.x
        return self
        
    def _predict(self, X: npt.NDArray) -> npt.NDArray:
        return findley_law(X.flatten(), *self.fitted_params_)

    def predict(self, X: npt.NDArray) -> npt.NDArray:
        if not self.is_fitted:
            raise ValueError("Model not fitted.")
        return self._predict(X)

class LocalModifiedFindleyModel(BaseCreepModel):
    def __init__(self, n_starts: int = 10, loss_func: Optional[Callable] = None):
        super().__init__()
        self.n_starts = n_starts
        self.epsilon = 2.5e-4
        self.loss_func = loss_func if loss_func is not None else self._custom_loss

    def _custom_loss(self, params: list[float], time_flat: npt.NDArray, y_true: npt.NDArray) -> float:
        eps0, a, m, n = params
        y_pred = modified_findley_law(time_flat, eps0, a, m, n)
        abs_residual = np.abs(y_true - y_pred)
        violation = np.maximum(0, abs_residual - self.epsilon)
        loss = np.sum(violation ** 2) + 1e-6 * np.sum(abs_residual ** 2)
        return float(loss)

    def _default_rounded_mle_loss(self, params: npt.NDArray, time: npt.NDArray, y_obs: npt.NDArray) -> float:
        eps0, a, m, n = params
        y_pred = modified_findley_law(time, eps0, a, m, n)
        half_width = 2.5e-4
        s = 1e-4 
        from scipy.stats import norm
        lower_bound = (y_obs - half_width - y_pred) / s
        upper_bound = (y_obs + half_width - y_pred) / s
        prob = norm.cdf(upper_bound) - norm.cdf(lower_bound)
        prob = np.clip(prob, 1e-15, 1.0)
        return float(-np.sum(np.log(prob)))

    def fit(self, X: npt.NDArray, y: npt.NDArray) -> "LocalModifiedFindleyModel":
        time_flat = X.flatten()
        y_flat = y.flatten()
        bounds = [(0.0, None), (0.0, None), (0.0, None), (0.01, 0.99)]
        
        eps0_low, eps0_high = 0.0, float(y_flat[0])
        avg_slope = (y_flat[-1] - y_flat[0]) / time_flat[-1] if time_flat[-1] > 0 else 0.0001
        a_low, a_high = 0.0, float(avg_slope)
        m_low, m_high = 0.0, float(avg_slope)
        n_low, n_high = 0.0, 1.0

        best_loss = np.inf
        best_params = None

        for i in range(self.n_starts):
            eps0_guess = np.random.uniform(eps0_low, eps0_high)
            a_guess = np.random.uniform(a_low, a_high)
            m_guess = np.random.uniform(m_low, m_high)
            n_guess = np.random.uniform(n_low, n_high)
            p0 = [eps0_guess, a_guess, m_guess, n_guess]
            
            result = minimize(
                self.loss_func, x0=p0, args=(time_flat, y_flat),
                bounds=bounds, method='L-BFGS-B', options={'maxiter': 10000}
            )
            
            if result.success and result.fun < best_loss:
                best_loss = result.fun
                best_params = result.x

        if best_params is None:
            raise RuntimeError("Optimization failed to converge on all multi-start attempts.")
            
        self.fitted_params_ = best_params
        return self

    def _predict(self, X: npt.NDArray) -> npt.NDArray:
        return modified_findley_law(X.flatten(), *self.fitted_params_)

    def predict(self, X: npt.NDArray) -> npt.NDArray:
        if not self.is_fitted:
            raise ValueError("Model not fitted.")
        return self._predict(X)
    
class GlobalMSEModel(BaseCreepModel):
    def __init__(self):
        super().__init__()

    def _prepare_input(self, X: npt.NDArray) -> npt.NDArray:
        if X.ndim == 2 and X.shape[1] == 2:
            return X.T
        elif X.ndim == 2 and X.shape[0] == 2:
            return X
        raise ValueError(f"Unexpected X shape: {X.shape}")
    
    def fit(self, X: npt.NDArray, y: npt.NDArray) -> "GlobalMSEModel":
        X_T = self._prepare_input(X)
        p0 = [0.001, 1e-5, 1.0, 0.3]
        bounds = ([0.0, 0.0, 0.01, 0.01], [np.inf, np.inf, 5.0, 0.99])
        
        popt, _ = curve_fit(f=global_creep_law, xdata=X_T, ydata=y, p0=p0, bounds=bounds, maxfev=10000)
        
        self.fitted_params_ = popt
        return self

    def _predict(self, X: npt.NDArray) -> npt.NDArray:
        X_input = X.T if X.ndim == 2 and X.shape[1] == 2 else X
        return global_creep_law(X_input, *self.fitted_params_)

    def predict(self, X: npt.NDArray) -> npt.NDArray:
        if not self.is_fitted:
            raise ValueError("Model not fitted.")
        return self._predict(X)