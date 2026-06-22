import numpy as np
import numpy.typing as npt
from scipy.optimize import curve_fit
from creep_model.modeling.base import BaseCreepModel
from scipy.optimize import minimize

# 1. THE PURE MATH FUNCTION
def findley_law(time: npt.NDArray[np.float64], eps0: float, m: float, n: float) -> npt.NDArray[np.float64]:
    """Pure math: Findley Viscoelastic Law."""
    return eps0 + m * np.power(time, n)

def modified_findley_law(time: npt.NDArray[np.float64], eps0: float, a: float, m: float, n: float) -> npt.NDArray[np.float64]:
    """
    Modified Findley Law with a Linear Term (Secondary Creep).
    """
    return eps0 + (a * time) + (m * np.power(time, n))

def global_creep_law(X_T: npt.NDArray[np.float64], A: float, B: float, n: float, m: float) -> npt.NDArray[np.float64]:
    """
    Global Model: Strain = (eps_coeff * Stress) + B * (Stress**n) * (Time**m)
    X_T is the TRANSPOSED input matrix of shape (2, N_samples).
    """
    time = X_T[0]    # Extracts all 2,529 time points
    stress = X_T[1]  # Extracts all 2,529 stress points
    
    return (A * stress) + (B * np.power(stress, n) * np.power(time, m))

# 2. THE MODEL CLASS
class LocalFindleyModel(BaseCreepModel):
    """Fits Findley law to a single CreepTest."""
    def __init__(self, p0: list[float] | None = None):
        super().__init__()
        # Set starting guess
        self.p0 = p0 or [0.0, 0.001, 0.3] 

    def fit(self, X: npt.NDArray[np.float64], y: npt.NDArray[np.float64]) -> "LocalFindleyModel":
        time_flat = X.flatten()
        
        # 1. Dynamic Initial Guesses (Help the optimizer out!)
        eps0_guess = y[0] # The first strain reading is a great guess for the intercept
        
        # Rough guess for m: (final_strain - initial_strain) / (final_time ** 0.3)
        delta_strain = y[-1] - y[0]
        m_guess = delta_strain / (time_flat[-1] ** 0.3) if time_flat[-1] > 0 else 0.001
        
        n_guess = 0.3 # Typical primary creep exponent
        
        p0_dynamic = [eps0_guess, m_guess, n_guess]
        
        # 2. Enforce Physical Bounds
        # Lower bounds: [eps0 > 0, m > 0, n > 0.01]
        # Upper bounds: [eps0 < infinity, m < infinity, n < 0.99]
        # Forcing n < 1 guarantees it will curve downwards (primary creep)
        physical_bounds = (
            [0.0, 0.0, 0.01], 
            [np.inf, np.inf, 0.99]
        )
        
        # 3. Fit with bounds
        popt, _ = curve_fit(
            findley_law, 
            time_flat, 
            y, 
            p0=p0_dynamic,
            bounds=physical_bounds,
            maxfev=10000 # Give it plenty of time to compute
        )
        
        self.fitted_params_ = popt
        return self
        
    def predict(self, X: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        # Run the Findley law with the fitted parameters
        if self.fitted_params_ is None:
            raise ValueError("Model must be fitted before prediction.")
        return findley_law(X.flatten(), *self.fitted_params_)
    

class QuantizedFindleyModel(BaseCreepModel):
    """Fits Findley law using a custom Epsilon-Insensitive Loss function."""
    
    def __init__(self, step_size: float = 0.0005):
        super().__init__()
        self.epsilon = step_size / 2.0  # The boundary of your red dashed lines (0.00025)

    def _custom_loss(self, params: list[float], time_flat: npt.NDArray, y_true: npt.NDArray) -> float:
        """
        Calculates how badly the model violates the sensor resolution bounds.
        """
        # Unpack params
        eps0, m, n = params
        
        # 1. Calculate prediction
        y_pred = findley_law(time_flat, eps0, m, n)
        
        # 2. Calculate the absolute residual
        abs_residual = np.abs(y_true - y_pred)
        
        # 3. Apply the Epsilon-Insensitive logic:
        # If the residual is less than epsilon (inside red lines), penalty is 0.
        # If the residual is greater than epsilon, we heavily punish the difference.
        violation = np.maximum(0, abs_residual - self.epsilon)
        
        # Square the violation so the optimizer has a smooth gradient to follow
        loss = np.sum(violation ** 2)
        
        # Optional: Add a tiny bit of MSE so the flat "valley" has a slight slope 
        # towards the exact center, preventing the Zero Gradient Problem.
        loss += 1e-6 * np.sum(abs_residual ** 2)
        
        return float(loss)

    def fit(self, X: npt.NDArray, y: npt.NDArray) -> "QuantizedFindleyModel":
        time_flat = X.flatten()
        
        # Use your dynamic guesses from before!
        eps0_guess = y[0]
        m_guess = (y[-1] - y[0]) / (time_flat[-1] ** 0.3) if time_flat[-1] > 0 else 0.001
        p0 = [eps0_guess, m_guess, 0.3]
        
        bounds = [(0.0, None), (0.0, None), (0.01, 0.99)] # Physical bounds
        
        # Run the minimizer on our custom loss function
        result = minimize(
            fun=self._custom_loss,
            x0=p0,
            args=(time_flat, y),
            bounds=bounds,
            method='L-BFGS-B' 
        )
        
        if not result.success:
            print(f"Warning: Optimizer struggled: {result.message}")
            
        self.fitted_params_ = result.x
        return self
        
    def predict(self, X: npt.NDArray) -> npt.NDArray:
        if not self.is_fitted:
            raise ValueError("Model not fitted.")
        return findley_law(X.flatten(), *self.fitted_params_)
    

import numpy as np
import numpy.typing as npt
from scipy.optimize import minimize
from typing import Callable, Optional

# Assuming BaseCreepModel and modified_findley_law are imported elsewhere

class LocalModifiedFindleyModel(BaseCreepModel):
    """Fits the 4-parameter modified Findley law using multi-start custom loss."""
    
    def __init__(self, n_starts: int = 10, loss_func: Optional[Callable] = None):
        """
        Args:
            n_starts: Number of random initializations to try.
            loss_func: A callable with signature (params, time, y_observed).
                       If None, defaults to the interval-censored (rounded) MLE loss.
        """
        super().__init__()
        self.n_starts = n_starts
        # Fall back to a default loss if none is supplied
        self.loss_func = loss_func if loss_func is not None else self._default_rounded_mle_loss

    def _default_rounded_mle_loss(self, params: npt.NDArray, time: npt.NDArray, y_obs: npt.NDArray) -> float:
        """Default interval-censored MLE loss for data rounded to the nearest 5e-4."""
        eps0, a, m, n = params
        
        # Calculate continuous model prediction
        # (Assuming modified_findley_law accepts time and the 4 parameters)
        y_pred = modified_findley_law(time, eps0, a, m, n)
        
        # Define rounding interval half-width (5e-4 / 2 = 2.5e-4)
        half_width = 2.5e-4
        
        # Internal noise parameter 's' (can be hardcoded or added to params if optimized)
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
        
        # 1. Parameter Boundary Setup (L-BFGS-B uses bounds as a list of tuples)
        # Structure: (min, max) for each parameter [eps0, a, m, n]
        bounds = [
            (0.0, None),       # eps0 >= 0
            (0.0, None),       # a >= 0
            (0.0, None),       # m >= 0
            (0.01, 0.99)       # 0.01 <= n <= 0.99
        ]
        
        # 2. Define Initial Guess Ranges for Multi-Start Generation
        eps0_low, eps0_high = 0.0, float(y_flat[0])
        
        avg_slope = (y_flat[-1] - y_flat[0]) / time_flat[-1] if time_flat[-1] > 0 else 0.0001
        a_low, a_high = 0.0, float(avg_slope)
        m_low, m_high = 0.0, float(avg_slope)
        n_low, n_high = 0.0, 1.0

        best_loss = np.inf
        best_params = None

        # 3. Multi-Start Optimization Loop
        for i in range(self.n_starts):
            # Generate random initial guesses within ranges
            eps0_guess = np.random.uniform(eps0_low, eps0_high)
            a_guess = np.random.uniform(a_low, a_high)
            m_guess = np.random.uniform(m_low, m_high)
            n_guess = np.random.uniform(n_low, n_high)
            p0 = [eps0_guess, a_guess, m_guess, n_guess]
            
            # Minimize using the custom loss function
            result = minimize(
                self.loss_func,
                x0=p0,
                args=(time_flat, y_flat),
                bounds=bounds,
                method='L-BFGS-B',
                options={'maxiter': 10000}
            )
            
            # Track the global minimum across all valid starts
            if result.success and result.fun < best_loss:
                best_loss = result.fun
                best_params = result.x

        # Fallback if every single optimization run fails entirely
        if best_params is None:
            raise RuntimeError("Optimization failed to converge on all multi-start attempts.")
            
        self.fitted_params_ = best_params
        return self

    def predict(self, X: npt.NDArray) -> npt.NDArray:
        if not hasattr(self, "fitted_params_"):
            raise ValueError("Model not fitted.")
        return modified_findley_law(X.flatten(), *self.fitted_params_)


class GlobalMLEModel(BaseCreepModel):
    def __init__(self, sensor_resolution: float = 0.01):
        super().__init__()
        self.resolution = sensor_resolution
        self.p0 = [1e-3, 1e-4, 2.0, 0.3] # [A, B, n, m]

    def _negative_log_likelihood(self, params: list[float], X: npt.NDArray[np.float64], y_true: npt.NDArray[np.float64]) -> float:
        """
        The custom loss function. Scipy will try to MINIMIZE this value.
        """
        A, B, n, m = params
        y_pred = global_creep_law(X, A, B, n, m)
        
        # -------------------------------------------------------------
        # TODO for you (Literature Review Integration):
        # Calculate the Negative Log-Likelihood (NLL) here.
        # Simple Gaussian MSE proxy: np.sum((y_pred - y_true)**2)
        # For quantization, you will write custom probability logic.
        # -------------------------------------------------------------
        nll_value = ... 
        
        return nll_value

    def fit(self, X: npt.NDArray[np.float64], y: npt.NDArray[np.float64]) -> "GlobalMLEModel":
        # Minimize the custom NLL
        result = minimize(
            fun=self._negative_log_likelihood,
            x0=self.p0,
            args=(X, y),
            method='Nelder-Mead', # Often better than L-BFGS-B for custom/staircase likelihoods
            options={'maxiter': 10000}
        )
        
        if not result.success:
            print(f"Warning: Optimizer failed to converge: {result.message}")
            
        self.fitted_params_ = result.x
        return self
    
class GlobalMSEModel(BaseCreepModel):
    """Fits all tests simultaneously using Least Squares (MSE)."""
    
    def fit(self, X: npt.NDArray, y: npt.NDArray) -> "GlobalMSEModel":
        # 1. Ensure inputs are properly flattened/shaped for curve_fit
        # If X is shape (N, 2), transposing it to (2, N) allows your function 
        # to cleanly unpack it via: stress, time = X
        X_input = X.T if X.ndim == 2 and X.shape[1] == 2 else X
        y_flat = y.flatten()
        
        # Initial Guesses [eps_coeff, B, n, m]
        p0 = [0.001, 1e-5, 1.0, 0.3]
        
        # Bounds to ensure physics holds (n > 0 for stress dependence, m in [0,1] for primary creep)
        bounds = (
            [0.0, 0.0, 0.01, 0.01],
            [np.inf, np.inf, 5.0, 0.99]
        )
        
        # TRANSPOSE X to match SciPy's expected shape (2, N)
        X_T = X.T 
        
        # 2. Use curve_fit to minimize the MSE between the model and observed data
        popt, _ = curve_fit(
            f=global_creep_law,
            xdata=X_T,
            ydata=y,
            p0=p0,
            bounds=bounds,
            maxfev=10000
        )
        
        self.fitted_params_ = popt
        return self

    def predict(self, X: npt.NDArray) -> npt.NDArray:
        if not self.is_fitted:
            raise ValueError("Model not fitted.")
        
        # Process X shape identically to the fitting stage
        X_input = X.T if X.ndim == 2 and X.shape[1] == 2 else X
        return global_creep_law(X_input, *self.fitted_params_)