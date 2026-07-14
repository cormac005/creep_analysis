from abc import ABC, abstractmethod
import numpy as np
import numpy.typing as npt

class BaseCreepModel(ABC):
    """
    Abstract Base Class for all creep models.
    Enforces a standard API: fit(X, y) and predict(X).
    """
    
    def __init__(self) -> None:
        # This will store the optimized parameters after fitting
        self.fitted_params_: npt.NDArray[np.float64] | None = None

    def predict(self, X: npt.NDArray) -> npt.NDArray:
        """Template method — enforces fit check, delegates to _predict."""
        if not self.is_fitted:
            raise RuntimeError(
                f"{type(self).__name__} must be fitted before prediction. Call .fit(X, y) first."
            )
        return self._predict(X)

    @abstractmethod
    def fit(self, X: npt.NDArray[np.float64], y: npt.NDArray[np.float64]) -> "BaseCreepModel":
        """
        Fits the model parameters to the data.
        
        Args:
            X: 2D array of shape (N_samples, N_features). 
               e.g., Columns for [Time, Temperature, Applied Stress]
            y: 1D array of shape (N_samples,) representing Strain.
            
        Returns:
            self (Allows for method chaining like model.fit(X,y).predict(X))
        """
        pass

    # @abstractmethod
    # def _predict(self, X: npt.NDArray) -> npt.NDArray:
    #     """Subclasses implement this instead of predict directly."""
    #     pass
        
    @property
    def is_fitted(self) -> bool:
        return self.fitted_params_ is not None