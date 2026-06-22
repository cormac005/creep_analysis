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

    @abstractmethod
    def predict(self, X: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """
        Predicts strain based on input features.
        
        Args:
            X: 2D array of shape (N_samples, N_features).
            
        Returns:
            1D array of shape (N_samples,) representing predicted Strain.
        """
        pass
        
    @property
    def is_fitted(self) -> bool:
        return self.fitted_params_ is not None