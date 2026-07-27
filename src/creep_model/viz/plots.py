import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt

from creep_model.domain import CreepTest
from creep_model.modelling.base import BaseCreepModel
from creep_model.modelling.assembler import DataAssembler

def plot_local_fit(test: CreepTest, model: BaseCreepModel, save_path: str | None = None) -> None:
    """
    Generates a publication-ready 2-panel plot: 
    Top panel: Raw Data vs Model Fit
    Bottom panel: Residuals
    """
    if not model.is_fitted:
        raise ValueError("Model must be fitted before plotting.")
    
    # Find smallest jump in strain to set y-limits for residuals
    strain_diff = np.diff(test.strain_series)
    smallest_jump = np.min(strain_diff[strain_diff > 0]) if np.any(strain_diff > 0) else 1e-6

    # 1. Extract and Predict
    X, y_true = DataAssembler.get_local_data(test)
    y_pred = model.predict(X)
    time_flat = X.flatten() # For plotting on the x-axis
    residuals = y_true - y_pred

    # 2. Set up a Matplotlib figure with two subplots sharing the x-axis
    fig, (ax_main, ax_res) = plt.subplots(
        nrows=2, 
        ncols=1, 
        figsize=(8, 6), 
        gridspec_kw={'height_ratios': [3, 1]}, 
        sharex=True
    )

    # Plot raw data and model fit on the main axis
    ax_main.scatter(time_flat, y_true, color='blue', label='Raw Data', alpha=0.6)
    ax_main.plot(time_flat, y_pred, color='red', label='Model Fit', linewidth=2)
    ax_main.set_ylabel('Strain')
    ax_main.set_xlabel('Time (s)')
    ax_main.set_title(f'Creep Test: {test.test_id}')
    #ax_main.legend()
    ax_main.grid(True)
    plt.tight_layout()

    # Plot residuals on the residual axis
    ax_res.scatter(time_flat, residuals, color='green', alpha=0.6)
    ax_res.axhline(0, color='black', linestyle='--')
    ax_res.set_ylabel('Residuals')
    ax_res.set_xlabel('Time (s)')
    ax_res.set_title('Residuals')
    #ax_res.legend()
    ax_res.axhline(smallest_jump/2, color='red', linestyle='--')
    ax_res.axhline(-smallest_jump/2, color='red', linestyle='--')
    plt.tight_layout()

    # 3. Output handling
    if save_path:
        plt.savefig(save_path, dpi=300) # dpi=300 is standard for publications
    else:
        plt.show()
        
    plt.close(fig) # Always close the figure to free up memory