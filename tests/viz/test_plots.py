import pytest
import numpy as np
from unittest.mock import MagicMock, patch
from creep_model.viz.plots import plot_local_fit
from creep_model.modelling.base import BaseCreepModel

@patch("creep_model.viz.plots.plt")
@patch("creep_model.viz.plots.DataAssembler")
def test_plot_local_fit(mock_da, mock_plt):
    # Setup test and model
    mock_test = MagicMock()
    mock_test.test_id = "T1"
    # Provide strain series to compute strain_diff natively
    mock_test.strain_series = np.array([0.0, 0.001, 0.002])
    
    mock_model = MagicMock(spec=BaseCreepModel)
    mock_model.is_fitted = True
    
    # Setup DataAssembler mock shape
    mock_X = np.array([[0], [1], [2]])
    mock_y_true = np.array([0.0, 0.001, 0.002])
    mock_da.get_local_data.return_value = (mock_X, mock_y_true)
    
    # Setup Model predict shape
    mock_model.predict.return_value = np.array([0.0, 0.0011, 0.0022])
    
    # Mock plt subplots output
    mock_fig = MagicMock()
    mock_ax_main = MagicMock()
    mock_ax_res = MagicMock()
    mock_plt.subplots.return_value = (mock_fig, (mock_ax_main, mock_ax_res))
    
    # Call function without saving
    plot_local_fit(mock_test, mock_model)
    
    # Assertions on pipeline flow
    mock_da.get_local_data.assert_called_once_with(mock_test)
    mock_model.predict.assert_called_once_with(mock_X)
    mock_plt.subplots.assert_called_once()
    mock_plt.show.assert_called_once()
    mock_plt.savefig.assert_not_called()
    mock_plt.close.assert_called_once_with(mock_fig)
    
    mock_ax_main.scatter.assert_called_once()
    mock_ax_main.plot.assert_called_once()
    mock_ax_res.scatter.assert_called_once()

@patch("creep_model.viz.plots.plt")
@patch("creep_model.viz.plots.DataAssembler")
def test_plot_local_fit_with_save(mock_da, mock_plt):
    mock_test = MagicMock()
    mock_test.strain_series = np.array([0.0, 0.001])
    
    mock_model = MagicMock(spec=BaseCreepModel)
    mock_model.is_fitted = True
    
    mock_da.get_local_data.return_value = (np.array([[1]]), np.array([1]))
    mock_model.predict.return_value = np.array([1])
    
    mock_fig = MagicMock()
    mock_plt.subplots.return_value = (mock_fig, (MagicMock(), MagicMock()))
    
    # Call function with saving triggers the savefig branch
    plot_local_fit(mock_test, mock_model, save_path="test.png")
    
    mock_plt.savefig.assert_called_once_with("test.png", dpi=300)
    mock_plt.show.assert_not_called()

def test_plot_local_fit_not_fitted():
    mock_test = MagicMock()
    mock_model = MagicMock(spec=BaseCreepModel)
    mock_model.is_fitted = False
    
    with pytest.raises(ValueError, match="must be fitted"):
        plot_local_fit(mock_test, mock_model)

@patch("creep_model.viz.plots.plt")
@patch("creep_model.viz.plots.DataAssembler")
def test_plot_local_fit_no_positive_strain_diff(mock_da, mock_plt):
    # Test case where smallest_jump logic triggers the fallback to 1e-6
    mock_test = MagicMock()
    # No positive differences (only equal or negative differences)
    mock_test.strain_series = np.array([0.002, 0.001, 0.0])
    
    mock_model = MagicMock(spec=BaseCreepModel)
    mock_model.is_fitted = True
    
    mock_da.get_local_data.return_value = (np.array([[1]]), np.array([1]))
    mock_model.predict.return_value = np.array([1])
    
    mock_fig = MagicMock()
    mock_ax_main = MagicMock()
    mock_ax_res = MagicMock()
    mock_plt.subplots.return_value = (mock_fig, (mock_ax_main, mock_ax_res))
    
    plot_local_fit(mock_test, mock_model)
    
    # Assert axhline was called precisely with fallback logic values 1e-6 / 2
    mock_ax_res.axhline.assert_any_call(1e-6/2, color='red', linestyle='--')