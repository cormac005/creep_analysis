from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest

from creep_model.viz.tlv_plots import TLVStyleConfig, plot_tlv_fit_summary

@pytest.fixture(autouse=True)
def close_plt_figures():
    yield
    plt.close("all")

def test_tlv_style_config():
    style = TLVStyleConfig(dpi=150, font_size_label=10)
    assert style.dpi == 150
    assert style.font_size_label == 10

def test_plot_tlv_fit_summary(tmp_path):
    style = TLVStyleConfig()
    stresses = [10.0, 20.0, 30.0]
    quality = "Standard"
    
    time_s = np.linspace(0, 100, 10)
    strain_m = np.linspace(0, 0.01, 10)
    strain_p = np.linspace(0, 0.009, 10)
    
    grouped_data = {
        ("Standard", 10.0): [
            {
                "test_id": "T1",
                "stress": 10.0,
                "time_s": time_s,
                "strain_measured": strain_m,
                "strain_predicted": strain_p,
            }
        ],
        ("Standard", 20.0): [
            {
                "test_id": "T2",
                "stress": 20.0,
                "time_s": time_s,
                "strain_measured": strain_m,
                "strain_predicted": strain_p,
            }
        ],
        ("Standard", 30.0): [],
    }
    
    out_path = plot_tlv_fit_summary(quality, grouped_data, stresses, tmp_path, style)
    assert out_path.exists()
    assert out_path.name == f"Summary_{quality}_Combined.png"

def test_plot_tlv_fit_summary_empty_data(tmp_path):
    style = TLVStyleConfig()
    stresses = [10.0, 20.0, 30.0]
    quality = "High"
    grouped_data = {}
    
    out_path = plot_tlv_fit_summary(quality, grouped_data, stresses, tmp_path, style)
    assert out_path.exists()