from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from creep_model.viz.eda_plots import (
    EDAStyleConfig,
    _nominal_stress,
    plot_distribution_by_quality,
    plot_eps_dot_ss_vs_stress,
    plot_eps_dot_ss_vs_temperature,
    plot_eps_tilde_0_vs_age,
    plot_eps_tilde_0_vs_stress,
    plot_mean_eps_dot_ss_bar,
    plot_pairwise_relationships,
)


@pytest.fixture(autouse=True)
def close_plt_figures():
    """Fixture to ensure all matplotlib figures are closed after every test."""
    yield
    plt.close("all")


@pytest.fixture
def sample_eda_df() -> pd.DataFrame:
    """Provides a realistic mock eda_summary DataFrame with all necessary columns."""
    return pd.DataFrame({
        "Applied_Stress_MPa": [10.0, 12.0, 20.0, 22.0, 30.0, 32.0],
        "Print_Quality": ["Standard", "Standard", "High", "High", "Standard", "High"],
        "Age_Days": [7, 14, 21, 28, 35, 42],
        "Eps_Tilde_0": [0.002, 0.003, 0.005, 0.006, 0.008, 0.009],
        "Eps_Dot_Ss": [1.0e-6, 1.2e-6, 2.0e-6, 2.5e-6, 3.0e-6, 3.5e-6],
        "Initial_Temp_C": [20.0, 21.0, 22.0, 23.0, 24.0, 25.0],
        "Mean_Temp_C_Secondary_Creep": [21.0, 22.0, 23.0, 24.0, 25.0, 26.0],
    })


@pytest.fixture
def default_style() -> EDAStyleConfig:
    return EDAStyleConfig()


# =============================================================================
# Helper & Config Tests
# =============================================================================

def test_nominal_stress_mapping():
    """Tests that _nominal_stress maps stress values to nearest bands (10, 20, 30)."""
    df = pd.DataFrame({"Applied_Stress_MPa": [10.0, 14.9, 15.0, 24.9, 25.0, 35.0]})
    mapped = _nominal_stress(df)
    expected = pd.Series([10, 10, 20, 20, 30, 30], name="Applied_Stress_MPa")
    pd.testing.assert_series_equal(mapped, expected)


def test_eda_style_config_custom():
    """Verifies custom EDAStyleConfig initialization."""
    style = EDAStyleConfig(dpi=150, show_titles=False, cmap="viridis")
    assert style.dpi == 150
    assert not style.show_titles
    assert style.cmap == "viridis"


# =============================================================================
# Plotting Function Tests
# =============================================================================

def test_plot_eps_tilde_0_vs_stress(sample_eda_df, default_style, tmp_path):
    """Tests plotting eps_tilde_0 vs stress with Initial_Temp_C column present."""
    out_path = plot_eps_tilde_0_vs_stress(sample_eda_df, tmp_path, default_style)
    assert out_path.exists()
    assert out_path.is_file()
    assert out_path.stat().st_size > 0
    assert out_path.name == "eps_tilde_0_vs_stress.png"


def test_plot_eps_tilde_0_vs_stress_missing_initial_temp_fallback(sample_eda_df, default_style, tmp_path):
    """Tests fallback to Mean_Temp_C_Secondary_Creep when Initial_Temp_C is absent."""
    df_no_init_temp = sample_eda_df.drop(columns=["Initial_Temp_C"])
    out_path = plot_eps_tilde_0_vs_stress(df_no_init_temp, tmp_path, default_style)
    assert out_path.exists()
    assert out_path.stat().st_size > 0


def test_plot_eps_dot_ss_vs_stress(sample_eda_df, default_style, tmp_path):
    """Tests plotting eps_dot_ss vs stress."""
    out_path = plot_eps_dot_ss_vs_stress(sample_eda_df, tmp_path, default_style)
    assert out_path.exists()
    assert out_path.stat().st_size > 0
    assert out_path.name == "eps_dot_ss_vs_stress.png"


def test_plot_eps_tilde_0_vs_age(sample_eda_df, default_style, tmp_path):
    """Tests plotting eps_tilde_0 vs specimen age."""
    out_path = plot_eps_tilde_0_vs_age(sample_eda_df, tmp_path, default_style)
    assert out_path.exists()
    assert out_path.stat().st_size > 0
    assert out_path.name == "eps_tilde_0_vs_age.png"


def test_plot_eps_dot_ss_vs_temperature(sample_eda_df, default_style, tmp_path):
    """Tests plotting eps_dot_ss vs temperature."""
    out_path = plot_eps_dot_ss_vs_temperature(sample_eda_df, tmp_path, default_style)
    assert out_path.exists()
    assert out_path.stat().st_size > 0
    assert out_path.name == "eps_dot_ss_vs_temperature.png"


def test_plot_distribution_by_quality(sample_eda_df, default_style, tmp_path):
    """Tests plotting distribution box+strip plots split by print quality."""
    out_path = plot_distribution_by_quality(sample_eda_df, tmp_path, default_style)
    assert out_path.exists()
    assert out_path.stat().st_size > 0
    assert out_path.name == "eda_distribution_by_quality.png"


def test_plot_mean_eps_dot_ss_bar(sample_eda_df, default_style, tmp_path):
    """Tests plotting mean strain rate bar chart grouped by quality and stress band."""
    out_path = plot_mean_eps_dot_ss_bar(sample_eda_df, tmp_path, default_style)
    assert out_path.exists()
    assert out_path.stat().st_size > 0
    assert out_path.name == "eda_mean_eps_dot_ss_bar.png"


def test_plot_pairwise_relationships(sample_eda_df, default_style, tmp_path):
    """Tests plotting Seaborn PairGrid across numeric EDA metrics."""
    out_path = plot_pairwise_relationships(sample_eda_df, tmp_path, default_style)
    assert out_path.exists()
    assert out_path.stat().st_size > 0
    assert out_path.name == "eda_pairwise_relationships.png"


def test_plots_work_with_show_titles_false(sample_eda_df, tmp_path):
    """Ensures plotting functions execute cleanly when show_titles is False."""
    no_title_style = EDAStyleConfig(show_titles=False)
    out1 = plot_eps_tilde_0_vs_stress(sample_eda_df, tmp_path, no_title_style)
    out2 = plot_distribution_by_quality(sample_eda_df, tmp_path, no_title_style)
    assert out1.exists()
    assert out2.exists()