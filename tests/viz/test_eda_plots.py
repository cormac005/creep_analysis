from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from creep_model.domain import CreepTest
from creep_model.viz.eda_plots import (
    EDAStyleConfig,
    _nominal_stress,
    plot_creep_stage_boundaries,
    plot_distribution_by_quality,
    plot_eps_dot_ss_vs_stress,
    plot_eps_dot_ss_vs_temperature,
    plot_eps_tilde_0_vs_age,
    plot_eps_tilde_0_vs_stress,
    plot_mean_eps_dot_ss_bar,
    plot_pairwise_relationships,
)

from creep_model.viz.eda_plots import plot_temperature_fit_example

def test_plot_temperature_fit_example(tmp_path, default_style):
    X = np.linspace(0, 100, 20)
    y = np.linspace(0, 0.05, 20)
    temps = np.full_like(X, 23.5)
    temps[5] = np.nan  # Includes NaNs to test valid_mask filtering
    
    test = CreepTest(
        test_id="Test.01",
        time_series=X,
        strain_series=y,
        temp_time_series=X,
        temperature_readings=temps,
        applied_stress_MPa=20.0,
        age_days=14,
        print_quality="Standard",
    )
    
    out_path = plot_temperature_fit_example(test, tmp_path, default_style)
    assert out_path.exists()

def test_plot_temperature_fit_example_with_title(tmp_path):
    style_with_title = EDAStyleConfig(show_titles=True)
    X = np.linspace(0, 50, 10)
    y = np.linspace(0, 0.02, 10)
    temps = np.full_like(X, 22.0)
    
    test = CreepTest(
        test_id="Test_02",
        time_series=X,
        strain_series=y,
        temp_time_series=X,
        temperature_readings=temps,
        applied_stress_MPa=10.0,
        age_days=7,
        print_quality="High",
    )
    
    out_path = plot_temperature_fit_example(test, tmp_path, style_with_title)
    assert out_path.exists()

def test_eda_plots_with_secondary_temp_col(sample_eda_df, default_style, tmp_path):
    df = sample_eda_df.copy()
    df["Mean_Temp_C_Secondary_Creep"] = [20.0, 21.0, 22.0, 23.0, 24.0, 25.0]
    df = df.drop(columns=["Initial_Temp_C"])
    
    plot_eps_tilde_0_vs_stress(df, tmp_path, default_style)
    plot_eps_dot_ss_vs_stress(df, tmp_path, default_style)
    plot_eps_dot_ss_vs_temperature(df, tmp_path, default_style)
    plot_pairwise_relationships(df, tmp_path, default_style)

def test_eda_plots_with_mean_temp_col(sample_eda_df, default_style, tmp_path):
    df = sample_eda_df.copy()
    df["Mean_Temp_C"] = [20.0, 21.0, 22.0, 23.0, 24.0, 25.0]
    
    plot_eps_tilde_0_vs_stress(df, tmp_path, default_style)
    plot_eps_dot_ss_vs_stress(df, tmp_path, default_style)
    plot_eps_dot_ss_vs_temperature(df, tmp_path, default_style)
    plot_pairwise_relationships(df, tmp_path, default_style)

def test_eda_plots_empty_eps_dot_ss(sample_eda_df, default_style, tmp_path):
    df = sample_eda_df.copy()
    df["Eps_Dot_Ss"] = np.nan
    
    plot_eps_dot_ss_vs_stress(df, tmp_path, default_style)
    plot_eps_dot_ss_vs_temperature(df, tmp_path, default_style)
    plot_distribution_by_quality(df, tmp_path, default_style)
    plot_mean_eps_dot_ss_bar(df, tmp_path, default_style)
    plot_pairwise_relationships(df, tmp_path, default_style)

@pytest.fixture(autouse=True)
def close_plt_figures():
    yield
    plt.close("all")

@pytest.fixture
def sample_eda_df() -> pd.DataFrame:
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

def test_nominal_stress_mapping():
    df = pd.DataFrame({"Applied_Stress_MPa": [10.0, 14.9, 15.0, 24.9, 25.0, 35.0]})
    mapped = _nominal_stress(df)
    expected = pd.Series([10, 10, 20, 20, 30, 30], name="Applied_Stress_MPa")
    pd.testing.assert_series_equal(mapped, expected)

def test_eda_style_config_custom():
    style = EDAStyleConfig(dpi=150, show_titles=False, cmap="viridis")
    assert style.dpi == 150
    assert not style.show_titles
    assert style.cmap == "viridis"

def test_plot_eps_tilde_0_vs_stress(sample_eda_df, default_style, tmp_path):
    out_path = plot_eps_tilde_0_vs_stress(sample_eda_df, tmp_path, default_style)
    assert out_path.exists()

def test_plot_eps_tilde_0_vs_stress_missing_initial_temp_fallback(sample_eda_df, default_style, tmp_path):
    df_no_init_temp = sample_eda_df.drop(columns=["Initial_Temp_C"])
    out_path = plot_eps_tilde_0_vs_stress(df_no_init_temp, tmp_path, default_style)
    assert out_path.exists()

def test_plot_eps_dot_ss_vs_stress(sample_eda_df, default_style, tmp_path):
    out_path = plot_eps_dot_ss_vs_stress(sample_eda_df, tmp_path, default_style)
    assert out_path.exists()

def test_plot_eps_tilde_0_vs_age(sample_eda_df, default_style, tmp_path):
    out_path = plot_eps_tilde_0_vs_age(sample_eda_df, tmp_path, default_style)
    assert out_path.exists()

def test_plot_eps_dot_ss_vs_temperature(sample_eda_df, default_style, tmp_path):
    out_path = plot_eps_dot_ss_vs_temperature(sample_eda_df, tmp_path, default_style)
    assert out_path.exists()

def test_plot_distribution_by_quality(sample_eda_df, default_style, tmp_path):
    out_path = plot_distribution_by_quality(sample_eda_df, tmp_path, default_style)
    assert out_path.exists()

def test_plot_mean_eps_dot_ss_bar(sample_eda_df, default_style, tmp_path):
    out_path = plot_mean_eps_dot_ss_bar(sample_eda_df, tmp_path, default_style)
    assert out_path.exists()

def test_plot_pairwise_relationships(sample_eda_df, default_style, tmp_path):
    out_path = plot_pairwise_relationships(sample_eda_df, tmp_path, default_style)
    assert out_path.exists()

def test_plots_work_with_show_titles_false(sample_eda_df, tmp_path):
    no_title_style = EDAStyleConfig(show_titles=False)
    out1 = plot_eps_tilde_0_vs_stress(sample_eda_df, tmp_path, no_title_style)
    out2 = plot_distribution_by_quality(sample_eda_df, tmp_path, no_title_style)
    assert out1.exists()
    assert out2.exists()

def test_eda_plots_with_titles(sample_eda_df, tmp_path):
    style_with_titles = EDAStyleConfig(show_titles=True)
    plot_eps_tilde_0_vs_stress(sample_eda_df, tmp_path, style_with_titles)
    plot_eps_dot_ss_vs_stress(sample_eda_df, tmp_path, style_with_titles)
    plot_eps_tilde_0_vs_age(sample_eda_df, tmp_path, style_with_titles)
    plot_eps_dot_ss_vs_temperature(sample_eda_df, tmp_path, style_with_titles)
    plot_distribution_by_quality(sample_eda_df, tmp_path, style_with_titles)
    plot_mean_eps_dot_ss_bar(sample_eda_df, tmp_path, style_with_titles)
    plot_pairwise_relationships(sample_eda_df, tmp_path, style_with_titles)

def test_eda_plots_empty_groups(sample_eda_df, default_style, tmp_path):
    df_one_quality = sample_eda_df[sample_eda_df["Print_Quality"] == "High"]
    plot_eps_tilde_0_vs_stress(df_one_quality, tmp_path, default_style)
    plot_eps_dot_ss_vs_stress(df_one_quality, tmp_path, default_style)

# --- test plot_creep_stage_boundaries ---

def test_plot_creep_stage_boundaries_primary_only(tmp_path, default_style):
    y = np.concatenate([np.repeat(i, i+1) for i in range(20)])
    X = np.arange(len(y))
    test = CreepTest(
        test_id="T1",
        time_series=X,
        strain_series=y,
        temp_time_series=X,
        temperature_readings=np.full_like(X, 20.0),
        applied_stress_MPa=10.0,
        age_days=7,
        print_quality="Standard",
    )
    out_path = plot_creep_stage_boundaries(test, k1=25, k2=30, output_dir=tmp_path, style=default_style)
    assert out_path.exists()

def test_plot_creep_stage_boundaries_no_tertiary(tmp_path, default_style):
    y1 = np.concatenate([np.repeat(i, i+1) for i in range(5)]) # 0..4
    y2 = np.concatenate([np.repeat(i, 6) for i in range(5, 15)]) # 5..14
    y = np.concatenate([y1, y2])
    X = np.arange(len(y))
    test = CreepTest(
        test_id="T2",
        time_series=X,
        strain_series=y,
        temp_time_series=X,
        temperature_readings=np.full_like(X, 20.0),
        applied_stress_MPa=10.0,
        age_days=7,
        print_quality="Standard",
    )
    out_path = plot_creep_stage_boundaries(test, k1=2, k2=20, output_dir=tmp_path, style=default_style)
    assert out_path.exists()

def test_plot_creep_stage_boundaries_full(tmp_path):
    y1 = np.concatenate([np.repeat(i, i+1) for i in range(5)]) # 0..4
    y2 = np.concatenate([np.repeat(i, 6) for i in range(5, 10)]) # 5..9
    y3 = np.concatenate([np.repeat(i, 4 - (i-10)) for i in range(10, 14)]) # 10..13
    y = np.concatenate([y1, y2, y3])
    X = np.arange(len(y))
    test = CreepTest(
        test_id="T3",
        time_series=X,
        strain_series=y,
        temp_time_series=X,
        temperature_readings=np.full_like(X, 20.0),
        applied_stress_MPa=10.0,
        age_days=7,
        print_quality="Standard",
    )
    
    style_with_titles = EDAStyleConfig(show_titles=True)
    out_path = plot_creep_stage_boundaries(test, k1=2, k2=6, output_dir=tmp_path, style=style_with_titles)
    assert out_path.exists()