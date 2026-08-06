"""
Generate publication-quality EDA figures and LaTeX summary tables from eda_results.h5.
This script should be run AFTER 03_compute_eda_stats.py has been executed.

Outputs generated:
    1. Plots (saved to <general_output_directory>/plots/eda/):
        - creep_rate_vs_stress.png: Secondary creep strain rate vs. applied stress
        - initial_strain_vs_stress.png: Initial strain intercept vs. applied stress
        - temperature_profiles.png: Raw vs interpolated thermal history during testing
        - eda_summary_2x2.png: Consolidated 2x2 grid comparing key EDA parameters
        - eda_pairwise_relationships.png: Pairwise relationship grid via eda_plots module
    2. Tables (saved to <general_output_directory>/tables/):
        - eda_summary_table.tex: LaTeX table with explicit Mean/Std. Dev. rows
"""
from pathlib import Path
import h5py
import matplotlib

# Force non-interactive backend for script execution
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from creep_model.config import config
from creep_model.viz.eda_plots import EDAStyleConfig, plot_pairwise_relationships

# --- CONFIGURATION & TYPOGRAPHY STANDARDS ---
SHOW_TITLE = False

# Canvas Sizing (Inches)
FIG_SIZE_SINGLE = (6.2, 4.0)
FIG_SIZE_2X2 = (8.5, 6.5)

# Font Size Hierarchy
FONT_SIZE_LABEL = 11
FONT_SIZE_TICK = 10
FONT_SIZE_LEGEND = 9.5
FONT_SIZE_TITLE = 11
FONT_SIZE_ANNOT = 9

# Color Palette & Styles
COLOR_HIGH = "#1f77b4"  # High Quality
COLOR_STD = "#ff7f0e"  # Standard Quality
COLOR_TEMP = "#d62728"  # Temperature Data
GRID_COLOR = "#CCCCCC"

EDA_H5_PATH = Path(config.data_output_directory) / "eda_results.h5"
PLOTS_DIR = Path(config.general_output_directory) / "plots" / "eda"
TABLES_DIR = Path(config.general_output_directory) / "tables"


def load_eda_data() -> tuple[pd.DataFrame, dict]:
    """
    Loads consolidated EDA summary table and test-level temperature profiles
    from eda_results.h5 and computes overall mean test temperatures.
    """
    if not EDA_H5_PATH.exists():
        raise FileNotFoundError(
            f"Required data file {EDA_H5_PATH} does not exist. "
            "Please run 03_compute_eda_stats.py first."
        )

    with h5py.File(EDA_H5_PATH, "r") as f:
        if "eda_summary" not in f:
            raise KeyError(f"'eda_summary' group missing in {EDA_H5_PATH}")

        summary_grp = f["eda_summary"]
        data = {}
        for key in summary_grp.keys():
            arr = summary_grp[key][:]
            if arr.dtype.kind in ["S", "O"]:
                arr = [x.decode("utf-8") if isinstance(x, bytes) else str(x) for x in arr]
            data[key] = arr

        df = pd.DataFrame(data)
        df["Nominal_Stress_MPa"] = df["Applied_Stress_MPa"].round(-1)

        temp_profiles = {}
        if "tests" in f:
            tests_grp = f["tests"]
            for test_id in tests_grp.keys():
                t_grp = tests_grp[test_id]
                entry = {}
                if "temp_time_s" in t_grp and "temperature_raw" in t_grp:
                    entry["temp_time_s"] = t_grp["temp_time_s"][:]
                    entry["temperature_raw"] = t_grp["temperature_raw"][:]
                if "time_s" in t_grp and "temperature_interpolated" in t_grp:
                    entry["time_s"] = t_grp["time_s"][:]
                    entry["temperature_interpolated"] = t_grp["temperature_interpolated"][:]
                if entry:
                    temp_profiles[test_id] = entry

        overall_mean_temps = []
        for _, row in df.iterrows():
            tid = row["Test_ID"]
            t_mean = None
            if tid in temp_profiles:
                p = temp_profiles[tid]
                if "temperature_raw" in p and len(p["temperature_raw"]) > 0:
                    t_mean = float(np.nanmean(p["temperature_raw"]))
                elif "temperature_interpolated" in p and len(p["temperature_interpolated"]) > 0:
                    t_mean = float(np.nanmean(p["temperature_interpolated"]))
            
            if t_mean is None or np.isnan(t_mean):
                t_mean = row.get("Mean_Temp_C_Secondary_Creep", np.nan)
                if pd.isnull(t_mean) or np.isnan(t_mean):
                    t_mean = row.get("Initial_Temp_C", 20.0)
            
            overall_mean_temps.append(t_mean)
        
        df["Mean_Temp_C"] = overall_mean_temps

    return df, temp_profiles


def plot_creep_rate_vs_stress(df: pd.DataFrame, output_dir: Path) -> None:
    """Plots secondary creep rate (Eps_Dot_Ss) vs applied stress (excludes non-detected secondary creep)."""
    fig, ax = plt.subplots(figsize=FIG_SIZE_SINGLE)
    
    valid_df = df[df["Eps_Dot_Ss"].notna() & (df["Eps_Dot_Ss"] > 0)].copy()
    qualities = [("High", COLOR_HIGH, "o"), ("Standard", COLOR_STD, "s")]
    has_positive = not valid_df.empty
    ax.grid(True, color=GRID_COLOR, linestyle="--", alpha=0.5, which="both" if has_positive else "major")

    for q_name, color, marker in qualities:
        sub = valid_df[valid_df["Print_Quality"] == q_name].sort_values("Applied_Stress_MPa")
        if sub.empty:
            continue

        ax.scatter(
            sub["Applied_Stress_MPa"],
            sub["Eps_Dot_Ss"],
            color=color,
            marker=marker,
            s=40,
            alpha=0.8,
            label=f"{q_name} Quality",
        )

        means = sub.groupby("Nominal_Stress_MPa", as_index=False)[
            ["Applied_Stress_MPa", "Eps_Dot_Ss"]
        ].mean()
        if not means.empty:
            ax.plot(
                means["Applied_Stress_MPa"],
                means["Eps_Dot_Ss"],
                color=color,
                linestyle="--",
                linewidth=1.5,
                alpha=0.7,
            )

    if has_positive:
        ax.set_yscale("log")

    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(loc="upper left", fontsize=FONT_SIZE_LEGEND, framealpha=0.9)
    ax.set_ylabel(
        r"Secondary Creep Rate $\dot{\varepsilon}_{ss}$ ($\text{s}^{-1}$)",
        fontsize=FONT_SIZE_LABEL,
        fontweight="bold",
    )
    ax.tick_params(axis="both", labelsize=FONT_SIZE_TICK)
    ax.legend(loc="upper left", fontsize=FONT_SIZE_LEGEND, framealpha=0.9)

    if SHOW_TITLE:
        ax.set_title("Secondary Creep Rate vs. Applied Stress", fontsize=FONT_SIZE_TITLE)

    plt.tight_layout()
    plt.savefig(output_dir / "creep_rate_vs_stress.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_initial_strain_vs_stress(df: pd.DataFrame, output_dir: Path) -> None:
    """Plots initial strain intercept (Eps_Tilde_0) vs applied stress."""
    fig, ax = plt.subplots(figsize=FIG_SIZE_SINGLE)
    ax.grid(True, color=GRID_COLOR, linestyle="--", alpha=0.5)

    qualities = [("High", COLOR_HIGH, "o"), ("Standard", COLOR_STD, "s")]

    for q_name, color, marker in qualities:
        sub = df[df["Print_Quality"] == q_name].sort_values("Applied_Stress_MPa")
        if sub.empty:
            continue

        ax.scatter(
            sub["Applied_Stress_MPa"],
            sub["Eps_Tilde_0"],
            color=color,
            marker=marker,
            s=40,
            alpha=0.8,
            label=f"{q_name} Quality",
        )

        means = sub.groupby("Nominal_Stress_MPa", as_index=False)[
            ["Applied_Stress_MPa", "Eps_Tilde_0"]
        ].mean()
        if not means.empty:
            ax.plot(
                means["Applied_Stress_MPa"],
                means["Eps_Tilde_0"],
                color=color,
                linestyle="--",
                linewidth=1.5,
                alpha=0.7,
            )

    ax.set_xlabel("Applied Stress (MPa)", fontsize=FONT_SIZE_LABEL, fontweight="bold")
    ax.set_ylabel(
        r"Initial Strain $\tilde{\varepsilon}_0$",
        fontsize=FONT_SIZE_LABEL,
        fontweight="bold",
    )
    ax.tick_params(axis="both", labelsize=FONT_SIZE_TICK)
    ax.legend(loc="upper left", fontsize=FONT_SIZE_LEGEND, framealpha=0.9)

    y_max = df["Eps_Tilde_0"].max() if ("Eps_Tilde_0" in df.columns and not df["Eps_Tilde_0"].empty) else 0.0
    top_limit = y_max * 1.15 if (y_max is not None and y_max > 0) else 1.0
    ax.set_ylim(bottom=0, top=top_limit)

    if SHOW_TITLE:
        ax.set_title("Initial Strain vs. Applied Stress", fontsize=FONT_SIZE_TITLE)

    plt.tight_layout()
    plt.savefig(output_dir / "initial_strain_vs_stress.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_temperature_profiles(temp_profiles: dict, output_dir: Path) -> None:
    """Plots representative raw and interpolated thermal history during tests."""
    if not temp_profiles:
        return

    fig, ax = plt.subplots(figsize=FIG_SIZE_SINGLE)
    ax.grid(True, color=GRID_COLOR, linestyle="--", alpha=0.5)

    sample_test_ids = list(temp_profiles.keys())[:3]
    linestyles = ["-", "--", "-."]

    for idx, test_id in enumerate(sample_test_ids):
        p_data = temp_profiles[test_id]
        ls = linestyles[idx % len(linestyles)]

        if "temp_time_s" in p_data and "temperature_raw" in p_data:
            ax.scatter(
                p_data["temp_time_s"],
                p_data["temperature_raw"],
                s=15,
                alpha=0.5,
                label=f"{test_id} (Raw)",
            )

        if "time_s" in p_data and "temperature_interpolated" in p_data:
            ax.plot(
                p_data["time_s"],
                p_data["temperature_interpolated"],
                linestyle=ls,
                linewidth=1.5,
                label=f"{test_id} (Interp.)",
            )

    ax.set_xlabel("Time (s)", fontsize=FONT_SIZE_LABEL, fontweight="bold")
    ax.set_ylabel("Temperature (°C)", fontsize=FONT_SIZE_LABEL, fontweight="bold")
    ax.tick_params(axis="both", labelsize=FONT_SIZE_TICK)
    ax.legend(loc="best", fontsize=FONT_SIZE_LEGEND, framealpha=0.9)

    if SHOW_TITLE:
        ax.set_title("Representative Thermal History Profiles", fontsize=FONT_SIZE_TITLE)

    plt.tight_layout()
    plt.savefig(output_dir / "temperature_profiles.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_eda_summary_2x2(df: pd.DataFrame, output_dir: Path) -> None:
    """Generates a consolidated 2x2 grid comparing key metrics across quality groups."""
    fig, axes = plt.subplots(2, 2, figsize=FIG_SIZE_2X2)
    qualities = [("High", COLOR_HIGH, "o"), ("Standard", COLOR_STD, "s")]

    metrics = [
        ("Eps_Tilde_0", r"Initial Strain $\tilde{\varepsilon}_0$", False),
        ("Eps_Dot_Ss", r"Creep Rate $\dot{\varepsilon}_{ss}$ ($\text{s}^{-1}$)", True),
        ("Initial_Temp_C", r"Initial Temp ($^\circ\text{C}$)", False),
        ("Mean_Temp_C", r"Mean Test Temp ($^\circ\text{C}$)", False),
    ]

    for (row_idx, col_idx), (col_name, label, use_log) in zip(
        [(0, 0), (0, 1), (1, 0), (1, 1)], metrics
    ):
        ax = axes[row_idx, col_idx]
        ax.grid(True, color=GRID_COLOR, linestyle="--", alpha=0.4)

        for q_name, color, marker in qualities:
            sub = df[df["Print_Quality"] == q_name].sort_values("Applied_Stress_MPa")
            if sub.empty:
                continue

            if col_name == "Eps_Dot_Ss":
                plot_data = sub[sub["Eps_Dot_Ss"].notna() & (sub["Eps_Dot_Ss"] > 0)]
            else:
                plot_data = sub

            if plot_data.empty:
                continue

            ax.scatter(
                plot_data["Applied_Stress_MPa"],
                plot_data[col_name],
                color=color,
                marker=marker,
                s=30,
                alpha=0.7,
                label=f"{q_name}" if (row_idx == 0 and col_idx == 0) else "",
            )

            means = plot_data.groupby("Nominal_Stress_MPa", as_index=False)[
                ["Applied_Stress_MPa", col_name]
            ].mean()
            if not means.empty:
                ax.plot(
                    means["Applied_Stress_MPa"],
                    means[col_name],
                    color=color,
                    linestyle="--",
                    linewidth=1.2,
                    alpha=0.7,
                )

        if use_log and (df[col_name] > 0).any():
            ax.set_yscale("log")

        ax.set_ylabel(label, fontsize=FONT_SIZE_LABEL)
        ax.tick_params(axis="both", labelsize=FONT_SIZE_TICK)

        if row_idx == 1:
            ax.set_xlabel("Applied Stress (MPa)", fontsize=FONT_SIZE_LABEL)

    axes[0, 0].legend(loc="upper left", fontsize=FONT_SIZE_LEGEND, framealpha=0.9)

    plt.tight_layout()
    plt.savefig(output_dir / "eda_summary_2x2.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def generate_eda_latex_table(df: pd.DataFrame, output_dir: Path) -> None:
    """Generates a publication-ready LaTeX table with Range instead of Std. Dev. and explicit creep counts."""
    output_dir.mkdir(parents=True, exist_ok=True)
    tex_path = output_dir / "eda_summary_table.tex"

    # Helper function to compute mean and range string
    def calc_stats(series, scale=1.0, fmt="{:.2f}", filter_positive=False):
        if filter_positive:
            valid = series[series.notna() & (series > 0)]
        else:
            valid = series.dropna()

        count = len(valid)
        if count == 0:
            return "N/A*", "N/A*"
        
        mean_val = valid.mean() * scale
        mean_str = fmt.format(mean_val)

        # Return N/A for range if there is only 1 value
        if count <= 1:
            range_str = "N/A"
        else:
            v_min = valid.min() * scale
            v_max = valid.max() * scale
            if np.isclose(v_min, v_max):
                range_str = "N/A"
            else:
                range_str = f"{fmt.format(v_min)}--{fmt.format(v_max)}"

        return mean_str, range_str

    latex_str = [
        r"\begin{table}[htbp]",
        r"    \centering",
        r"    \caption{Exploratory Data Analysis (EDA) statistics for creep tests grouped by print quality and stress level.}",
        r"    \label{tab:eda_summary}",
        r"    \resizebox{\linewidth}{!}{%",
        r"    \begin{tabular}{ccccccccc}",
        r"        \toprule",
        r"        \textbf{\shortstack{$\sigma$\\(\text{MPa})}} & "
        r"\textbf{\shortstack{$N$\\(\text{Total})}} & "
        r"\textbf{\shortstack{$N_{\text{sec}}$\\(\text{Secondary})}} & "
        r"\textbf{\shortstack{$N_{\text{tert}}$\\(\text{Tertiary})}} & "
        r"\textbf{Statistic} & "
        r"\textbf{\shortstack{Initial Temp\\(${^\circ}\text{C}$)}} & "
        r"\textbf{\shortstack{Mean Temp\\(${^\circ}\text{C}$)}} & "
        r"\textbf{\shortstack{$\tilde{\varepsilon}_0$\\($\times 10^{-3}$)}} & "
        r"\textbf{\shortstack{$\dot{\varepsilon}_{ss}$\\($\times 10^{-7} \text{s}^{-1}$)}} \\",
        r"        \midrule",
    ]

    qualities = ["High", "Standard"]
    for q_idx, q_name in enumerate(qualities):
        q_df = df[df["Print_Quality"] == q_name]
        if q_df.empty:
            continue

        latex_str.append(f"        \\multicolumn{{9}}{{l}}{{\\textit{{{q_name} Quality}}}} \\\\")
        latex_str.append(r"        \midrule")

        stresses = sorted(q_df["Nominal_Stress_MPa"].unique())
        for stress in stresses:
            sub = q_df[q_df["Nominal_Stress_MPa"] == stress]
            n_total = len(sub)

            # Secondary creep count (Eps_Dot_Ss > 0)
            if "Eps_Dot_Ss" in sub.columns:
                n_sec = int((sub["Eps_Dot_Ss"].notna() & (sub["Eps_Dot_Ss"] > 0)).sum())
            else:
                n_sec = 0

            # Tertiary creep count detection
            if "Has_Tertiary" in sub.columns:
                n_tert = int(sub["Has_Tertiary"].sum())
            elif "Tertiary" in sub.columns:
                n_tert = int(sub["Tertiary"].sum())
            elif "k2" in sub.columns:
                n_tert = int((sub["k2"] > 0).sum())
            elif "has_tertiary" in sub.columns:
                n_tert = int(sub["has_tertiary"].sum())
            else:
                n_tert = 0

            # Calculate statistics for each metric
            t_init_m, t_init_r = calc_stats(sub["Initial_Temp_C"], scale=1.0, fmt="{:.1f}")
            
            temp_sec_col = "Mean_Temp_C" if "Mean_Temp_C" in sub.columns else "Mean_Temp_C_Secondary_Creep"
            t_sec_m, t_sec_r = calc_stats(sub[temp_sec_col], scale=1.0, fmt="{:.1f}")
            
            eps0_m, eps0_r = calc_stats(sub["Eps_Tilde_0"], scale=1e3, fmt="{:.2f}")
            
            eps_ss_m, eps_ss_r = calc_stats(sub["Eps_Dot_Ss"], scale=1e7, fmt="{:.2f}", filter_positive=True)

            # Mean Row
            latex_str.append(
                f"        {int(stress)} & {n_total} & {n_sec} & {n_tert} & Mean & {t_init_m} & {t_sec_m} & {eps0_m} & {eps_ss_m} \\\\"
            )
            # Range Row
            latex_str.append(
                f"        & & & & Range & {t_init_r} & {t_sec_r} & {eps0_r} & {eps_ss_r} \\\\"
            )
            latex_str.append(r"        \addlinespace")

        if q_idx < len(qualities) - 1:
            latex_str.append(r"        \midrule")

    latex_str.extend([
        r"        \bottomrule",
        r"    \end{tabular}%",
        r"    }",
        r"    \begin{flushleft}",
        r"        \footnotesize{*N/A indicates secondary creep was not detected for any specimens in this test group. Range is reported as N/A when $N \le 1$ or values are identical.}",
        r"    \end{flushleft}",
        r"\end{table}",
    ])

    with open(tex_path, "w", encoding="utf-8") as f:
        f.write("\n".join(latex_str))

    print(f"Successfully generated LaTeX table at: {tex_path}")


def main() -> None:
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading EDA results from {EDA_H5_PATH}...")
    df, temp_profiles = load_eda_data()

    print("Generating EDA plots...")
    plot_creep_rate_vs_stress(df, PLOTS_DIR)
    plot_initial_strain_vs_stress(df, PLOTS_DIR)
    plot_temperature_profiles(temp_profiles, PLOTS_DIR)
    plot_eda_summary_2x2(df, PLOTS_DIR)
    
    print("Generating pairwise relationships plot...")
    style = EDAStyleConfig()
    out_pair = plot_pairwise_relationships(df, PLOTS_DIR, style)
    print(f"Successfully generated pairwise plot at: {out_pair}")

    print(f"All EDA plots saved to: {PLOTS_DIR.absolute()}")

    print("Generating LaTeX summary table...")
    generate_eda_latex_table(df, TABLES_DIR)


if __name__ == "__main__":
    main()