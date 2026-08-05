"""
Generate publication-quality EDA figures and LaTeX summary tables from eda_results.h5.
This script should be run AFTER 03_compute_eda_stats.py has been executed.

Outputs generated:
    1. Plots (saved to <general_output_directory>/plots/eda/):
        - creep_rate_vs_stress.png: Secondary creep strain rate vs. applied stress
        - initial_strain_vs_stress.png: Initial strain intercept vs. applied stress
        - temperature_profiles.png: Raw vs interpolated thermal history during testing
        - eda_summary_2x2.png: Consolidated 2x2 grid comparing key EDA parameters
    2. Tables (saved to <general_output_directory>/tables/):
        - eda_summary_table.tex: LaTeX table of mean ± std EDA stats per quality & stress level
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
    Loads the consolidated EDA summary table and test-level temperature profiles
    from eda_results.h5.
    """
    if not EDA_H5_PATH.exists():
        raise FileNotFoundError(
            f"Required data file {EDA_H5_PATH} does not exist. "
            "Please run 03_compute_eda_stats.py first."
        )

    with h5py.File(EDA_H5_PATH, "r") as f:
        # Load consolidated EDA summary table
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

        # Load per-test raw & interpolated temperature series
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

    return df, temp_profiles


def plot_creep_rate_vs_stress(df: pd.DataFrame, output_dir: Path) -> None:
    """Plots secondary creep rate (Eps_Dot_Ss) vs applied stress on log scale."""
    fig, ax = plt.subplots(figsize=FIG_SIZE_SINGLE)
    ax.grid(True, color=GRID_COLOR, linestyle="--", alpha=0.5, which="both")

    qualities = [("High", COLOR_HIGH, "o"), ("Standard", COLOR_STD, "s")]

    for q_name, color, marker in qualities:
        sub = df[df["Print_Quality"] == q_name].sort_values("Applied_Stress_MPa")
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

        # Compute mean per stress level to draw connecting line
        means = sub.groupby("Nominal_Stress_MPa", as_index=False)[
            ["Applied_Stress_MPa", "Eps_Dot_Ss"]
        ].mean()
        ax.plot(
            means["Applied_Stress_MPa"],
            means["Eps_Dot_Ss"],
            color=color,
            linestyle="--",
            linewidth=1.5,
            alpha=0.7,
        )

    ax.set_yscale("log")
    ax.set_xlabel("Applied Stress (MPa)", fontsize=FONT_SIZE_LABEL, fontweight="bold")
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

    # Enforce zero-origin for strain axis
    y_max = df["Eps_Tilde_0"].max()
    ax.set_ylim(bottom=0, top=y_max * 1.15)

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

    # Select up to 3 representative test IDs for clarity
    sample_test_ids = list(temp_profiles.keys())[:3]
    linestyles = ["-", "--", "-."]

    for idx, test_id in enumerate(sample_test_ids):
        p_data = temp_profiles[test_id]
        ls = linestyles[idx % len(linestyles)]

        # Raw Discrete Readings
        if "temp_time_s" in p_data and "temperature_raw" in p_data:
            ax.scatter(
                p_data["temp_time_s"],
                p_data["temperature_raw"],
                s=15,
                alpha=0.5,
                label=f"{test_id} (Raw)",
            )

        # Smooth Interpolated Curve
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
        ("Mean_Temp_C_Secondary_Creep", r"Secondary Temp ($^\circ\text{C}$)", False),
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

            ax.scatter(
                sub["Applied_Stress_MPa"],
                sub[col_name],
                color=color,
                marker=marker,
                s=30,
                alpha=0.7,
                label=f"{q_name}" if (row_idx == 0 and col_idx == 0) else "",
            )

            means = sub.groupby("Nominal_Stress_MPa", as_index=False)[
                ["Applied_Stress_MPa", col_name]
            ].mean()
            ax.plot(
                means["Applied_Stress_MPa"],
                means[col_name],
                color=color,
                linestyle="--",
                linewidth=1.2,
                alpha=0.7,
            )

        if use_log:
            ax.set_yscale("log")

        ax.set_ylabel(label, fontsize=FONT_SIZE_LABEL)
        ax.tick_params(axis="both", labelsize=FONT_SIZE_TICK)

        if row_idx == 1:
            ax.set_xlabel("Applied Stress (MPa)", fontsize=FONT_SIZE_LABEL)

    # Single Legend for top-left axis
    axes[0, 0].legend(loc="upper left", fontsize=FONT_SIZE_LEGEND, framealpha=0.9)

    plt.tight_layout()
    plt.savefig(output_dir / "eda_summary_2x2.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def generate_eda_latex_table(df: pd.DataFrame, output_dir: Path) -> None:
    """Generates a publication-ready LaTeX table summarizing EDA parameters."""
    output_dir.mkdir(parents=True, exist_ok=True)
    tex_path = output_dir / "eda_summary_table.tex"

    # Aggregate metrics by Print Quality and Nominal Stress
    agg_df = (
        df.groupby(["Print_Quality", "Nominal_Stress_MPa"])
        .agg(
            n=("Test_ID", "count"),
            age_mean=("Age_Days", "mean"),
            age_std=("Age_Days", "std"),
            temp_init_mean=("Initial_Temp_C", "mean"),
            temp_init_std=("Initial_Temp_C", "std"),
            temp_sec_mean=("Mean_Temp_C_Secondary_Creep", "mean"),
            temp_sec_std=("Mean_Temp_C_Secondary_Creep", "std"),
            eps0_mean=("Eps_Tilde_0", "mean"),
            eps0_std=("Eps_Tilde_0", "std"),
            eps_ss_mean=("Eps_Dot_Ss", "mean"),
            eps_ss_std=("Eps_Dot_Ss", "std"),
        )
        .reset_index()
    )

    latex_str = [
        r"\begin{table}[htbp]",
        r"    \centering",
        r"    \caption{Exploratory Data Analysis (EDA) statistics for creep tests grouped by print quality and stress level.}",
        r"    \label{tab:eda_summary}",
        r"    \begin{tabular}{ccccccc}",
        r"        \toprule",
        r"        \textbf{Quality} & \textbf{$\sigma$ (MPa)} & \textbf{$N$} & \textbf{Initial Temp ($^\circ$C)} & \textbf{Secondary Temp ($^\circ$C)} & \textbf{$\tilde{\varepsilon}_0$ ($\times 10^{-3}$)} & \textbf{$\dot{\varepsilon}_{ss}$ ($\times 10^{-7} \text{s}^{-1}$)} \\",
        r"        \midrule",
    ]

    for _, row in agg_df.iterrows():
        q_name = row["Print_Quality"]
        stress = int(row["Nominal_Stress_MPa"])
        n = int(row["n"])

        t_init = f"{row['temp_init_mean']:.1f} \\pm {row['temp_init_std']:.1f}" if pd.notnull(row["temp_init_std"]) else f"{row['temp_init_mean']:.1f}"
        t_sec = f"{row['temp_sec_mean']:.1f} \\pm {row['temp_sec_std']:.1f}" if pd.notnull(row["temp_sec_std"]) else f"{row['temp_sec_mean']:.1f}"

        # Scaled values for LaTeX readability
        eps0_m, eps0_s = row["eps0_mean"] * 1e3, (row["eps0_std"] * 1e3 if pd.notnull(row["eps0_std"]) else 0.0)
        eps0_str = f"{eps0_m:.2f} \\pm {eps0_s:.2f}"

        eps_ss_m, eps_ss_s = row["eps_ss_mean"] * 1e7, (row["eps_ss_std"] * 1e7 if pd.notnull(row["eps_ss_std"]) else 0.0)
        eps_ss_str = f"{eps_ss_m:.2f} \\pm {eps_ss_s:.2f}"

        latex_str.append(
            f"        {q_name} & {stress} & {n} & ${t_init}$ & ${t_sec}$ & ${eps0_str}$ & ${eps_ss_str}$ \\\\"
        )

    latex_str.extend([
        r"        \bottomrule",
        r"    \end{tabular}",
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
    print(f"All EDA plots saved to: {PLOTS_DIR.absolute()}")

    print("Generating LaTeX summary table...")
    generate_eda_latex_table(df, TABLES_DIR)


if __name__ == "__main__":
    main()