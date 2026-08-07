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
        - stage_boundaries/creep_stages_<test_id>.png: Individual test stage boundary plots
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
from creep_model.viz.eda_plots import (
    EDAStyleConfig,
    plot_pairwise_relationships,
    plot_creep_stage_boundaries,
)

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
COLOR_HIGH = "#1f77b4"  
COLOR_STD = "#ff7f0e"  
COLOR_TEMP = "#d62728"  
GRID_COLOR = "#CCCCCC"

# Paths
DATA_EXCEL_PATH = Path(config.data_directory) / "CreepData.xlsx"
EDA_H5_PATH = Path(config.data_output_directory) / "eda_results.h5"
PROCESSED_H5_PATH = Path(config.data_output_directory) / "processed_experimental_data.h5"
PLOTS_DIR = Path(config.general_output_directory) / "plots" / "eda"
TABLES_DIR = Path(config.general_output_directory) / "tables"

# Stage Classification Thresholds
K1 = config.K1
K2 = config.K2


def _collect_all_test_groups(group):
    """Recursively collects test groups containing HDF5 Datasets."""
    test_dict = {}
    for key, item in group.items():
        if isinstance(item, h5py.Group):
            has_datasets = any(isinstance(v, h5py.Dataset) for v in item.values())
            if has_datasets:
                test_id = item.attrs.get("test_id", key)
                if isinstance(test_id, bytes):
                    test_id = test_id.decode("utf-8")
                test_dict[test_id] = item
            else:
                nested = _collect_all_test_groups(item)
                test_dict.update(nested)
    return test_dict


def _load_untrimmed_raw_tests() -> dict:
    """
    Loads raw untrimmed CreepTest objects directly from the raw Excel dataset via ExcelCreepParser.
    Falls back to processed_experimental_data.h5 if the Excel file is unavailable.
    """
    if DATA_EXCEL_PATH.exists():
        try:
            from creep_model.io.parser import ExcelCreepParser

            parser = ExcelCreepParser(DATA_EXCEL_PATH)
            experiment = parser.load_experiment()

            # Safely extract tests whether it's a dict or list
            if hasattr(experiment, "tests"):
                tests_attr = experiment.tests
            elif hasattr(experiment, "get_all_tests"):
                tests_attr = experiment.get_all_tests()
            else:
                tests_attr = []

            # If tests_attr is a dictionary, extract the values (the actual CreepTest objects)
            if isinstance(tests_attr, dict):
                tests_list = list(tests_attr.values())
            else:
                tests_list = list(tests_attr)

            raw_dict = {}
            for t in tests_list:
                tid = getattr(t, "test_id", getattr(t, "id", None))
                if tid is not None:
                    raw_dict[str(tid)] = t

            if raw_dict:
                return raw_dict
        except Exception as e:
            print(f"Notice: Could not parse raw Excel data directly ({e}). Falling back to processed HDF5.")

    # Fallback to processed_experimental_data.h5 if Excel file cannot be loaded
    if PROCESSED_H5_PATH.exists():
        from creep_model.domain import CreepTest
        test_dict = {}
        with h5py.File(PROCESSED_H5_PATH, "r") as f_proc:
            proc_tests = _collect_all_test_groups(f_proc)
            for tid, p_grp in proc_tests.items():
                time_series = p_grp["time_series"][:]
                strain_series = p_grp["strain_series"][:]
                temp_time = p_grp["temp_time_series"][:] if "temp_time_series" in p_grp else time_series
                temp_readings = p_grp["temperature_readings"][:] if "temperature_readings" in p_grp else np.full_like(time_series, 20.0)
                stress = float(p_grp.attrs.get("applied_stress_MPa", p_grp.attrs.get("Applied_Stress_MPa", 0.0)))
                age = int(p_grp.attrs.get("age_days", p_grp.attrs.get("Age_Days", 0)))
                quality = str(p_grp.attrs.get("print_quality", p_grp.attrs.get("Print_Quality", "High")))

                test_dict[tid] = CreepTest(
                    test_id=tid,
                    time_series=time_series,
                    strain_series=strain_series,
                    temp_time_series=temp_time,
                    temperature_readings=temp_readings,
                    applied_stress_MPa=stress,
                    age_days=age,
                    print_quality=quality,
                )
        return test_dict

    return {}


def load_eda_data() -> tuple[pd.DataFrame, dict]:
    """
    Loads consolidated EDA summary table and test-level temperature profiles
    from eda_results.h5, computes overall mean test temperatures, and resolves Has_Tertiary flags.
    """
    if not EDA_H5_PATH.exists():
        raise FileNotFoundError(
            f"Required data file {EDA_H5_PATH} does not exist. "
            "Please run 03_compute_eda_stats.py first."
        )

    temp_profiles = {}
    
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

        tertiary_map = {}
        if "tests" in f:
            tests_grp = f["tests"]
            for test_id in tests_grp.keys():
                t_grp = tests_grp[test_id]
                
                # Capture tertiary flag if stored on test attribute
                has_tert = t_grp.attrs.get("has_tertiary", t_grp.attrs.get("Has_Tertiary", None))
                if has_tert is not None:
                    tertiary_map[test_id] = bool(has_tert)

                entry = {}
                if "temp_time_s" in t_grp and "temperature_raw" in t_grp:
                    entry["temp_time_s"] = t_grp["temp_time_s"][:]
                    entry["temperature_raw"] = t_grp["temperature_raw"][:]
                if "time_s" in t_grp and "temperature_interpolated" in t_grp:
                    entry["time_s"] = t_grp["time_s"][:]
                    entry["temperature_interpolated"] = t_grp["temperature_interpolated"][:]
                if entry:
                    temp_profiles[test_id] = entry

        # Fallback: dynamically re-classify stage boundaries from untrimmed time series if Has_Tertiary is missing
        if "Has_Tertiary" not in df.columns and "has_tertiary" not in df.columns:
            if len(tertiary_map) < len(df):
                try:
                    from creep_model.eda.stage_classification import classify_stages

                    raw_tests = _load_untrimmed_raw_tests()
                    for tid, ctest in raw_tests.items():
                        if tid not in tertiary_map:
                            cls = classify_stages(ctest, k1=K1, k2=K2)
                            has_tert = getattr(
                                cls,
                                "has_tertiary",
                                (
                                    cls.primary_end_idx is not None
                                    and cls.secondary_end_idx is not None
                                    and len(cls.plateaus) >= 2
                                    and cls.secondary_end_idx < cls.plateaus[-2].end_idx
                                ),
                            )
                            tertiary_map[tid] = bool(has_tert)
                except Exception as e:
                    print(f"Notice: Fallback tertiary stage resolution skipped: {e}")

            has_tertiary_list = [tertiary_map.get(tid, False) for tid in df["Test_ID"]]
            df["Has_Tertiary"] = has_tertiary_list
        elif "has_tertiary" in df.columns and "Has_Tertiary" not in df.columns:
            df["Has_Tertiary"] = df["has_tertiary"].astype(bool)
        else:
            df["Has_Tertiary"] = df["Has_Tertiary"].astype(bool)

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
    
    handles, labels = ax.get_legend_handles_labels()
    if handles:
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

    handles, labels = ax.get_legend_handles_labels()
    if handles:
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

    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        axes[0, 0].legend(loc="upper left", fontsize=FONT_SIZE_LEGEND, framealpha=0.9)

    plt.tight_layout()
    plt.savefig(output_dir / "eda_summary_2x2.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_stage_boundaries_all_tests(output_dir: Path, style: EDAStyleConfig) -> None:
    """Generates creep stage boundary plots for all experimental tests using untrimmed test data."""
    stages_dir = output_dir / "stage_boundaries"
    stages_dir.mkdir(parents=True, exist_ok=True)

    try:
        raw_tests = _load_untrimmed_raw_tests()
        if not raw_tests:
            print("Notice: No test data found. Skipping stage boundary plots.")
            return

        for tid, ctest in raw_tests.items():
            plot_creep_stage_boundaries(
                test=ctest,
                k1=K1,
                k2=K2,
                output_dir=stages_dir,
                style=style,
            )
        print(f"Successfully generated stage boundary plots in: {stages_dir}")
    except Exception as e:
        print(f"Notice: Stage boundary plots generation skipped: {e}")


def generate_eda_latex_table(df: pd.DataFrame, output_dir: Path) -> None:
    """Generates a publication-ready LaTeX table with Range instead of Std. Dev. and explicit creep counts."""
    output_dir.mkdir(parents=True, exist_ok=True)
    tex_path = output_dir / "eda_summary_table.tex"

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
                n_tert = int(sub["Has_Tertiary"].astype(bool).sum())
            else:
                n_tert = 0

            # Calculate statistics for each metric
            t_init_m, t_init_r = calc_stats(sub.get("Initial_Temp_C", pd.Series(dtype=float)), scale=1.0, fmt="{:.1f}")
            
            temp_sec_col = "Mean_Temp_C" if "Mean_Temp_C" in sub.columns else "Mean_Temp_C_Secondary_Creep"
            t_sec_m, t_sec_r = calc_stats(sub.get(temp_sec_col, pd.Series(dtype=float)), scale=1.0, fmt="{:.1f}")
            
            eps0_m, eps0_r = calc_stats(sub.get("Eps_Tilde_0", pd.Series(dtype=float)), scale=1e3, fmt="{:.2f}")
            eps_ss_m, eps_ss_r = calc_stats(sub.get("Eps_Dot_Ss", pd.Series(dtype=float)), scale=1e7, fmt="{:.2f}", filter_positive=True)

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
    
    style = EDAStyleConfig()

    print("Generating stage boundary plots...")
    plot_stage_boundaries_all_tests(PLOTS_DIR, style)

    print("Generating pairwise relationships plot...")
    out_pair = plot_pairwise_relationships(df, PLOTS_DIR, style)
    print(f"Successfully generated pairwise plot at: {out_pair}")

    print(f"All EDA plots saved to: {PLOTS_DIR.absolute()}")

    print("Generating LaTeX summary table...")
    generate_eda_latex_table(df, TABLES_DIR)


if __name__ == "__main__":
    main()