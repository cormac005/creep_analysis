"""
Generate plot figures (strain vs. time) and LaTeX parameter summary tables from tlv_fit_results.h5.
This script should be run AFTER 02_fit_tlv.py has been executed.
"""
from pathlib import Path

import h5py
import matplotlib

# Use the 'Agg' backend to completely disable interactive plot windows
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

from creep_model.config import config

# --- CONFIGURATION & TYPOGRAPHY STANDARDS ---
SHOW_TITLE = False

# Canvas Sizing (Inches) - Taller heights allocated to accommodate top legends
FIG_SIZE_SINGLE = (6.2, 4.0)  # Full-width individual plot
FIG_SIZE_1X3 = (8.5, 5.2)  # Multi-panel row figure (taller to fit top legend)
FIG_SIZE_2X3 = (8.5, 6.6)  # Consolidated 2x3 summary figure

# Font Size Hierarchy
FONT_SIZE_LABEL = 11
FONT_SIZE_TICK = 10
FONT_SIZE_LEGEND = 9.5
FONT_SIZE_TITLE = 11
FONT_SIZE_ANNOT = 9

# Color Palette & Styles
COLOR_PRIMARY = "#1f77b4"  # Strain / Main Data
COLOR_SECONDARY = "#d62728"  # Raw Temperature
COLOR_FIT = "black"  # Model Fits
GRID_COLOR = "#CCCCCC"

# File Paths
H5_PATH = Path(config.data_output_directory) / "tlv_fit_results.h5"
PLOTS_DIR = Path(config.general_output_directory) / "plots" / "eps_v_time"
OUTPUT_TEX = Path(config.general_output_directory) / "tables" / "fitted_tlv_params.tex"


def apply_shared_thesis_limits(axes, all_x_data, all_y_data):
    """
    Computes global limits across all shared axes to enforce zero-origins, 
    15% top headroom, and 3% right margin across the entire subplot grid.
    """
    x_valid = all_x_data[~np.isnan(all_x_data)] if len(all_x_data) > 0 else np.array([])
    y_valid = all_y_data[~np.isnan(all_y_data)] if len(all_y_data) > 0 else np.array([])

    if len(x_valid) > 0 and len(y_valid) > 0:
        x_min, x_max = min(0, np.min(x_valid)), np.max(x_valid)
        x_range = x_max - x_min if x_max > x_min else 1.0

        y_min, y_max = min(0, np.min(y_valid)), np.max(y_valid)
        y_range = y_max - y_min if y_max > y_min else 1.0

        for ax in np.ravel(axes):
            ax.set_xlim(left=x_min, right=x_max + 0.03 * x_range)
            ax.set_ylim(bottom=y_min, top=y_max + 0.15 * y_range)


def _collect_test_groups(group):
    """
    Recursively collects test groups containing measured and predicted strain data,
    supporting both flat (/tests/<test_id>) and nested (/tests/<quality>/<test_id>) HDF5 structures.
    """
    test_nodes = []
    for key, item in group.items():
        if isinstance(item, h5py.Group):
            if "strain_predicted" in item and "strain_measured" in item:
                test_nodes.append((key, item))
            else:
                test_nodes.extend(_collect_test_groups(item))
    return test_nodes


def generate_parameter_table():
    """Reads fitted TLV parameters from HDF5 and formats them into a LaTeX table."""
    if not H5_PATH.exists():
        raise FileNotFoundError(f"Cannot generate table: {H5_PATH} does not exist.")

    OUTPUT_TEX.parent.mkdir(parents=True, exist_ok=True)

    # 1. Read parameters from the HDF5 file
    with h5py.File(H5_PATH, "r") as f:
        if "fitted_parameters" not in f:
            print("Warning: 'fitted_parameters' group not found in HDF5 file.")
            return

        high = dict(f["fitted_parameters"]["High"].attrs) if "High" in f["fitted_parameters"] else {}
        std = dict(f["fitted_parameters"]["Standard"].attrs) if "Standard" in f["fitted_parameters"] else {}

    if not high and not std:
        print("Warning: No fitted parameters available for LaTeX table generation.")
        return

    # 2. Define the table rows, formatting, and mathematical symbols
    rows = [
        (r"$A_{20}$", r"$\text{MPa}^{-n}\cdot\text{s}^{-(m+1)}$", "A20", "{:.2e}"),
        (r"$A_{30}$", r"$\text{MPa}^{-n}\cdot\text{s}^{-(m+1)}$", "A30", "{:.2e}"),
        (r"$n_{20}$", r"--", "n20", "{:.3f}"),
        (r"$n_{30}$", r"--", "n30", "{:.3f}"),
        (r"$m_{20}$", r"--", "m20", "{:.3f}"),
        (r"$m_{30}$", r"--", "m30", "{:.3f}"),
        (r"$E_{e,20}$", r"MPa", "Ee20", "{:.1f}"),
        (r"$E_{e,30}$", r"MPa", "Ee30", "{:.1f}"),
        (r"$E_{v,20}$", r"MPa", "Ev20", "{:.1f}"),
        (r"$E_{v,30}$", r"MPa", "Ev30", "{:.1f}"),
    ]

    # 3. Construct the LaTeX string
    latex_str = [
        r"\begin{table}[htbp]",
        r"    \centering",
        r"    \caption{Fitted TLV model parameters for High and Standard print qualities.}",
        r"    \label{tab:fitted_tlv_params}",
        r"    \begin{tabular}{llcc}",
        r"        \toprule",
        r"        \textbf{Parameter} & \textbf{Units} & \textbf{High Quality} & \textbf{Standard Quality} \\",
        r"        \midrule",
    ]

    for symbol, units, key, fmt in rows:
        val_high = fmt.format(high[key]) if key in high else "N/A"
        val_std = fmt.format(std[key]) if key in std else "N/A"

        # Convert Python scientific notation (e.g. 1.23e-04) to LaTeX ($1.23 \times 10^{-4}$)
        if "e" in val_high:
            val_high = val_high.replace("e", r" \times 10^{").replace("+0", "").replace("+", "").replace("-0", "-") + "}"
            val_high = f"${val_high}$"
        if "e" in val_std:
            val_std = val_std.replace("e", r" \times 10^{").replace("+0", "").replace("+", "").replace("-0", "-") + "}"
            val_std = f"${val_std}$"

        latex_str.append(f"        {symbol} & {units} & {val_high} & {val_std} \\\\")

    latex_str.extend([
        r"        \bottomrule",
        r"    \end{tabular}",
        r"\end{table}",
    ])

    # 4. Save to file
    with open(OUTPUT_TEX, "w", encoding="utf-8") as f:
        f.write("\n".join(latex_str))

    print(f"LaTeX table successfully generated at {OUTPUT_TEX}")


def main():
    if not H5_PATH.exists():
        raise FileNotFoundError(
            f"Input file {H5_PATH} does not exist. Please run 02_fit_tlv.py first."
        )

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    grouped_tests = {}

    print(f"Reading processed data and temperatures directly from {H5_PATH}...")
    with h5py.File(H5_PATH, "r") as f:
        if "tests" not in f:
            raise KeyError(f"'tests' group not found in {H5_PATH}")

        test_items = _collect_test_groups(f["tests"])

        for test_id, test_group in test_items:
            attrs = test_group.attrs

            print_quality = attrs.get("print_quality")
            if isinstance(print_quality, bytes):
                print_quality = print_quality.decode("utf-8")

            raw_stress = attrs.get("applied_stress_MPa")
            if print_quality is None or raw_stress is None:
                continue

            actual_stress = float(raw_stress)
            nominal_stress = float(round(actual_stress, -1))

            key = (print_quality, nominal_stress)
            if key not in grouped_tests:
                grouped_tests[key] = []

            time_s = test_group["time_s"][:]
            strain_measured = test_group["strain_measured"][:]
            strain_predicted = test_group["strain_predicted"][:]

            if "temperature_interpolated" in test_group:
                temp_array = test_group["temperature_interpolated"][:]
            else:
                temp_array = None

            mean_t = attrs.get("mean_temp_c_secondary_creep", None)
            if (mean_t is None or np.isnan(mean_t)) and temp_array is not None:
                mean_t = np.nanmean(temp_array)

            grouped_tests[key].append({
                "test_id": test_id,
                "time_s": time_s,
                "strain_measured": strain_measured,
                "strain_predicted": strain_predicted,
                "mean_temp": mean_t,
                "temp_array": temp_array,
                "actual_stress": actual_stress,
                "nominal_stress": nominal_stress,
            })

    row_order = ["Standard", "High"]
    col_order = [10.0, 20.0, 30.0]
    palette_colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    # =========================================================================
    # 1. Space-Saving Summary Plots: 1x3 Subplots per Print Quality
    # =========================================================================
    print("Generating non-overlapping 1x3 summary plots...")

    for print_quality in row_order:
        fig, axes = plt.subplots(1, 3, figsize=FIG_SIZE_1X3, sharex=True, sharey=True)
        all_x, all_y = [], []
        legend_handles = []

        # Generic style handles for data types
        legend_handles.append(
            Line2D([0], [0], marker="o", color="w", markerfacecolor="gray", markersize=5, label="Measured Data")
        )
        legend_handles.append(
            Line2D([0], [0], color="gray", linestyle="--", linewidth=1.8, label="TLV Fit")
        )

        for col_idx, stress in enumerate(col_order):
            ax = axes[col_idx]
            key = (print_quality, stress)
            tests = grouped_tests.get(key, [])

            ax.grid(True, color=GRID_COLOR, linestyle="--", alpha=0.4)
            ax.tick_params(axis="both", labelsize=FONT_SIZE_TICK)

            # Subpanel Header Annotation
            ax.text(
                0.04, 0.90, f"{int(stress)} MPa Nominal", transform=ax.transAxes,
                fontsize=FONT_SIZE_ANNOT, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.85, edgecolor="#CCCCCC"),
            )

            for i, test in enumerate(tests):
                c = palette_colors[i % len(palette_colors)]
                rep_num = test["test_id"].split(".")[-1]
                t_str = f"{test['mean_temp']:.1f} °C" if test["mean_temp"] is not None else "N/A"
                actual_sigma_str = f"{test['actual_stress']:.2f} MPa"

                ax.scatter(test["time_s"], test["strain_measured"], alpha=0.6, s=10, color=c, marker="o")
                ax.plot(test["time_s"], test["strain_predicted"], linewidth=1.8, linestyle="--", color=c, alpha=0.8)

                legend_handles.append(
                    mpatches.Patch(color=c, label=f"{int(stress)} MPa Rep {rep_num} ({actual_sigma_str}, {t_str})")
                )

                all_x.extend(test["time_s"])
                all_y.extend(test["strain_measured"])
                all_y.extend(test["strain_predicted"])

            ax.set_xlabel("Time (s)", fontsize=FONT_SIZE_LABEL)

        axes[0].set_ylabel("Strain", fontsize=FONT_SIZE_LABEL, fontweight="bold")
        apply_shared_thesis_limits(axes, np.asarray(all_x), np.asarray(all_y))

        plt.tight_layout(rect=[0, 0, 1, 0.68])
        fig.legend(
            handles=legend_handles, loc="upper center", bbox_to_anchor=(0.5, 0.99),
            ncol=3, fontsize=FONT_SIZE_LEGEND, frameon=True, framealpha=0.9,
        )

        out_name = f"Summary_{print_quality}_Combined.png"
        plt.savefig(PLOTS_DIR / out_name, dpi=300, bbox_inches="tight")
        plt.close(fig)

    # =========================================================================
    # 2. Consolidated 2x3 Grid Summary Plot
    # =========================================================================
    print("Generating non-overlapping 2x3 grid summary plot...")

    fig, axes = plt.subplots(2, 3, figsize=FIG_SIZE_2X3, sharex=True, sharey=True)
    all_x_grid, all_y_grid = [], []
    grid_legend_elements = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="gray", markersize=5, label="Measured Data"),
        Line2D([0], [0], color="gray", linestyle="--", linewidth=1.8, label="TLV Fit"),
    ]

    for row_idx, print_quality in enumerate(row_order):
        for col_idx, stress in enumerate(col_order):
            ax = axes[row_idx, col_idx]
            key = (print_quality, stress)
            tests = grouped_tests.get(key, [])

            ax.grid(True, color=GRID_COLOR, linestyle="--", alpha=0.4)
            ax.tick_params(axis="both", labelsize=FONT_SIZE_TICK)

            ax.text(
                0.04, 0.90, f"{print_quality} | {int(stress)} MPa", transform=ax.transAxes,
                fontsize=FONT_SIZE_ANNOT, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.25", facecolor="white", alpha=0.85, edgecolor="#CCCCCC"),
            )

            for i, test in enumerate(tests):
                c = palette_colors[i % len(palette_colors)]
                ax.scatter(test["time_s"], test["strain_measured"], alpha=0.6, s=10, color=c, marker="o")
                ax.plot(test["time_s"], test["strain_predicted"], linewidth=1.8, linestyle="--", color=c, alpha=0.8)

                all_x_grid.extend(test["time_s"])
                all_y_grid.extend(test["strain_measured"])
                all_y_grid.extend(test["strain_predicted"])

            if row_idx == 1:
                ax.set_xlabel("Time (s)", fontsize=FONT_SIZE_LABEL)

        axes[row_idx, 0].set_ylabel("Strain", fontsize=FONT_SIZE_LABEL, fontweight="bold")

    apply_shared_thesis_limits(axes, np.asarray(all_x_grid), np.asarray(all_y_grid))
    plt.tight_layout(rect=[0, 0, 1, 0.92])

    fig.legend(
        handles=grid_legend_elements, loc="upper center", bbox_to_anchor=(0.5, 0.99),
        ncol=2, fontsize=FONT_SIZE_LEGEND, frameon=True, framealpha=0.9,
    )

    plt.savefig(PLOTS_DIR / "Summary_All_Qualities_2x3.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    # =========================================================================
    # 3. Individual Test Plots
    # =========================================================================
    print("Generating strictly-styled individual test plots...")

    for key, tests in grouped_tests.items():
        print_quality, nominal_stress = key

        for test in tests:
            fig, ax1 = plt.subplots(figsize=FIG_SIZE_SINGLE)
            test_id = test["test_id"]
            actual_stress = test["actual_stress"]

            mean_t = test["mean_temp"]
            avg_temp_str = f"{mean_t:.1f} °C" if mean_t is not None else "N/A"
            stress_str = f"{actual_stress:.2f} MPa"

            ax1.set_xlabel("Time (s)", fontsize=FONT_SIZE_LABEL)
            ax1.set_ylabel("Strain", color=COLOR_PRIMARY, fontsize=FONT_SIZE_LABEL, fontweight="bold")

            ax1.scatter(
                test["time_s"], test["strain_measured"],
                color=COLOR_PRIMARY, alpha=0.6, s=10, marker="o", label="Measured Strain",
            )
            ax1.plot(
                test["time_s"], test["strain_predicted"],
                color=COLOR_FIT, linewidth=1.8, linestyle="--", alpha=0.8,
                label=f"TLV Prediction ({stress_str}, Avg T: {avg_temp_str})",
            )

            ax1.tick_params(axis="x", labelsize=FONT_SIZE_TICK)
            ax1.tick_params(axis="y", labelcolor=COLOR_PRIMARY, labelsize=FONT_SIZE_TICK)
            ax1.grid(True, color=GRID_COLOR, linestyle="--", alpha=0.4)

            all_y1 = np.concatenate([test["strain_measured"], test["strain_predicted"]])
            apply_shared_thesis_limits([ax1], test["time_s"], all_y1)

            ax2 = None
            if test["temp_array"] is not None:
                ax2 = ax1.twinx()
                ax2.set_ylabel("Temperature (°C)", color=COLOR_SECONDARY, fontsize=FONT_SIZE_LABEL, fontweight="bold")
                ax2.plot(
                    test["time_s"], test["temp_array"],
                    color=COLOR_SECONDARY, linewidth=1.2, alpha=0.7, label="Interpolated Temp.",
                )
                ax2.tick_params(axis="y", labelcolor=COLOR_SECONDARY, labelsize=FONT_SIZE_TICK)

                y2_valid = test["temp_array"][~np.isnan(test["temp_array"])]
                if len(y2_valid) > 0:
                    y2_min = min(0, np.min(y2_valid))
                    y2_max = np.max(y2_valid)
                    y2_range = y2_max - y2_min if y2_max > y2_min else 1.0
                    ax2.set_ylim(bottom=y2_min, top=y2_max + 0.15 * y2_range)

            if SHOW_TITLE:
                ax1.set_title(
                    f"Test: {test_id} ({print_quality}, {actual_stress:.2f} MPa)",
                    fontsize=FONT_SIZE_TITLE, fontweight="bold",
                )

            lines1, labels1 = ax1.get_legend_handles_labels()
            if ax2 is not None:
                lines2, labels2 = ax2.get_legend_handles_labels()
                ax1.legend(lines1 + lines2, labels1 + labels2, loc="lower right", fontsize=FONT_SIZE_LEGEND, framealpha=0.9)
            else:
                ax1.legend(lines1, labels1, loc="lower right", fontsize=FONT_SIZE_LEGEND, framealpha=0.9)

            plt.tight_layout()
            indiv_save_path = PLOTS_DIR / f"{test_id}_Plot.png"
            plt.savefig(indiv_save_path, dpi=300, bbox_inches="tight")
            plt.close(fig)

    print(f"All updated non-overlapping plots saved to: {PLOTS_DIR.absolute()}")

    # 4. Generate LaTeX summary table
    print("Generating LaTeX parameter summary table...")
    generate_parameter_table()


if __name__ == "__main__":
    main()