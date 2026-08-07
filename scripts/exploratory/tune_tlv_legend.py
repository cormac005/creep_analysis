"""
Fine-Grained Tuner for Ultra-Compact TLV Summary Plot Layouts.

Generates candidate PNGs in 'tune_output/' with sub-millimeter gap variations.
"""
from pathlib import Path
import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import numpy as np

FIT_H5_PATH = Path("data/processed/tlv_fit_results.h5")
OUTPUT_DIR = Path("tune_output")


def load_sample_grouped_data():
    grouped = {}
    if FIT_H5_PATH.exists():
        with h5py.File(FIT_H5_PATH, "r") as f:
            tests_group = f["tests"]
            for test_id in tests_group.keys():
                g = tests_group[test_id]
                quality = g.attrs["print_quality"]
                stress = float(g.attrs["applied_stress_MPa"])
                nominal = round(stress, -1)
                grouped.setdefault((quality, nominal), []).append({
                    "test_id": test_id,
                    "time_s": g["time_s"][:],
                    "strain_measured": g["strain_measured"][:],
                    "strain_predicted": g["strain_predicted"][:],
                    "stress": stress,
                })
    else:
        print("HDF5 file not found — using synthetic data for tuning.")
        t = np.linspace(0, 8000, 100)
        for q in ["Standard", "High"]:
            for s in [10.0, 20.0, 30.0]:
                for i in range(1, 5):
                    test_id = f"{q[0]}.{int(s)}.{i}"
                    base_strain = 0.01 * (s / 10.0) + (t / 8000.0) * 0.01 * (s / 10.0)
                    grouped.setdefault((q, s), []).append({
                        "test_id": test_id,
                        "time_s": t,
                        "strain_measured": base_strain + np.random.normal(0, 0.0005, len(t)),
                        "strain_predicted": base_strain,
                        "stress": s + (i * 0.05),
                    })
    return grouped


def generate_single_tuned_plot(
    quality: str,
    grouped_data: dict,
    fig_w: float,
    fig_h: float,
    rect_top: float,
    legend_y: float,
    font_size_legend: float,
    ncol: int,
    out_name: str,
):
    stresses = [10.0, 20.0, 30.0]
    fig, axes = plt.subplots(1, 3, figsize=(fig_w, fig_h), sharex=True, sharey=True)

    legend_handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="gray", markersize=4, label="Measured Data"),
        Line2D([0], [0], color="gray", linestyle="--", linewidth=1.5, label="TLV Fit"),
    ]

    palette = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    subplot_labels = ['(a)', '(b)', '(c)']
    all_x, all_y = [], []

    for col_idx, stress in enumerate(stresses):
        ax = axes[col_idx]
        tests = grouped_data.get((quality, stress), [])

        ax.set_xlabel("Time (s)", fontsize=11)
        if col_idx == 0:
            ax.set_ylabel("Strain", fontsize=11, fontweight="bold")

        ax.grid(True, linestyle="--", alpha=0.4, color="#CCCCCC")
        ax.tick_params(axis='both', labelsize=10)
        ax.set_title(f"{subplot_labels[col_idx]} {int(stress)} MPa", fontsize=11, pad=4)

        for i, t in enumerate(tests):
            c = palette[i % len(palette)]
            ax.scatter(t["time_s"], t["strain_measured"], alpha=0.6, s=8, color=c, marker='o')
            ax.plot(t["time_s"], t["strain_predicted"], linewidth=1.5, linestyle="--", color=c, alpha=0.8)
            legend_handles.append(mpatches.Patch(color=c, label=f"{t['test_id']} ({t['stress']:.2f} MPa)"))
            all_x.extend(t["time_s"])
            all_y.extend(t["strain_measured"])

    if all_x and all_y:
        x_arr, y_arr = np.array(all_x), np.array(all_y)
        x_min, x_max = min(0, np.nanmin(x_arr)), np.nanmax(x_arr)
        y_min, y_max = min(0, np.nanmin(y_arr)), np.nanmax(y_arr)
        x_range = x_max - x_min if x_max > x_min else 1
        y_range = y_max - y_min if y_max > y_min else 1

        axes[0].set_xlim(left=x_min, right=x_max + 0.03 * x_range)
        axes[0].set_ylim(bottom=y_min, top=y_max + 0.15 * y_range)

    fig.tight_layout(rect=[0, 0, 1, rect_top])

    fig.legend(
        handles=legend_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, legend_y),
        ncol=ncol,
        fontsize=font_size_legend,
        frameon=True,
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / f"{out_name}.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def main():
    grouped = load_sample_grouped_data()

    # Fine-grained ultra-compact presets around 3.6" - 3.8" height
    # (fig_h, rect_top, legend_y, font_size, ncol, label)
    presets = [
        (3.8, 0.81, 0.820, 8.0, 3, "H3.8_Gap_Minimal_0.01"),
        (3.8, 0.80, 0.810, 8.0, 3, "H3.8_Gap_Default_0.01"),
        (3.8, 0.79, 0.805, 8.0, 3, "H3.8_Gap_Slight_0.015"),
        (3.6, 0.79, 0.800, 7.5, 3, "H3.6_Font7.5_Tight"),
        (3.6, 0.78, 0.795, 8.0, 3, "H3.6_Font8.0_Tight"),
    ]

    print("Generating ultra-compact candidate layouts in 'tune_output/'...\n")
    for fig_h, rect_top, legend_y, font_sz, ncol, label in presets:
        generate_single_tuned_plot(
            quality="Standard",
            grouped_data=grouped,
            fig_w=6.2,
            fig_h=fig_h,
            rect_top=rect_top,
            legend_y=legend_y,
            font_size_legend=font_sz,
            ncol=ncol,
            out_name=label,
        )

    print("\nInspect 'tune_output/'. Once chosen, update 'tlv_plots.py'.")


if __name__ == "__main__":
    main()