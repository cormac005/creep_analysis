"""
Thesis-ready TLV plotting functions.

Strictly adhering to thesis figure styling requirements.
"""
from dataclasses import dataclass
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import numpy as np


@dataclass
class TLVStyleConfig:
    """Central configuration for TLV plot styling choices."""
    figsize_full: tuple = (6.2, 3.8)
    font_size_label: int = 9
    font_size_tick: int = 9
    font_size_legend: float = 9.0
    font_size_title: int = 9
    grid_color: str = "#CCCCCC"
    grid_alpha: float = 0.4
    dpi: int = 300


def plot_tlv_fit_summary(quality: str, grouped_data: dict, stresses: list, output_dir: Path, style: TLVStyleConfig) -> Path:
    """
    Generates a 1x3 subfigure grid for a given quality, comparing TLV fit to measured data across stresses.
    Strictly applies thesis formatting rules (0-origin, 15% headroom, 3% margin padding, subfigure tagging).
    """
    fig, axes = plt.subplots(1, 3, figsize=style.figsize_full, sharex=True, sharey=True)
    
    legend_handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="gray", markersize=5, label="Measured Data"),
        Line2D([0], [0], color="gray", linestyle="--", linewidth=1.8, label="TLV Fit"),
    ]
    
    palette = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    subplot_labels = ['(a)', '(b)', '(c)']
    
    all_x = []
    all_y = []
    
    for col_idx, stress in enumerate(stresses):
        ax = axes[col_idx]
        tests = grouped_data.get((quality, stress), [])
        
        ax.set_xlabel("Time (s)", fontsize=style.font_size_label)
        if col_idx == 0:
            ax.set_ylabel("Strain", fontsize=style.font_size_label, fontweight='bold')
            
        ax.grid(True, linestyle="--", alpha=style.grid_alpha, color=style.grid_color)
        ax.tick_params(axis='x', labelsize=style.font_size_tick)
        ax.tick_params(axis='y', labelsize=style.font_size_tick)
        
        # Subfigure tagging
        ax.set_title(f"{subplot_labels[col_idx]} {int(stress)} MPa", fontsize=style.font_size_label)
        
        for i, t in enumerate(tests):
            c = palette[i % len(palette)]
            
            # Experimental Data Points: discrete markers, no connecting lines
            ax.scatter(t["time_s"], t["strain_measured"], alpha=0.6, s=10, color=c, marker='o')
            
            # Analytical / Fitted Curves: continuous or dashed lines
            ax.plot(t["time_s"], t["strain_predicted"], linewidth=1.8, linestyle="--", color=c, alpha=0.8)
            
            legend_handles.append(mpatches.Patch(color=c, label=f"{t['test_id']} ({t['stress']:.2f} MPa)"))
            
            all_x.extend(t["time_s"])
            all_y.extend(t["strain_measured"])
            all_y.extend(t["strain_predicted"])

    # Compute and enforce global thesis limits across all shared axes
    if all_x and all_y:
        x_arr = np.array(all_x)
        y_arr = np.array(all_y)
        x_min, x_max = min(0, np.nanmin(x_arr)), np.nanmax(x_arr)
        y_min, y_max = min(0, np.nanmin(y_arr)), np.nanmax(y_arr)
        
        x_range = x_max - x_min if x_max > x_min else 1
        y_range = y_max - y_min if y_max > y_min else 1
        
        axes[0].set_xlim(left=x_min, right=x_max + 0.03 * x_range)
        axes[0].set_ylim(bottom=y_min, top=y_max + 0.15 * y_range)

    # Legend strictly outside to avoid overlapping data
    fig.legend(handles=legend_handles, loc="upper center", bbox_to_anchor=(0.5, 0.99),
               ncol=3, fontsize=style.font_size_legend, frameon=True)
    
    fig.tight_layout(rect=[0, 0, 1, 0.75])
    
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"Summary_{quality}_Combined.png"
    fig.savefig(out_path, dpi=style.dpi, bbox_inches="tight")
    plt.close(fig)
    return out_path