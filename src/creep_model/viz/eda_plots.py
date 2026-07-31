"""
Thesis-ready EDA plotting functions (docs/methodology.md Sec 1.2).

Each function takes the eda_summary DataFrame (as loaded from
tlv_fit_results.h5's eda_summary group -- see
scripts/generate_eda_plots.py) and an output directory, and saves ONE
publication-styled figure. Kept as pure functions (df in, file out, style in) 
so they're reusable from the runner script, a notebook, or future
diagnostics without re-deriving the EDA statistics each time.
"""
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.lines import Line2D


@dataclass
class EDAStyleConfig:
    """Central configuration for all EDA plot styling choices."""
    quality_colors: dict = field(default_factory=lambda: {"High": "#1f77b4", "Standard": "#d62728"})
    quality_markers: dict = field(default_factory=lambda: {"Standard": "o", "High": "s"})
    stress_markers: dict = field(default_factory=lambda: {10: "o", 20: "s", 30: "D"})
    quality_order: list = field(default_factory=lambda: ["Standard", "High"])
    stress_order: list = field(default_factory=lambda: [10.0, 20.0, 30.0])
    
    cmap: str = "coolwarm"
    dpi: int = 300
    base_marker_size: int = 75
    alpha: float = 0.9
    show_titles: bool = True


def _nominal_stress(df: pd.DataFrame) -> pd.Series:
    """Maps raw Applied_Stress_MPa to the nearest nominal band (10/20/30)."""
    return df["Applied_Stress_MPa"].apply(lambda v: 10 if v < 15 else (20 if v < 25 else 30))


def plot_eps_tilde_0_vs_stress(df: pd.DataFrame, output_dir: Path, style: EDAStyleConfig) -> Path:
    """eps~0 (instantaneous elastic strain) vs. applied stress."""
    output_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 5)) 
    
    temp_col = "Initial_Temp_C" 
    
    if temp_col in df.columns:
        vmin, vmax = df[temp_col].min(), df[temp_col].max()
    else:
        print(f"Warning: '{temp_col}' not found in DataFrame. Falling back to Mean_Temp_C_Secondary_Creep.")
        temp_col = "Mean_Temp_C_Secondary_Creep"
        vmin, vmax = df[temp_col].min(), df[temp_col].max()

    scatter_plots = []
    for quality in style.quality_order:
        sub = df[df["Print_Quality"] == quality]
        sc = ax.scatter(sub["Applied_Stress_MPa"], sub["Eps_Tilde_0"],
                        c=sub[temp_col], cmap=style.cmap, vmin=vmin, vmax=vmax,
                        marker=style.quality_markers[quality], label=quality,
                        s=style.base_marker_size, alpha=style.alpha, edgecolor="black", linewidth=0.7)
        if len(sub) > 0:
            scatter_plots.append(sc)
        
    ax.set_xlabel("Applied Stress (MPa)")
    ax.set_ylabel(r"$\tilde{\epsilon}_0$ (Initial Elastic Strain)")
    if style.show_titles:
        ax.set_title("Initial Strain vs. Applied Stress")
    ax.grid(True, linestyle="--", alpha=0.5)

    if scatter_plots:
        cbar = plt.colorbar(scatter_plots[0], ax=ax)
        cbar.set_label('Initial Temperature (°C)', fontsize=10)

    legend_elements = [
        Line2D([0], [0], marker=style.quality_markers[q], color='w', label=q,
               markerfacecolor='gray', markersize=9, markeredgecolor='black')
        for q in style.quality_order
    ]
    ax.legend(handles=legend_elements, title="Print Quality", loc="upper left")

    out_path = output_dir / "eps_tilde_0_vs_stress.png"
    fig.tight_layout()
    fig.savefig(out_path, dpi=style.dpi, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_eps_dot_ss_vs_stress(df: pd.DataFrame, output_dir: Path, style: EDAStyleConfig) -> Path:
    """eps_dot_ss (steady-state strain rate) vs. applied stress."""
    output_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 5))
    
    temp_col = "Mean_Temp_C_Secondary_Creep"
    vmin, vmax = df[temp_col].min(), df[temp_col].max()

    scatter_plots = []
    for quality in style.quality_order:
        sub = df[df["Print_Quality"] == quality]
        sc = ax.scatter(sub["Applied_Stress_MPa"], sub["Eps_Dot_Ss"],
                        c=sub[temp_col], cmap=style.cmap, vmin=vmin, vmax=vmax,
                        marker=style.quality_markers[quality], label=quality,
                        s=style.base_marker_size, alpha=style.alpha, edgecolor="black", linewidth=0.7)
        if len(sub) > 0:
            scatter_plots.append(sc)
        
    ax.set_xlabel("Applied Stress (MPa)")
    ax.set_ylabel(r"$\hat{\dot\epsilon}_{ss}$ (Steady-State Strain Rate, s$^{-1}$)")
    if style.show_titles:
        ax.set_title("Steady-State Strain Rate vs. Applied Stress")
    ax.grid(True, linestyle="--", alpha=0.5)

    if scatter_plots:
        cbar = plt.colorbar(scatter_plots[0], ax=ax)
        cbar.set_label('Mean Secondary Creep Temperature (°C)', fontsize=10)

    legend_elements = [
        Line2D([0], [0], marker=style.quality_markers[q], color='w', label=q,
               markerfacecolor='gray', markersize=9, markeredgecolor='black')
        for q in style.quality_order
    ]
    ax.legend(handles=legend_elements, title="Print Quality", loc="upper left")

    out_path = output_dir / "eps_dot_ss_vs_stress.png"
    fig.tight_layout()
    fig.savefig(out_path, dpi=style.dpi, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_eps_tilde_0_vs_age(df: pd.DataFrame, output_dir: Path, style: EDAStyleConfig) -> Path:
    """eps~0 vs. specimen age, bubble-sized by applied stress."""
    output_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 5))
    for quality in style.quality_order:
        sub = df[df["Print_Quality"] == quality]
        sizes = 20 + 6 * sub["Applied_Stress_MPa"]
        ax.scatter(sub["Age_Days"], sub["Eps_Tilde_0"], s=sizes,
                   color=style.quality_colors[quality], label=quality,
                   alpha=0.7, edgecolor="white")
    ax.set_xlabel("Specimen Age (days)")
    ax.set_ylabel(r"$\tilde{\epsilon}_0$ (Initial Elastic Strain)")
    if style.show_titles:
        ax.set_title("Initial Strain vs. Specimen Age (bubble size = applied stress)")
    ax.legend(title="Print Quality")
    ax.grid(True, linestyle="--", alpha=0.5)

    out_path = output_dir / "eps_tilde_0_vs_age.png"
    fig.savefig(out_path, dpi=style.dpi, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_eps_dot_ss_vs_temperature(df: pd.DataFrame, output_dir: Path, style: EDAStyleConfig) -> Path:
    """eps_dot_ss vs. mean secondary-creep temperature, bubble-sized by applied stress."""
    output_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 5))
    for quality in style.quality_order:
        sub = df[df["Print_Quality"] == quality]
        sizes = 20 + 6 * sub["Applied_Stress_MPa"]
        ax.scatter(sub["Mean_Temp_C_Secondary_Creep"], sub["Eps_Dot_Ss"], s=sizes,
                   color=style.quality_colors[quality], label=quality,
                   alpha=0.7, edgecolor="white")
    ax.set_xlabel("Mean Temperature During Secondary Creep (°C)")
    ax.set_ylabel(r"$\hat{\dot\epsilon}_{ss}$ (s$^{-1}$)")
    if style.show_titles:
        ax.set_title("Steady-State Strain Rate vs. Temperature (bubble size = applied stress)")
    ax.legend(title="Print Quality")
    ax.grid(True, linestyle="--", alpha=0.5)

    out_path = output_dir / "eps_dot_ss_vs_temperature.png"
    fig.savefig(out_path, dpi=style.dpi, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_distribution_by_quality(df: pd.DataFrame, output_dir: Path, style: EDAStyleConfig) -> Path:
    """Box+strip comparison of both EDA statistics, split by print quality."""
    output_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    for ax, col, ylabel in zip(
        axes,
        ["Eps_Tilde_0", "Eps_Dot_Ss"],
        [r"$\tilde{\epsilon}_0$", r"$\hat{\dot\epsilon}_{ss}$ (s$^{-1}$)"],
    ):
        sns.boxplot(data=df, x="Print_Quality", y=col, hue="Print_Quality", order=style.quality_order,
                    palette=style.quality_colors, ax=ax, showfliers=False, legend=False)
        
        sns.stripplot(data=df, x="Print_Quality", y=col, order=style.quality_order,
                      color="black", size=4, alpha=0.6, ax=ax)
        ax.set_xlabel("Print Quality")
        ax.set_ylabel(ylabel)
        ax.grid(True, linestyle="--", alpha=0.4, axis="y")

    if style.show_titles:
        fig.suptitle("Distribution of EDA Statistics by Print Quality")
    fig.tight_layout()

    out_path = output_dir / "eda_distribution_by_quality.png"
    fig.savefig(out_path, dpi=style.dpi, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_mean_eps_dot_ss_bar(df: pd.DataFrame, output_dir: Path, style: EDAStyleConfig) -> Path:
    """Mean +/- std of the steady-state strain rate, grouped by print quality and nominal stress band."""
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_df = df.copy()
    plot_df["Nominal_Stress"] = _nominal_stress(plot_df)

    summary = (
        plot_df.groupby(["Print_Quality", "Nominal_Stress"])["Eps_Dot_Ss"]
        .agg(["mean", "std", "count"])
        .reset_index()
    )

    fig, ax = plt.subplots(figsize=(7, 5))
    width = 0.35
    x = np.arange(len(style.stress_order))
    for i, quality in enumerate(style.quality_order):
        sub = summary[summary["Print_Quality"] == quality].set_index("Nominal_Stress")
        sub = sub.reindex(style.stress_order)
        offset = (i - 0.5) * width
        ax.bar(x + offset, sub["mean"], width, yerr=sub["std"],
               label=quality, color=style.quality_colors[quality], capsize=4, alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels([f"{s:.0f} MPa" for s in style.stress_order])
    ax.set_xlabel("Nominal Applied Stress")
    ax.set_ylabel(r"Mean $\hat{\dot\epsilon}_{ss}$ (s$^{-1}$) $\pm$ 1 std")
    if style.show_titles:
        ax.set_title("Steady-State Strain Rate by Stress Band and Print Quality")
    ax.legend(title="Print Quality")
    ax.grid(True, linestyle="--", alpha=0.4, axis="y")

    out_path = output_dir / "eda_mean_eps_dot_ss_bar.png"
    fig.savefig(out_path, dpi=style.dpi, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_pairwise_relationships(df: pd.DataFrame, output_dir: Path, style: EDAStyleConfig) -> Path:
    """Pairwise scatter/KDE grid across the numeric EDA columns."""
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_df = df.copy()
    plot_df["Nominal_Stress"] = _nominal_stress(plot_df)

    plot_vars = ["Applied_Stress_MPa", "Age_Days", "Mean_Temp_C_Secondary_Creep",
                 "Eps_Tilde_0", "Eps_Dot_Ss"]

    g = sns.PairGrid(
        plot_df, vars=plot_vars, hue="Print_Quality", hue_order=style.quality_order,
        palette=style.quality_colors,
    )
    g.map_offdiag(sns.scatterplot, style=plot_df["Nominal_Stress"],
                  markers=style.stress_markers, alpha=0.75, s=50)
    g.map_diag(sns.kdeplot, fill=True)
    g.add_legend(title="Print Quality")
    
    if style.show_titles:
        g.fig.suptitle("Pairwise EDA Relationships", y=1.02)

    out_path = output_dir / "eda_pairwise_relationships.png"
    g.fig.savefig(out_path, dpi=style.dpi, bbox_inches="tight")
    plt.close(g.fig)
    return out_path