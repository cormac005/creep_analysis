"""
Thesis-ready EDA plotting functions (docs/methodology.md Sec 1.2).

Each function takes the eda_summary DataFrame and an output directory,
and saves publication-styled figures.
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
    base_marker_size: int = 40
    alpha: float = 0.7
    show_titles: bool = False
    
    # Thesis Style Specifications
    figsize_full: tuple = (6.2, 3.8)
    figsize_side: tuple = (4.2, 3.0)
    font_size_label: int = 11
    font_size_tick: int = 10
    font_size_legend: float = 9.5
    font_size_annot: int = 9
    font_size_title: int = 11
    grid_color: str = "#CCCCCC"
    grid_alpha: float = 0.4
    boundary_color: str = "#d62728"
    fit_color: str = "black"


def _apply_thesis_style(ax, style: EDAStyleConfig, x_data=None, y_data=None, is_categorical_x=False):
    """Applies universal thesis plotting standards to a given matplotlib axis."""
    ax.tick_params(axis='x', labelsize=style.font_size_tick)
    ax.tick_params(axis='y', labelsize=style.font_size_tick)
    ax.grid(True, color=style.grid_color, linestyle='--', alpha=style.grid_alpha)
    
    y_label = ax.get_ylabel()
    if y_label:
        ax.set_ylabel(y_label, fontsize=style.font_size_label, fontweight='bold')
    
    x_label = ax.get_xlabel()
    if x_label:
        ax.set_xlabel(x_label, fontsize=style.font_size_label)
        
    if style.show_titles and ax.get_title():
        ax.set_title(ax.get_title(), fontsize=style.font_size_title, fontweight='bold')
    else:
        ax.set_title("")

    if x_data is not None and not is_categorical_x:
        x_arr = np.asarray(x_data, dtype=float)
        x_arr = x_arr[~np.isnan(x_arr)]
        if len(x_arr) > 0:
            x_min, x_max = min(0, np.min(x_arr)), np.max(x_arr)
            x_range = x_max - x_min if x_max > x_min else 1
            ax.set_xlim(left=x_min, right=x_max + 0.03 * x_range)
            
    if y_data is not None:
        y_arr = np.asarray(y_data, dtype=float)
        y_arr = y_arr[~np.isnan(y_arr)]
        if len(y_arr) > 0:
            y_min, y_max = min(0, np.min(y_arr)), np.max(y_arr)
            y_range = y_max - y_min if y_max > y_min else 1
            ax.set_ylim(bottom=y_min, top=y_max + 0.15 * y_range)


def _nominal_stress(df: pd.DataFrame) -> pd.Series:
    """Maps raw Applied_Stress_MPa to the nearest nominal band (10/20/30)."""
    return df["Applied_Stress_MPa"].apply(lambda v: 10 if v < 15 else (20 if v < 25 else 30))


def plot_creep_stage_boundaries(X_raw, y_raw, test_id: str, output_dir: Path, style: EDAStyleConfig) -> Path:
    """Visualizes boundaries of each creep stage calculated via raw data algorithm."""
    X = np.asarray(X_raw).flatten()
    y = np.asarray(y_raw).flatten()
    
    output_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=style.figsize_full)
    
    ax.scatter(X, y, color=style.quality_colors["High"], s=10, alpha=0.6, marker='o', label="Measured Strain")
    
    unique_strains, counts = np.unique(y, return_counts=True)
    unique_strains = unique_strains[:-1]
    counts = counts[:-1]
    diffs = np.diff(counts)
    
    is_decreasing = (diffs <= 0)
    secondary_trigger = np.where(is_decreasing[:-2] & is_decreasing[1:-1] & is_decreasing[2:])[0]
    
    y_max = np.max(y)
    y_annot = y_max + 0.07 * (y_max - min(0, np.min(y)))
    
    if len(secondary_trigger) > 0:
        first_dec_idx = secondary_trigger[0]
        strain_start = float(unique_strains[first_dec_idx])
        matching_start_indices = np.where(y == strain_start)[0]
        time_start = float(X[matching_start_indices[0]])
        
        ax.axvline(x=time_start, color=style.boundary_color, linestyle='--', linewidth=1.5, alpha=0.8)
        
        plateau_value = counts[first_dec_idx]
        is_tertiary_drop = (counts[first_dec_idx:] < plateau_value)
        tertiary_trigger = np.where(is_tertiary_drop[:-2] & is_tertiary_drop[1:-1] & is_tertiary_drop[2:])[0]
        
        if len(tertiary_trigger) > 0:
            end_idx = first_dec_idx + tertiary_trigger[0]
            strain_end = float(unique_strains[end_idx])
            matching_end_indices = np.where(y == strain_end)[0]
            time_end = float(X[matching_end_indices[0]])
            
            ax.axvline(x=time_end, color=style.boundary_color, linestyle='--', linewidth=1.5, alpha=0.8)
            
            ax.text(time_start / 2, y_annot, "Primary", ha='center', fontsize=style.font_size_annot, style='italic')
            ax.text((time_start + time_end) / 2, y_annot, "Secondary", ha='center', fontsize=style.font_size_annot, style='italic')
            ax.text(time_end + (np.max(X) - time_end) / 2, y_annot, "Tertiary", ha='center', fontsize=style.font_size_annot, style='italic')
            
            delta_strain = strain_end - strain_start
            delta_time = time_end - time_start
            if delta_time != 0:
                rate = delta_strain / delta_time
                line_x = np.array([time_start, time_end])
                line_y = strain_start + rate * (line_x - time_start)
                ax.plot(line_x, line_y, color=style.fit_color, linewidth=1.8, linestyle='--', label=f"Secondary Fit (Rate: {rate:.2e})")
                
        else:
            strain_end = float(unique_strains[-1])
            matching_end_indices = np.where(y == strain_end)[0]
            time_end = float(X[matching_end_indices[0]])
            
            ax.text(time_start / 2, y_annot, "Primary", ha='center', fontsize=style.font_size_annot, style='italic')
            ax.text((time_start + np.max(X)) / 2, y_annot, "Secondary", ha='center', fontsize=style.font_size_annot, style='italic')
            
            delta_strain = strain_end - strain_start
            delta_time = time_end - time_start
            if delta_time != 0:
                rate = delta_strain / delta_time
                line_x = np.array([time_start, time_end])
                line_y = strain_start + rate * (line_x - time_start)
                ax.plot(line_x, line_y, color=style.fit_color, linewidth=1.8, linestyle='--', label=f"Secondary Fit (Rate: {rate:.2e})")
    else:
        ax.text(np.max(X) / 2, y_annot, "Primary", ha='center', fontsize=style.font_size_annot, style='italic')

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Strain")
    if style.show_titles:
        ax.set_title(f"Creep Stages Boundary Identification - Test {test_id}")
        
    _apply_thesis_style(ax, style, X, y)
    
    ax.legend(loc="lower right", fontsize=style.font_size_legend, framealpha=0.9)
    
    out_path = output_dir / f"creep_stages_{test_id}.png"
    fig.tight_layout()
    fig.savefig(out_path, dpi=style.dpi, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_eps_tilde_0_vs_stress(df: pd.DataFrame, output_dir: Path, style: EDAStyleConfig) -> Path:
    """eps~0 (initial elastic strain) vs. applied stress."""
    output_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=style.figsize_full) 
    
    temp_col = "Mean_Temp_C" if "Mean_Temp_C" in df.columns else "Mean_Temp_C_Secondary_Creep"
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
        
    _apply_thesis_style(ax, style, df["Applied_Stress_MPa"], df["Eps_Tilde_0"])

    if scatter_plots:
        cbar = plt.colorbar(scatter_plots[0], ax=ax)
        cbar.set_label('Mean Test Temperature (°C)', fontsize=style.font_size_label, fontweight='bold')
        cbar.ax.tick_params(labelsize=style.font_size_tick)

    legend_elements = [
        Line2D([0], [0], marker=style.quality_markers[q], color='w', label=q,
               markerfacecolor='gray', markersize=7, markeredgecolor='black')
        for q in style.quality_order
    ]
    
    leg = ax.legend(handles=legend_elements, title="Print Quality", loc="upper left",
                    fontsize=style.font_size_legend, framealpha=0.9)
    plt.setp(leg.get_title(), fontsize=style.font_size_legend, fontweight='bold')

    out_path = output_dir / "eps_tilde_0_vs_stress.png"
    fig.tight_layout()
    fig.savefig(out_path, dpi=style.dpi, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_eps_dot_ss_vs_stress(df: pd.DataFrame, output_dir: Path, style: EDAStyleConfig) -> Path:
    """eps_dot_ss (steady-state strain rate) vs. applied stress (excludes tests without detected secondary creep)."""
    output_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=style.figsize_full)
    
    # Filter out specimens where secondary creep was not detected
    valid_df = df[df["Eps_Dot_Ss"].notna() & (df["Eps_Dot_Ss"] > 0)].copy()
    
    temp_col = "Mean_Temp_C" if "Mean_Temp_C" in valid_df.columns else "Mean_Temp_C_Secondary_Creep"
    vmin, vmax = valid_df[temp_col].min(), valid_df[temp_col].max() if not valid_df.empty else (20, 25)

    scatter_plots = []
    for quality in style.quality_order:
        sub = valid_df[valid_df["Print_Quality"] == quality]
        if sub.empty:
            continue
        sc = ax.scatter(sub["Applied_Stress_MPa"], sub["Eps_Dot_Ss"],
                        c=sub[temp_col], cmap=style.cmap, vmin=vmin, vmax=vmax,
                        marker=style.quality_markers[quality], label=quality,
                        s=style.base_marker_size, alpha=style.alpha, edgecolor="black", linewidth=0.7)
        scatter_plots.append(sc)
        
    ax.set_xlabel("Applied Stress (MPa)")
    ax.set_ylabel(r"$\hat{\dot\epsilon}_{ss}$ (Steady-State Strain Rate, s$^{-1}$)")
    if style.show_titles:
        ax.set_title("Steady-State Strain Rate vs. Applied Stress")
        
    _apply_thesis_style(ax, style, valid_df["Applied_Stress_MPa"], valid_df["Eps_Dot_Ss"])

    if scatter_plots:
        cbar = plt.colorbar(scatter_plots[0], ax=ax)
        cbar.set_label('Mean Secondary Creep Temp (°C)', fontsize=style.font_size_label, fontweight='bold')
        cbar.ax.tick_params(labelsize=style.font_size_tick)

    legend_elements = [
        Line2D([0], [0], marker=style.quality_markers[q], color='w', label=q,
               markerfacecolor='gray', markersize=7, markeredgecolor='black')
        for q in style.quality_order
    ]
    leg = ax.legend(handles=legend_elements, title="Print Quality", loc="upper left",
                    fontsize=style.font_size_legend, framealpha=0.9)
    plt.setp(leg.get_title(), fontsize=style.font_size_legend, fontweight='bold')

    out_path = output_dir / "eps_dot_ss_vs_stress.png"
    fig.tight_layout()
    fig.savefig(out_path, dpi=style.dpi, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_eps_tilde_0_vs_age(df: pd.DataFrame, output_dir: Path, style: EDAStyleConfig) -> Path:
    """eps~0 vs. specimen age, bubble-sized by applied stress."""
    output_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=style.figsize_full)
    for quality in style.quality_order:
        sub = df[df["Print_Quality"] == quality]
        sizes = (style.base_marker_size / 2.0) + (2.5 * sub["Applied_Stress_MPa"])
        ax.scatter(sub["Age_Days"], sub["Eps_Tilde_0"], s=sizes,
                   color=style.quality_colors[quality], label=quality,
                   alpha=style.alpha, edgecolor="white")
                   
    ax.set_xlabel("Specimen Age (days)")
    ax.set_ylabel(r"$\tilde{\epsilon}_0$ (Initial Elastic Strain)")
    if style.show_titles:
        ax.set_title("Initial Strain vs. Specimen Age")
        
    _apply_thesis_style(ax, style, df["Age_Days"], df["Eps_Tilde_0"])

    leg = ax.legend(title="Print Quality", fontsize=style.font_size_legend, framealpha=0.9)
    plt.setp(leg.get_title(), fontsize=style.font_size_legend, fontweight='bold')

    out_path = output_dir / "eps_tilde_0_vs_age.png"
    fig.tight_layout()
    fig.savefig(out_path, dpi=style.dpi, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_eps_dot_ss_vs_temperature(df: pd.DataFrame, output_dir: Path, style: EDAStyleConfig) -> Path:
    """eps_dot_ss vs. mean temperature (excludes tests without detected secondary creep)."""
    output_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=style.figsize_full)
    
    # Filter out specimens where secondary creep was not detected
    valid_df = df[df["Eps_Dot_Ss"].notna() & (df["Eps_Dot_Ss"] > 0)].copy()
    temp_col = "Mean_Temp_C" if "Mean_Temp_C" in valid_df.columns else "Mean_Temp_C_Secondary_Creep"
    
    for quality in style.quality_order:
        sub = valid_df[valid_df["Print_Quality"] == quality]
        if sub.empty:
            continue
        sizes = (style.base_marker_size / 2.0) + (2.5 * sub["Applied_Stress_MPa"])
        ax.scatter(sub[temp_col], sub["Eps_Dot_Ss"], s=sizes,
                   color=style.quality_colors[quality], label=quality,
                   alpha=style.alpha, edgecolor="white")
                   
    ax.set_xlabel("Mean Test Temperature (°C)")
    ax.set_ylabel(r"$\hat{\dot\epsilon}_{ss}$ (s$^{-1}$)")
    if style.show_titles:
        ax.set_title("Steady-State Strain Rate vs. Temperature")
        
    _apply_thesis_style(ax, style, valid_df[temp_col], valid_df["Eps_Dot_Ss"])

    leg = ax.legend(title="Print Quality", fontsize=style.font_size_legend, framealpha=0.9)
    plt.setp(leg.get_title(), fontsize=style.font_size_legend, fontweight='bold')

    out_path = output_dir / "eps_dot_ss_vs_temperature.png"
    fig.tight_layout()
    fig.savefig(out_path, dpi=style.dpi, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_distribution_by_quality(df: pd.DataFrame, output_dir: Path, style: EDAStyleConfig) -> Path:
    """Box+strip comparison of both EDA statistics, split by print quality."""
    output_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=style.figsize_full)

    for ax, col, ylabel in zip(
        axes,
        ["Eps_Tilde_0", "Eps_Dot_Ss"],
        [r"$\tilde{\epsilon}_0$", r"$\hat{\dot\epsilon}_{ss}$ (s$^{-1}$)"],
    ):
        plot_df = df[df[col].notna() & (df[col] > 0)] if col == "Eps_Dot_Ss" else df
        
        sns.boxplot(data=plot_df, x="Print_Quality", y=col, hue="Print_Quality", order=style.quality_order,
                    palette=style.quality_colors, ax=ax, showfliers=False, legend=False)
        
        sns.stripplot(data=plot_df, x="Print_Quality", y=col, order=style.quality_order,
                      color="black", size=4, alpha=style.alpha, ax=ax)
                      
        ax.set_xlabel("Print Quality")
        ax.set_ylabel(ylabel)
        
        _apply_thesis_style(ax, style, y_data=plot_df[col], is_categorical_x=True)

    if style.show_titles:
        fig.suptitle("Distribution of EDA Statistics by Print Quality", fontsize=style.font_size_title, fontweight='bold')
    
    fig.tight_layout()

    out_path = output_dir / "eda_distribution_by_quality.png"
    fig.savefig(out_path, dpi=style.dpi, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_mean_eps_dot_ss_bar(df: pd.DataFrame, output_dir: Path, style: EDAStyleConfig) -> Path:
    """Mean +/- std of steady-state strain rate (excludes tests without detected secondary creep)."""
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_df = df[df["Eps_Dot_Ss"].notna() & (df["Eps_Dot_Ss"] > 0)].copy()
    plot_df["Nominal_Stress"] = _nominal_stress(plot_df)

    summary = (
        plot_df.groupby(["Print_Quality", "Nominal_Stress"])["Eps_Dot_Ss"]
        .agg(["mean", "std", "count"])
        .reset_index()
    )

    fig, ax = plt.subplots(figsize=style.figsize_full)
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
        
    max_vals = summary["mean"] + summary["std"]
    _apply_thesis_style(ax, style, y_data=max_vals, is_categorical_x=True)

    leg = ax.legend(title="Print Quality", fontsize=style.font_size_legend, framealpha=0.9)
    plt.setp(leg.get_title(), fontsize=style.font_size_legend, fontweight='bold')

    out_path = output_dir / "eda_mean_eps_dot_ss_bar.png"
    fig.tight_layout()
    fig.savefig(out_path, dpi=style.dpi, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_pairwise_relationships(df: pd.DataFrame, output_dir: Path, style: EDAStyleConfig) -> Path:
    """Pairwise scatter/KDE grid (excludes non-detected secondary creep rates from eps_dot_ss panels)."""
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_df = df.copy()
    plot_df["Nominal_Stress"] = _nominal_stress(plot_df)

    # Set non-detected secondary creep rates (NaN or <= 0) to NaN so Seaborn excludes them from eps_dot_ss plots
    plot_df.loc[plot_df["Eps_Dot_Ss"].isna() | (plot_df["Eps_Dot_Ss"] <= 0), "Eps_Dot_Ss"] = np.nan

    # Prefer Mean_Temp_C, fallback to Mean_Temp_C_Secondary_Creep or Initial_Temp_C
    if "Mean_Temp_C" in plot_df.columns:
        temp_col = "Mean_Temp_C"
    elif "Mean_Temp_C_Secondary_Creep" in plot_df.columns:
        temp_col = "Mean_Temp_C_Secondary_Creep"
    else:
        temp_col = "Initial_Temp_C"

    if temp_col in plot_df.columns and "Initial_Temp_C" in plot_df.columns:
        plot_df[temp_col] = plot_df[temp_col].fillna(plot_df["Initial_Temp_C"])

    rename_map = {
        "Applied_Stress_MPa": "Stress\n(MPa)",
        "Age_Days": "Age\n(Days)",
        temp_col: "Temp\n(°C)",
        "Eps_Tilde_0": r"$\tilde{\epsilon}_0$",
        "Eps_Dot_Ss": r"$\hat{\dot\epsilon}_{ss}$"
    }
    plot_df = plot_df.rename(columns=rename_map)
    plot_vars = [v for v in rename_map.values() if v in plot_df.columns]

    facet_size = 2.0
    label_fs = style.font_size_label + 2
    tick_fs = style.font_size_tick + 1
    legend_fs = style.font_size_legend + 1.5
    
    g = sns.PairGrid(
        plot_df, vars=plot_vars, hue="Print_Quality", hue_order=style.quality_order,
        palette=style.quality_colors, height=facet_size, diag_sharey=False
    )
    
    g.map_offdiag(
        sns.scatterplot,
        style=plot_df["Nominal_Stress"],
        markers=style.stress_markers,
        alpha=style.alpha,
        s=style.base_marker_size
    )
    
    g.map_diag(sns.kdeplot, fill=True, common_norm=False, alpha=0.5, warn_singular=False)
    
    g.add_legend(title="Print Quality", fontsize=legend_fs, framealpha=0.9, bbox_to_anchor=(1.02, 0.5))
    if g._legend:
        plt.setp(g._legend.get_title(), fontsize=legend_fs, fontweight='bold')
    
    for i, y_col in enumerate(plot_vars):
        for j, x_col in enumerate(plot_vars):
            ax = g.axes[i, j]
            ax.tick_params(axis='both', labelsize=tick_fs)
            ax.grid(True, color=style.grid_color, linestyle='--', alpha=style.grid_alpha)
            
            y_label = ax.get_ylabel()
            if y_label:
                ax.set_ylabel(y_label, fontsize=label_fs, fontweight='bold')
            x_label = ax.get_xlabel()
            if x_label:
                ax.set_xlabel(x_label, fontsize=label_fs, fontweight='bold')

    if style.show_titles:
        g.fig.suptitle("Pairwise EDA Relationships", y=1.02, fontsize=label_fs + 2, fontweight='bold')

    out_path = output_dir / "eda_pairwise_relationships.png"
    g.fig.savefig(out_path, dpi=style.dpi, bbox_inches="tight")
    plt.close(g.fig)
    return out_path