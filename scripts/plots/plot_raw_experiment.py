"""
Generates a diagnostic plot for a specific test showing:
1. Measured strain (scatter)
2. Raw temperature readings (scatter)
3. Continuous temperature fit used by the TLV solver (line)
"""
import os
from pathlib import Path
import h5py
import matplotlib
matplotlib.use('Agg')  # Use Agg backend to save files without popping up windows
import matplotlib.pyplot as plt
import numpy as np
from creep_model.config import config

# --- CONFIGURATION ---
TEST_ID = "S.30.1"  
SHOW_TITLE = False  # Toggle to True to display the title on the plot

# Typography & Figure Sizing for LaTeX Thesis Insertion
# Target document main text size: ~10-11 pt
FIG_SIZE = (4.2, 3.0)       # Dimensioned for a 0.48\textwidth subfigure
FONT_SIZE_LABEL = 11        # Axis labels
FONT_SIZE_TICK = 10         # Axis tick values
FONT_SIZE_LEGEND = 9.5      # Legend entries
FONT_SIZE_TITLE = 11        # Title (if SHOW_TITLE = True)
# ---------------------

def main():
    # Pull Paths from Config and explicitly convert them to Path objects
    data_path = Path(config.data_output_directory) / "tlv_fit_results.h5"
    output_dir = Path(config.general_output_directory) / "raw_data_plots"
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if not data_path.exists():
        print(f"Error: Could not find {data_path}. Run process_data_and_fit.py first.")
        return
        
    print(f"Reading data for {TEST_ID} from {data_path.name}...")
    
    with h5py.File(data_path, "r") as f:
        tests_group = f.get("tests")
        
        if tests_group is None or TEST_ID not in tests_group:
            print(f"Error: Test '{TEST_ID}' not found in the HDF5 file.")
            print(f"Available tests: {list(tests_group.keys()) if tests_group else 'None'}")
            return
            
        group = tests_group[TEST_ID]
        
        # Ensure temperature data actually exists in this group
        if "temperature_raw" not in group:
            print(f"Error: Temperature datasets missing for {TEST_ID}. "
                  f"Please run append_processed_data_with_eda.py to append them.")
            return
            
        # Load arrays into memory
        time_s = group["time_s"][:]
        strain_measured = group["strain_measured"][:]
        temp_time_s = group["temp_time_s"][:]
        temp_raw = group["temperature_raw"][:]
        temp_interp = group["temperature_interpolated"][:]
        
        # Load metadata for title
        quality = group.attrs.get("print_quality", b"").decode("utf-8") if isinstance(group.attrs.get("print_quality"), bytes) else group.attrs.get("print_quality", "Unknown")
        stress = group.attrs.get("applied_stress_MPa", "Unknown")

    # --- PLOTTING ---
    print(f"Generating plot for {TEST_ID}...")
    
    fig, ax1 = plt.subplots(figsize=FIG_SIZE)
    
    # 1. Primary Y-Axis: Strain
    color_strain = '#1f77b4' # Blue
    ax1.set_xlabel("Time (s)", fontsize=FONT_SIZE_LABEL)
    ax1.set_ylabel("Strain", color=color_strain, fontsize=FONT_SIZE_LABEL, fontweight='bold')
    
    scatter_strain = ax1.scatter(time_s, strain_measured, color=color_strain, 
                                 alpha=0.6, s=10, label="Measured Strain")
    
    # Set tick font sizes for X and Primary Y axes
    ax1.tick_params(axis='x', labelsize=FONT_SIZE_TICK)
    ax1.tick_params(axis='y', labelcolor=color_strain, labelsize=FONT_SIZE_TICK)
    ax1.grid(True, linestyle='--', alpha=0.4)
    
    # Include 0 on X and primary Y axes, with top/right headroom
    time_min = min(0, np.nanmin(time_s))
    time_max = np.nanmax(time_s)
    ax1.set_xlim(left=time_min, right=time_max + 0.03 * (time_max - time_min))
    
    strain_min = min(0, np.nanmin(strain_measured))
    strain_max = np.nanmax(strain_measured)
    ax1.set_ylim(bottom=strain_min, top=strain_max + 0.15 * (strain_max - strain_min))
    
    # 2. Secondary Y-Axis: Temperature
    ax2 = ax1.twinx()  
    color_temp_raw = '#d62728'  # Red
    color_temp_fit = 'black'
    
    ax2.set_ylabel("Temperature (°C)", color=color_temp_raw, fontsize=FONT_SIZE_LABEL, fontweight='bold')
    
    # Plot raw temperature points
    valid_mask = ~np.isnan(temp_raw)
    scatter_temp = ax2.scatter(temp_time_s[valid_mask], temp_raw[valid_mask], 
                               color=color_temp_raw, marker='x', s=35, linewidths=1.2,
                               label="Raw Temp. Readings")
    
    # Plot continuous temperature fit
    line_temp = ax2.plot(time_s, temp_interp, color=color_temp_fit, 
                         linestyle='--', linewidth=1.8, alpha=0.8,
                         label="Continuous Temp. Fit")
                         
    # Set tick font size for Secondary Y axis
    ax2.tick_params(axis='y', labelcolor=color_temp_raw, labelsize=FONT_SIZE_TICK)
    
    # Include 0 on secondary Y axis with 15% top headroom
    temp_min = min(0, np.nanmin(temp_raw[valid_mask]))
    temp_max = max(np.nanmax(temp_raw[valid_mask]), np.nanmax(temp_interp))
    ax2.set_ylim(bottom=temp_min, top=temp_max + 0.15 * (temp_max - temp_min))
    
    # Add title conditionally based on SHOW_TITLE
    if SHOW_TITLE:
        plt.title(f"Test {TEST_ID}: Strain and Temperature Profile ({quality} Quality, {stress} MPa)", 
                  fontsize=FONT_SIZE_TITLE, fontweight='bold')
              
    # Combine legends with specified font size
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc="center right", 
               fontsize=FONT_SIZE_LEGEND, framealpha=0.9)

    plt.tight_layout()
    
    # Save Plot
    out_path = output_dir / f"{TEST_ID}_raw_data_and_temp.png"
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    print(f"Saved: {out_path.absolute()}")

if __name__ == "__main__":
    main()