import os
from pathlib import Path

import h5py
import matplotlib
# Use the 'Agg' backend to completely disable interactive plot windows
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
import numpy as np

# We need the parser to fetch the test objects for temperature interpolation
from creep_model.io.parser import ExcelCreepParser
from creep_model.modelling.tlv.trimming import trim_and_partition

def main():
    # 1. Determine paths dynamically relative to this script's location
    project_root = Path(__file__).resolve().parent.parent
    
    data_path = project_root / "data/processed/tlv_fit_results.h5"
    raw_path = project_root / "data/raw/CreepData.xlsx"
    output_dir = project_root / "plots/eps_v_time"
    
    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("Loading raw data to retrieve test objects...")
    parser = ExcelCreepParser(raw_path)
    experiment = parser.load_experiment()
    # Trim with standard k1/k2 to ensure structural alignment
    trimmed_groups = trim_and_partition(experiment, k1=3, k2=3)
    
    # Flatten trimmed tests into a dictionary for easy lookup by ID
    trimmed_tests_dict = {}
    for quality, tests in trimmed_groups.items():
        for t in tests:
            trimmed_tests_dict[t.test_id] = t
    
    # Group the tests by Print_Quality and rounded Applied_Stress_MPa
    grouped_tests = {}
    
    print(f"Reading processed predictions from {data_path}...")
    with h5py.File(data_path, "r") as f:
        for test_id, test_group in f["tests"].items():
            
            if "strain_predicted" not in test_group or "strain_measured" not in test_group:
                continue
                
            attrs = test_group.attrs
            
            print_quality = attrs.get("print_quality")
            if isinstance(print_quality, bytes):
                print_quality = print_quality.decode("utf-8")
                
            stress = attrs.get("applied_stress_MPa")
            if print_quality is None or stress is None:
                continue
                
            key = (print_quality, round(stress, -1))
            if key not in grouped_tests:
                grouped_tests[key] = []
                
            time_s = test_group["time_s"][:]
            
            # Fetch the interpolated temperature array directly
            temp_array = None
            if test_id in trimmed_tests_dict:
                t_obj = trimmed_tests_dict[test_id]
                try:
                    # interpolate_temperature() returns the array directly
                    temp_array = t_obj.interpolate_temperature()
                    
                    # Failsafe: if lengths don't match, fall back to None
                    if len(temp_array) != len(time_s):
                        print(f"  -> Warning: Temp array length mismatch for {test_id}. Skipping color coding.")
                        temp_array = None
                except Exception as e:
                    print(f"  -> Warning: Could not fetch temperature for {test_id}: {e}")
                    temp_array = None
                
            grouped_tests[key].append({
                "test_id": test_id,
                "time_s": time_s,
                "strain_measured": test_group["strain_measured"][:],
                "strain_predicted": test_group["strain_predicted"][:],
                "mean_temp": attrs.get("mean_temp_c_secondary_creep", None),
                "temp_array": temp_array
            })

    # ==========================================
    # 1. Generate and save the 2x3 Grid Plot
    # ==========================================
    print("Generating 2x3 grid summary plot...")
    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    
    row_order = ["Standard", "High"]
    col_order = [10.0, 20.0, 30.0]

    for row_idx, print_quality in enumerate(row_order):
        for col_idx, stress in enumerate(col_order):
            ax = axes[row_idx, col_idx]
            key = (print_quality, stress)
            
            if key not in grouped_tests:
                ax.text(0.5, 0.5, 'No Data Available', ha='center', va='center')
                ax.set_title(f'{print_quality} | {stress} MPa', fontsize=12, fontweight='bold')
                continue
                
            tests = grouped_tests[key]
            
            prop_cycle = plt.rcParams['axes.prop_cycle']
            colors = prop_cycle.by_key()['color']
            
            for i, test in enumerate(tests):
                color = colors[i % len(colors)]
                
                if test["mean_temp"] is not None:
                    label_text = f'ID: {test["test_id"]} Mean Temp: {test["mean_temp"]:.1f} °C'
                else:
                    label_text = f'ID: {test["test_id"]} (Temp N/A)'
                    
                ax.scatter(test["time_s"], test["strain_measured"], 
                           label=label_text, alpha=0.6, s=15, color=color)
                
                ax.plot(test["time_s"], test["strain_predicted"], 
                        linewidth=2.5, linestyle='-', color=color, alpha=0.8)
            
            ax.set_title(f'{print_quality} | {stress} MPa', fontsize=12, fontweight='bold')
            ax.set_xlabel('Time (s)', fontsize=10)
            ax.set_ylabel('Strain', fontsize=10)
            ax.legend(title='Test Runs (Dots = Measured, Line = TLV)', 
                      fontsize=8, title_fontsize=9, loc='upper left')
            ax.grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.subplots_adjust(hspace=0.35, wspace=0.25)
    
    # Save Grid Plot
    grid_save_path = output_dir / "Grid_Summary_Plot.png"
    plt.savefig(grid_save_path, dpi=300, bbox_inches='tight')
    plt.close(fig) # Closes the figure to free memory
    
    # ==========================================
    # 2. Generate and save Individual Plots
    # ==========================================
    print("Generating individual test plots colored by temperature...")
    
    for key, tests in grouped_tests.items():
        print_quality, stress = key
        
        for test in tests:
            fig, ax = plt.subplots(figsize=(9, 6))
            test_id = test["test_id"]
            
            # If temperature data exists, color both scatter and line
            if test["temp_array"] is not None:
                # Get min/max temperature to lock the color scale across both elements
                vmin = np.min(test["temp_array"])
                vmax = np.max(test["temp_array"])
                
                # 1. Plot Measured Data
                scatter = ax.scatter(test["time_s"], test["strain_measured"], 
                                     c=test["temp_array"], cmap='coolwarm', 
                                     vmin=vmin, vmax=vmax,
                                     label='Measured Data', alpha=0.8, s=20, zorder=2)
                
                # Add a single shared colorbar
                cbar = plt.colorbar(scatter, ax=ax)
                cbar.set_label('Point Temperature (°C)', fontsize=10)
                
                # 2. Plot TLV Prediction as a LineCollection to allow color mapping
                # Restructure data into segments connecting point i to point i+1
                points = np.array([test["time_s"], test["strain_predicted"]]).T.reshape(-1, 1, 2)
                segments = np.concatenate([points[:-1], points[1:]], axis=1)
                
                lc = LineCollection(segments, cmap='coolwarm', norm=plt.Normalize(vmin, vmax), zorder=1)
                # Map temperature to the segments (use length n-1 to match segments)
                lc.set_array(test["temp_array"][:-1])
                lc.set_linewidth(2.5)
                ax.add_collection(lc)
                
                # 3. Create a custom proxy artist for the legend (LineCollections don't auto-populate legends cleanly)
                handles, labels = ax.get_legend_handles_labels()
                proxy_line = Line2D([0], [0], color='grey', linewidth=2.5, label='TLV Prediction (Temp Mapped)')
                handles.append(proxy_line)
                ax.legend(handles=handles, loc='upper left')

            else:
                ax.scatter(test["time_s"], test["strain_measured"], 
                           label='Measured Data (Temp N/A)', alpha=0.6, s=20, color='blue')
                ax.plot(test["time_s"], test["strain_predicted"], 
                        label='TLV Prediction', linewidth=2.5, linestyle='-', color='black', alpha=0.8)
                ax.legend(loc='upper left')
            
            # Formatting
            ax.set_title(f'Test: {test_id} ({print_quality} Quality, {stress} MPa)', fontsize=14, fontweight='bold')
            ax.set_xlabel('Time (s)', fontsize=12)
            ax.set_ylabel('Strain', fontsize=12)
            ax.grid(True, linestyle='--', alpha=0.5)
            
            plt.tight_layout()
            
            # Save Individual Plot
            indiv_save_path = output_dir / f"{test_id}_Plot.png"
            plt.savefig(indiv_save_path, dpi=300, bbox_inches='tight')
            plt.close(fig) # Prevent plots from accumulating in memory

    print(f"All plots have been successfully saved to: {output_dir.absolute()}")

if __name__ == "__main__":
    main()