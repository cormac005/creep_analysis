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
    # Because you are running the terminal from C:\Y4-Summer\FYP\Code\creep_model>
    # We can safely use direct relative paths!
    data_path = Path("data/processed/tlv_fit_results.h5")
    raw_path = Path("data/raw/CreepData.xlsx")
    output_dir = Path("plots/temp_v_time")
    
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
    
    print(f"Reading processed predictions from {data_path.absolute()}...")
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
    # Generate and save Individual Plots
    # ==========================================
    print("Generating individual test plots with temperature overlay...")
    
    for key, tests in grouped_tests.items():
        print_quality, stress = key
        
        for test in tests:
            fig, ax1 = plt.subplots(figsize=(10, 6))
            test_id = test["test_id"]
            
            # If temperature data exists, color strain and plot temp on secondary axis
            if test["temp_array"] is not None:
                vmin = np.min(test["temp_array"])
                vmax = np.max(test["temp_array"])
                
                # 1. Plot Measured Data (Strain)
                scatter = ax1.scatter(test["time_s"], test["strain_measured"], 
                                      c=test["temp_array"], cmap='coolwarm', 
                                      vmin=vmin, vmax=vmax,
                                      label='Measured Strain', alpha=0.8, s=20, zorder=2)
                
                # Add a single shared colorbar for the strain points
                cbar = plt.colorbar(scatter, ax=ax1, pad=0.1)
                cbar.set_label('Point Temperature (°C)', fontsize=10)
                
                # 2. Plot TLV Prediction as a LineCollection to allow color mapping
                points = np.array([test["time_s"], test["strain_predicted"]]).T.reshape(-1, 1, 2)
                segments = np.concatenate([points[:-1], points[1:]], axis=1)
                
                lc = LineCollection(segments, cmap='coolwarm', norm=plt.Normalize(vmin, vmax), zorder=1)
                lc.set_array(test["temp_array"][:-1])
                lc.set_linewidth(2.5)
                ax1.add_collection(lc)
                
                # Create a custom proxy artist for the LineCollection legend
                handles1, labels1 = ax1.get_legend_handles_labels()
                proxy_line = Line2D([0], [0], color='grey', linewidth=2.5, label='TLV Prediction')
                handles1.append(proxy_line)
                labels1.append('TLV Prediction')
                
                # 3. Plot Actual Temperature Profile on a Secondary Y-Axis
                ax2 = ax1.twinx()
                temp_line, = ax2.plot(test["time_s"], test["temp_array"], 
                                      color='red', linestyle='--', alpha=0.6, linewidth=2, 
                                      label='Interpolated Temp. Profile')
                
                # Configure Secondary Y-Axis
                ax2.set_ylabel('Temperature (°C)', color='red', fontsize=12, fontweight='bold')
                ax2.tick_params(axis='y', labelcolor='red')
                
                # Combine legends from both axes
                handles2, labels2 = ax2.get_legend_handles_labels()
                ax1.legend(handles1 + handles2, labels1 + labels2, loc='upper left')

            else:
                # Fallback if no temperature data is available
                ax1.scatter(test["time_s"], test["strain_measured"], 
                            label='Measured Strain (Temp N/A)', alpha=0.6, s=20, color='blue')
                ax1.plot(test["time_s"], test["strain_predicted"], 
                         label='TLV Prediction', linewidth=2.5, linestyle='-', color='black', alpha=0.8)
                ax1.legend(loc='upper left')
            
            # Formatting for Primary Axis
            ax1.set_title(f'Test: {test_id} ({print_quality} Quality, {stress} MPa)', fontsize=14, fontweight='bold')
            ax1.set_xlabel('Time (s)', fontsize=12)
            ax1.set_ylabel('Strain', fontsize=12)
            ax1.grid(True, linestyle='--', alpha=0.5)
            
            plt.tight_layout()
            
            # Save Individual Plot
            indiv_save_path = output_dir / f"{test_id}_Overlay_Plot.png"
            plt.savefig(indiv_save_path, dpi=300, bbox_inches='tight')
            plt.close(fig) # Prevent plots from accumulating in memory
            
            print(f"  -> Saved: {indiv_save_path.name}")

    print(f"\nAll plots have been successfully saved to: {output_dir.absolute()}")

if __name__ == "__main__":
    main()