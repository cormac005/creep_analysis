from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np

def main():
    # Load the processed HDF5 file containing both measured data and TLV predictions
    data_path = Path("data/processed/tlv_fit_results.h5") 
    
    # Group the tests by Print_Quality and rounded Applied_Stress_MPa
    grouped_tests = {}
    
    with h5py.File(data_path, "r") as f:
        # Iterate over all the tests saved in the HDF5 file
        for test_id, test_group in f["tests"].items():
            
            # Skip tests that don't have predictions/measurements (e.g., if solver failed)
            if "strain_predicted" not in test_group or "strain_measured" not in test_group:
                continue
                
            attrs = test_group.attrs
            
            # Decode print_quality from bytes if necessary (h5py string handling)
            print_quality = attrs.get("print_quality")
            if isinstance(print_quality, bytes):
                print_quality = print_quality.decode("utf-8")
                
            stress = attrs.get("applied_stress_MPa")
            if print_quality is None or stress is None:
                continue
                
            # Round stress to nearest 10 MPa for grouping
            key = (print_quality, round(stress, -1))
            
            if key not in grouped_tests:
                grouped_tests[key] = []
                
            # Extract data into memory
            grouped_tests[key].append({
                "test_id": test_id,
                "time_s": test_group["time_s"][:],
                "strain_measured": test_group["strain_measured"][:],
                "strain_predicted": test_group["strain_predicted"][:],
                "mean_temp": attrs.get("mean_temp_c_secondary_creep", None)
            })

    # Initialize a 2x3 grid layout
    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    
    # Define strict row and column order maps matching your layout specification
    row_order = ["Standard", "High"]
    col_order = [10.0, 20.0, 30.0]

    # Map each subplot via nested matrix positions
    for row_idx, print_quality in enumerate(row_order):
        for col_idx, stress in enumerate(col_order):
            ax = axes[row_idx, col_idx]
            key = (print_quality, stress)
            
            # Pull tests if the group exists, else skip safely
            if key not in grouped_tests:
                ax.text(0.5, 0.5, 'No Data Available', ha='center', va='center')
                ax.set_title(f'{print_quality} | {stress} MPa', fontsize=12, fontweight='bold')
                continue
                
            tests = grouped_tests[key]
            
            # Get default color cycle so the scatter and line plots match per test
            prop_cycle = plt.rcParams['axes.prop_cycle']
            colors = prop_cycle.by_key()['color']
            
            for i, test in enumerate(tests):
                color = colors[i % len(colors)]
                
                # Format label text
                if test["mean_temp"] is not None:
                    label_text = f'ID: {test["test_id"]} Mean Temp: {test["mean_temp"]:.1f} °C'
                else:
                    label_text = f'ID: {test["test_id"]} (Temp N/A)'
                    
                # 1. Plot the measured data (Ground truth) as scatter points
                ax.scatter(test["time_s"], test["strain_measured"], 
                           label=label_text, alpha=0.6, s=15, color=color)
                
                # 2. Plot the TLV model predictions as a solid line
                # We don't add a label here so the legend doesn't duplicate entries
                ax.plot(test["time_s"], test["strain_predicted"], 
                        linewidth=2.5, linestyle='-', color=color, alpha=0.8)
            
            # Formatting configurations
            ax.set_title(f'{print_quality} | {stress} MPa', fontsize=12, fontweight='bold')
            ax.set_xlabel('Time (s)', fontsize=10)
            ax.set_ylabel('Strain', fontsize=10)
            
            # Updated legend title to clarify the line vs dots
            ax.legend(title='Test Runs (Dots = Measured, Line = TLV)', 
                      fontsize=8, title_fontsize=9, loc='upper left')
            ax.grid(True, linestyle='--', alpha=0.5)

    # Tight layout combined with explicit padding adjustments to stop vertical overlaps
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.35, wspace=0.25)
    plt.show() 

if __name__ == "__main__":
    main()