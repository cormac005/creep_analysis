from pathlib import Path

import matplotlib
# Use the 'Agg' backend to avoid Tkinter GUI errors
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import numpy as np

from creep_model.io.parser import ExcelCreepParser
from creep_model.modelling.assembler import DataAssembler
from creep_model.eda.stage_classification import classify_stages

# --- CONFIGURATION ---
MANUAL_K1 = 3
MANUAL_K2 = 3

def main():
    # Load the Excel file and parse the experiment
    data_path = Path("data/raw/CreepData.xlsx") 
    parser = ExcelCreepParser(data_path)
    experiment = parser.load_experiment()
    
    # Group the tests by Print_Quality and rounded Applied_Stress_MPa
    grouped_tests = {}
    for test_id, test in experiment.tests.items():
        if test.is_empty:
            continue
        key = (test.print_quality, round(test.applied_stress_MPa, -1))  # Round to nearest 10 MPa
        if key not in grouped_tests:
            grouped_tests[key] = []
        grouped_tests[key].append(test)

    # Define strict row and column order maps matching your layout specification
    row_order = ["Standard", "High"]
    col_order = [10.0, 20.0, 30.0]

    # Initialize two distinct grid layouts matching your specifications
    fig1, axes1 = plt.subplots(2, 3, figsize=(18, 11))
    fig2, axes2 = plt.subplots(2, 3, figsize=(18, 11))

    # Map each subplot via nested matrix positions
    for row_idx, print_quality in enumerate(row_order):
        for col_idx, stress in enumerate(col_order):
            ax1 = axes1[row_idx, col_idx]
            ax2 = axes2[row_idx, col_idx]
            key = (print_quality, stress)
            
            # Pull tests if the group exists, else skip safely
            if key not in grouped_tests:
                for ax in [ax1, ax2]:
                    ax.text(0.5, 0.5, 'No Data Available', ha='center', va='center')
                    ax.set_title(f'{print_quality} | {stress} MPa', fontsize=12, fontweight='bold')
                continue
                
            tests = grouped_tests[key]
            
            for test_idx, test in enumerate(tests):
                X, y = DataAssembler.get_local_data(test)
                time_flat = X.flatten()
                
                # Calculate the mean from the 'temperature_readings' array
                if test.temperature_readings is not None and len(test.temperature_readings) > 0:
                    test_mean_temp = test.temperature_readings.mean()
                    label_text = f'ID: {test.test_id} ({test_mean_temp:.1f} °C)'
                else:
                    label_text = f'ID: {test.test_id} (Temp N/A)'
                    
                # Run the stage classification up front
                result = classify_stages(test, k1=MANUAL_K1, k2=MANUAL_K2)
                
                # --- FIGURE 1: Physical Strain vs. Time Plot with Markers ---
                ax1.scatter(time_flat, y, label=label_text, alpha=0.4, s=15)
                
                if result.primary_end_idx is not None:
                    p_idx = result.primary_end_idx
                    ax1.scatter(
                        time_flat[p_idx], y[p_idx], 
                        color='red', marker='X', s=120, zorder=5, edgecolor='black',
                        label='Primary End' if test_idx == 0 else ""
                    )
                    
                if result.secondary_end_idx is not None:
                    s_idx = result.secondary_end_idx
                    ax1.scatter(
                        time_flat[s_idx], y[s_idx], 
                        color='orange', marker='s', s=100, zorder=5, edgecolor='black',
                        label='Secondary End' if test_idx == 0 else ""
                    )

                # --- FIGURE 2: Distribution Scatter Plot with Transition Markers ---
                unique_strains, counts = np.unique(y, return_counts=True)
                ax2.scatter(unique_strains[:-1], counts[:-1], label=label_text, alpha=0.4, s=20)
                
                # Map physical indices down into the unique strain frequency bins
                if result.primary_end_idx is not None:
                    transition_strain_1 = y[result.primary_end_idx]
                    match_mask = (unique_strains == transition_strain_1)
                    if np.any(match_mask):
                        ax2.scatter(
                            transition_strain_1, counts[match_mask], 
                            color='red', marker='X', s=120, zorder=5, edgecolor='black',
                            label='Primary End' if test_idx == 0 else ""
                        )
                        
                if result.secondary_end_idx is not None:
                    transition_strain_2 = y[result.secondary_end_idx]
                    match_mask = (unique_strains == transition_strain_2)
                    if np.any(match_mask):
                        ax2.scatter(
                            transition_strain_2, counts[match_mask], 
                            color='orange', marker='s', s=100, zorder=5, edgecolor='black',
                            label='Secondary End' if test_idx == 0 else ""
                        )
            
            # --- Format Figure 1 Axes ---
            ax1.set_title(f'{print_quality} | {stress} MPa', fontsize=12, fontweight='bold')
            ax1.set_xlabel('Time (s)', fontsize=10)
            ax1.set_ylabel('Strain', fontsize=10)
            h1, l1 = ax1.get_legend_handles_labels()
            ax1.legend(dict(zip(l1, h1)).values(), dict(zip(l1, h1)).keys(), title='Test Runs', fontsize=8, loc='upper left')
            ax1.grid(True, linestyle='--', alpha=0.5)

            # --- Format Figure 2 Axes ---
            ax2.set_title(f'{print_quality} | {stress} MPa', fontsize=12, fontweight='bold')
            ax2.set_xlabel('Strain', fontsize=10)
            ax2.set_ylabel('Number of Data Points', fontsize=10)
            h2, l2 = ax2.get_legend_handles_labels()
            ax2.legend(dict(zip(l2, h2)).values(), dict(zip(l2, h2)).keys(), title='Distribution Profiles', fontsize=8, loc='upper left')
            ax2.grid(True, linestyle='--', alpha=0.5)

    # Clean and save Figure 1
    fig1.tight_layout()
    fig1.subplots_adjust(hspace=0.35, wspace=0.25)
    output_f1 = "strain_vs_time_transitions.png"
    fig1.savefig(output_f1, dpi=300, bbox_inches='tight')
    print(f"Success! Physical time timeline plot saved to: {Path(output_f1).absolute()}")

    # Clean and save Figure 2
    fig2.tight_layout()
    fig2.subplots_adjust(hspace=0.35, wspace=0.25)
    output_f2 = "strain_distribution_transitions.png"
    fig2.savefig(output_f2, dpi=300, bbox_inches='tight')
    print(f"Success! Value density layout saved to: {Path(output_f2).absolute()}")

if __name__ == "__main__":
    main()
