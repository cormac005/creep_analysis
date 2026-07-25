from pathlib import Path

import matplotlib
from creep_model.io.parser import ExcelCreepParser
from creep_model.modelling.assembler import DataAssembler
import matplotlib.pyplot as plt
import numpy as np

def main():
    # Load the Excel file and parse the experiment
    data_path = Path("data/raw/CreepData.xlsx") 
    parser = ExcelCreepParser(data_path)
    experiment = parser.load_experiment()
    
    # Group the tests by Print_Quality and rounded Applied_Stress_MPa
    grouped_tests = {}
    for test_id, test in experiment.tests.items():
        key = (test.print_quality, round(test.applied_stress_MPa, -1))  # Round to nearest 10 MPa
        if key not in grouped_tests:
            grouped_tests[key] = []
        grouped_tests[key].append(test)

    # Initialize a 2x3 grid layout
    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    
    # NEW: Define strict row and column order maps matching your layout specification
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
            
            for test in tests:
                X, y = DataAssembler.get_local_data(test)
                
                # Calculate the mean from the 'temperature_readings' array
                if test.temperature_readings is not None and len(test.temperature_readings) > 0:
                    test_mean_temp = test.temperature_readings.mean()
                    label_text = f'ID: {test.test_id} Mean Temp: {test_mean_temp:.1f} °C'
                else:
                    label_text = f'ID: {test.test_id} (Temp N/A)'
                    
                ax.scatter(X.flatten(), y, label=label_text, alpha=0.6, s=15)
            
            # Formatting configurations
            ax.set_title(f'{print_quality} | {stress} MPa', fontsize=12, fontweight='bold')
            ax.set_xlabel('Time (s)', fontsize=10)
            ax.set_ylabel('Strain', fontsize=10)
            ax.legend(title='Test Runs', fontsize=8, title_fontsize=9, loc='upper left')
            ax.grid(True, linestyle='--', alpha=0.5)

    # NEW: Tight layout combined with explicit padding adjustments to stop vertical overlaps
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.35, wspace=0.25) # Adds breathing room between grid units
    plt.show() 

if __name__ == "__main__":
    main()
