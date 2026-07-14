from pathlib import Path
import matplotlib
# Use the 'Agg' backend to avoid Tkinter GUI errors
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import numpy as np

from creep_model.io.parser import ExcelCreepParser
from creep_model.modeling.assembler import DataAssembler

def main():
    # Load the Excel file and parse the experiment
    data_path = Path("data/raw/CreepData.xlsx") 
    parser = ExcelCreepParser(data_path)
    experiment = parser.load_experiment()
    
    # Loop through all tests
    for test_id, test in experiment.tests.items():
        X_raw, y = DataAssembler.get_local_data(test)
        X = X_raw.flatten()
        
        # Extract unique quantized strain values and counts
        unique_strains, counts = np.unique(y, return_counts=True)
        unique_strains = unique_strains[:-1]
        counts = counts[:-1]

        # Calculate directional differences between consecutive counts
        diffs = np.diff(counts)
        
        # --- 1. IDENTIFY SECONDARY CREEP (3 consecutive non-increasing counts in a row) ---
        is_decreasing = (diffs <= 0)
        
        # Look for 3 consecutive Trues: index i, i+1, i+2
        secondary_trigger = np.where(
            is_decreasing[:-2] & is_decreasing[1:-1] & is_decreasing[2:]
        )[0]

        if len(secondary_trigger) > 0:
            # The transition stabilizes at the start of the 3-step sequence
            first_dec_idx = secondary_trigger[0]
            
            strain_start = float(unique_strains[first_dec_idx])
            matching_start_indices = np.where(y == strain_start)[0]
            time_start = float(X[matching_start_indices[0]])
            
            print(f"Test {test_id} begins secondary creep at time {time_start:.4f}s with strain {strain_start:.4e}.")
            
            # --- 2. IDENTIFY TERTIARY CREEP (3 consecutive counts smaller than the plateau level) ---
            # Look at all counts following the established secondary creep start
            plateau_value = counts[first_dec_idx]
            is_tertiary_drop = (counts[first_dec_idx:] < plateau_value)
            
            # Look for 3 consecutive drops below the plateau
            tertiary_trigger = np.where(
                is_tertiary_drop[:-2] & is_tertiary_drop[1:-1] & is_tertiary_drop[2:]
            )[0]
            
            if len(tertiary_trigger) > 0:
                # Align the index back to the global unique_strains array layout
                end_idx = first_dec_idx + tertiary_trigger[0]
                strain_end = float(unique_strains[end_idx])
                
                matching_end_indices = np.where(y == strain_end)[0]
                time_end = float(X[matching_end_indices[0]])
                
                print(f"Test {test_id} exhibits tertiary creep at time {time_end:.4f}s with strain {strain_end:.4e}.")
                
                # --- 3. CALCULATE STRAIN RATE ---
                delta_strain = strain_end - strain_start
                delta_time = time_end - time_start
                
                if delta_time != 0:
                    secondary_rate = delta_strain / delta_time
                    print(f"Test {test_id} Secondary Creep Strain Rate: {secondary_rate:.4e} s^-1\n")
                else:
                    print(f"Test {test_id} Error: Division by zero time interval.\n")
            else:
                # Default to final recorded elements if tertiary acceleration is never sustained
                strain_end = float(unique_strains[-1])
                matching_end_indices = np.where(y == strain_end)[0]
                time_end = float(X[matching_end_indices[0]])
                
                delta_strain = strain_end - strain_start
                delta_time = time_end - time_start
                
                if delta_time != 0:
                    secondary_rate = delta_strain / delta_time
                    print(f"Test {test_id} only exhibits primary and secondary creep.")
                    print(f"Test {test_id} Secondary Creep Strain Rate: {secondary_rate:.4e} s^-1\n")
                else:
                    print(f"Test {test_id} only exhibits primary and secondary creep (insufficient data).\n")
        else:
            print(f"Test {test_id} only exhibits primary creep.\n")

if __name__ == "__main__":
    main()
