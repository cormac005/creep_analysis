import dataclasses
from pathlib import Path
import h5py

from creep_model.io.parser import ExcelCreepParser
from creep_model.modelling.tlv.trimming import trim_and_partition
from creep_model.modelling.tlv.solver import solve_tlv, SolverConvergenceError

# Attempt to import TLVParameters to preserve strict type checking
try:
    from creep_model.modelling.tlv.parameters import TLVParameters
except ImportError:
    TLVParameters = None
    from types import SimpleNamespace

# --- Configuration ---
DATA_PATH = Path("data/raw/CreepData.xlsx")
OUTPUT_PATH = Path("data/processed/tlv_fit_results.h5")

# Locked hyperparameters from EDA tuning to ensure trims match exactly
K1 = 3
K2 = 3

def main() -> None:
    if not OUTPUT_PATH.exists():
        print(f"Error: Could not find {OUTPUT_PATH}.")
        return

    # 1. Load and parse the original raw data to get the fully functional test objects
    print(f"Loading raw data from {DATA_PATH}...")
    parser = ExcelCreepParser(DATA_PATH)
    experiment = parser.load_experiment()

    print(f"Trimming and partitioning tests (K1={K1}, K2={K2})...")
    groups = trim_and_partition(experiment, k1=K1, k2=K2)

    # Flatten the groups into a dictionary of test_id -> (quality, test_obj)
    test_objects = {}
    for quality, tests in groups.items():
        for test in tests:
            test_objects[test.test_id] = (quality, test)

    print(f"Opening {OUTPUT_PATH} to re-calculate and update TLV predictions...")
    
    # 2. Open the HDF5 file in read/write mode ("r+") to modify it in place
    with h5py.File(OUTPUT_PATH, "r+") as f:
        
        # Pull the fitted parameters stored in the file
        params_dict = {}
        for quality in ["High", "Standard"]:
            group_path = f"fitted_parameters/{quality}"
            if group_path in f:
                attrs = dict(f[group_path].attrs)
                
                # Use standard TLVParameters if imported, otherwise fall back to SimpleNamespace
                if TLVParameters is not None:
                    params_dict[quality] = TLVParameters(**attrs)
                else:
                    params_dict[quality] = SimpleNamespace(**attrs)
                    
        if not params_dict:
            print("Error: No fitted parameters found in the HDF5 file.")
            return

        # 3. Iterate through all valid tests in the HDF5 file
        tests_group = f["tests"]
        success_count = 0
        
        for test_id, group in tests_group.items():
            print(f"Processing test {test_id}...")
            
            if test_id not in test_objects:
                print(f"  -> Skipping {test_id}: Not found in parsed Excel data.")
                continue
                
            quality, original_test_obj = test_objects[test_id]

            if quality not in params_dict:
                print(f"  -> Skipping {test_id}: No parameters for print quality '{quality}'.")
                continue
            
            # Since the test object is a frozen dataclass, we must use dataclasses.replace()
            # to safely overwrite the arrays with the exact ones from the .h5 file
            test_for_solver = dataclasses.replace(
                original_test_obj,
                time_series=group["time_s"][:],
                strain_series=group["strain_measured"][:]
            )
            
            params = params_dict[quality]
            
            # 4. Solve the model using the fully-featured test object
            try:
                y_pred = solve_tlv(test_for_solver, params)
            except SolverConvergenceError as e:
                print(f"  -> Warning: Solver did not converge for {test_id} ({e})")
                continue
            except Exception as e:
                print(f"  -> Error solving {test_id}: {e}")
                continue
                
            # Verify arrays are identically sized before inserting
            if len(y_pred) != len(test_for_solver.time_series):
                print(f"  -> Warning: y_pred length ({len(y_pred)}) doesn't match time_s length ({len(test_for_solver.time_series)})")
                continue
                
            # 5. Overwrite the strain_predicted dataset safely
            if "strain_predicted" in group:
                del group["strain_predicted"] 
                
            group.create_dataset("strain_predicted", data=y_pred)
            success_count += 1
            print(f"  -> Successfully updated strain_predicted.")

    print(f"\nDone! Re-calculated and replaced theoretical curves for {success_count} tests.")


if __name__ == "__main__":
    main()