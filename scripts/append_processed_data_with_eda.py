"""
Append EDA summary statistics and temperature arrays to an already-produced 
tlv_fit_results.h5, without re-running the expensive DE->LM fit in 
process_data_and_fit.py.

Run this AFTER process_data_and_fit.py -- it opens that file in APPEND
mode and adds data alongside the existing fitted_parameters/tests groups,
it does not touch or recompute the model parameters already in there.
"""
from pathlib import Path

import h5py
import numpy as np

from creep_model.io.parser import ExcelCreepParser
from creep_model.eda.stage_classification import classify_stages
from creep_model.eda.statistics import build_eda_dataframe

DATA_PATH = Path("data/raw/CreepData.xlsx")
OUTPUT_PATH = Path("data/processed/tlv_fit_results.h5")

# MUST match the k1/k2 used for trimming in process_data_and_fit.py --
# a mismatch here would silently classify stages differently from what was
# actually used to produce the fitted parameters already stored in the file.
K1 = 3
K2 = 3


def main() -> None:
    if not OUTPUT_PATH.exists():
        raise FileNotFoundError(
            f"{OUTPUT_PATH} does not exist yet -- run process_data_and_fit.py first."
        )

    print(f"Loading raw data from {DATA_PATH}...")
    parser = ExcelCreepParser(DATA_PATH)
    experiment = parser.load_experiment()

    completed = {tid: t for tid, t in experiment.tests.items() if not t.is_empty}
    classifications = {
        tid: classify_stages(t, k1=K1, k2=K2) for tid, t in completed.items()
    }
    df = build_eda_dataframe(completed, classifications)

    # --- NEW: Calculate Initial_Temp_C for every test ---
    initial_temps = []
    for test_id in df["Test_ID"]:
        t_obj = experiment.tests[test_id]
        # Get the very first timestamp recorded for strain
        t0 = t_obj.time_series[0]
        # Evaluate the temperature polynomial at t0 (with clamping just to be safe)
        t0_clamped = np.clip(t0, t_obj.temp_time_series.min(), t_obj.temp_time_series.max())
        temp_0 = t_obj.temperature_polynomial()(t0_clamped)
        initial_temps.append(temp_0)
        
    # Append it as a new column to the dataframe
    df["Initial_Temp_C"] = initial_temps

    print(f"Opening {OUTPUT_PATH} to append EDA stats and Temperature data...")
    with h5py.File(OUTPUT_PATH, "a") as f:
        # 1. Attach per-test EDA stats and Temperature arrays onto the existing per-test groups
        tests_group = f.require_group("tests")
        for _, row in df.iterrows():
            test_id = row["Test_ID"]
            
            if test_id not in tests_group:
                t_group = tests_group.create_group(test_id)
            else:
                t_group = tests_group[test_id]

            # --- Append EDA Attributes ---
            t_group.attrs["eps_tilde_0"] = row["Eps_Tilde_0"]
            t_group.attrs["eps_dot_ss"] = row["Eps_Dot_Ss"]
            t_group.attrs["initial_temp_c"] = row["Initial_Temp_C"] # Added this!
            t_group.attrs["mean_temp_c_secondary_creep"] = row["Mean_Temp_C_Secondary_Creep"]
            t_group.attrs["k1"] = K1
            t_group.attrs["k2"] = K2

            # --- Append Temperature Datasets ---
            t_obj = experiment.tests[test_id]
            
            # Safely delete existing temperature datasets if re-running this script
            for ds in ["temp_time_s", "temperature_raw", "temperature_interpolated"]:
                if ds in t_group:
                    del t_group[ds]
            
            # Save the raw, coarse temperature grid
            t_group.create_dataset("temp_time_s", data=t_obj.temp_time_series)
            t_group.create_dataset("temperature_raw", data=t_obj.temperature_readings)
            
            # If the test was successfully fitted, it will have a 'time_s' array. 
            if "time_s" in t_group:
                time_s = t_group["time_s"][:]
                poly = t_obj.temperature_polynomial()
                
                # Apply the same clamping logic from domain.py
                t_clamped = np.clip(
                    time_s,
                    t_obj.temp_time_series.min(),
                    t_obj.temp_time_series.max()
                )
                
                temperature_interpolated = poly(t_clamped)
                t_group.create_dataset("temperature_interpolated", data=temperature_interpolated)

        # 2. Store one consolidated EDA table at the root
        if "eda_summary" in f:
            del f["eda_summary"]  # overwrite cleanly if this script is re-run
        eda_group = f.create_group("eda_summary")
        eda_group.create_dataset(
            "Test_ID", data=np.array(df["Test_ID"], dtype=h5py.string_dtype())
        )
        eda_group.create_dataset(
            "Print_Quality", data=np.array(df["Print_Quality"], dtype=h5py.string_dtype())
        )
        for col in [
            "Applied_Stress_MPa", "Age_Days", "Initial_Temp_C", # Added Initial_Temp_C here!
            "Mean_Temp_C_Secondary_Creep", "Eps_Tilde_0", "Eps_Dot_Ss",
        ]:
            eda_group.create_dataset(col, data=df[col].to_numpy(dtype=np.float64))

    print(f"Successfully appended EDA statistics and Temperature arrays for {len(df)} test(s).")


if __name__ == "__main__":
    main()