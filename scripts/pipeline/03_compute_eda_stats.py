"""
Compute EDA summary statistics (eps_tilde_0, eps_dot_ss, initial/mean
temperature, has_tertiary) directly from the raw workbook via stage classification --
independent of the TLV model fit, so this can be re-run in seconds
without waiting on (or requiring) 02_fit_tlv.py's search.
"""
from pathlib import Path

import h5py
import numpy as np

from creep_model.config import config
from creep_model.io.parser import ExcelCreepParser
from creep_model.eda.stage_classification import classify_stages
from creep_model.eda.statistics import build_eda_dataframe

DATA_PATH = Path(config.data_directory) / "CreepData.xlsx"
OUTPUT_PATH = Path(config.data_output_directory) / "eda_results.h5"

# Must match the k1/k2 used for trimming elsewhere in the pipeline.
K1 = config.K1
K2 = config.K2


def main() -> None:
    parser = ExcelCreepParser(DATA_PATH)
    experiment = parser.load_experiment()

    completed = {tid: t for tid, t in experiment.tests.items() if not t.is_empty}
    classifications = {
        tid: classify_stages(t, k1=K1, k2=K2) for tid, t in completed.items()
    }
    df = build_eda_dataframe(completed, classifications)

    # 1. Record initial temperature at t=0
    df["Initial_Temp_C"] = [
        completed[tid].temperature_readings[0] for tid in df["Test_ID"]
    ]

    # 2. Record tertiary creep detection flag directly from stage classifications
    df["Has_Tertiary"] = [
        classifications[tid].has_tertiary for tid in df["Test_ID"]
    ]

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(OUTPUT_PATH, "w") as f:
        eda_group = f.create_group("eda_summary")
        eda_group.create_dataset(
            "Test_ID", data=np.array(df["Test_ID"], dtype=h5py.string_dtype())
        )
        eda_group.create_dataset(
            "Print_Quality", data=np.array(df["Print_Quality"], dtype=h5py.string_dtype())
        )
        
        # Write numeric and boolean columns to HDF5
        for col in [
            "Applied_Stress_MPa", "Age_Days", "Initial_Temp_C",
            "Mean_Temp_C_Secondary_Creep", "Eps_Tilde_0", "Eps_Dot_Ss",
            "Has_Tertiary",
        ]:
            dtype = np.bool_ if col == "Has_Tertiary" else np.float64
            eda_group.create_dataset(col, data=df[col].to_numpy(dtype=dtype))

    print(f"Computed EDA statistics for {len(df)} test(s) and saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()