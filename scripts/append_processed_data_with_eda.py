"""
Append EDA summary statistics (eps_tilde_0, eps_dot_ss, mean secondary-creep
temperature) to an already-produced tlv_fit_results.h5, without re-running
the expensive DE->LM fit in process_data_and_fit.py.

Run this AFTER process_data_and_fit.py -- it opens that file in APPEND
mode and adds data alongside the existing fitted_parameters/tests groups,
it does not touch or recompute anything already in there.
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

    parser = ExcelCreepParser(DATA_PATH)
    experiment = parser.load_experiment()

    completed = {tid: t for tid, t in experiment.tests.items() if not t.is_empty}
    classifications = {
        tid: classify_stages(t, k1=K1, k2=K2) for tid, t in completed.items()
    }
    df = build_eda_dataframe(completed, classifications)

    with h5py.File(OUTPUT_PATH, "a") as f:
        # 1. Attach per-test EDA stats onto the existing per-test groups
        #    (created by process_data_and_fit.py), so anything plotting a
        #    single test can pull both the TLV fit arrays and the EDA
        #    stats from one place.
        tests_group = f.require_group("tests")
        for _, row in df.iterrows():
            test_id = row["Test_ID"]
            if test_id not in tests_group:
                # This test had no successful TLV fit entry (e.g. the
                # solver never converged for it during LM) -- still record
                # its EDA stats rather than silently dropping them.
                t_group = tests_group.create_group(test_id)
            else:
                t_group = tests_group[test_id]

            t_group.attrs["eps_tilde_0"] = row["Eps_Tilde_0"]
            t_group.attrs["eps_dot_ss"] = row["Eps_Dot_Ss"]
            t_group.attrs["mean_temp_c_secondary_creep"] = row["Mean_Temp_C_Secondary_Creep"]
            t_group.attrs["k1"] = K1
            t_group.attrs["k2"] = K2

        # 2. Also store one consolidated table at the root -- convenient
        #    for loading straight into a DataFrame for the EDA plots
        #    without walking every test group individually.
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
            "Applied_Stress_MPa", "Age_Days",
            "Mean_Temp_C_Secondary_Creep", "Eps_Tilde_0", "Eps_Dot_Ss",
        ]:
            eda_group.create_dataset(col, data=df[col].to_numpy(dtype=np.float64))

    print(f"Appended EDA statistics for {len(df)} test(s) to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()