"""
Compute statistics for exploratory data analysis (EDA) and save them to eda_results.h5.
This script reads the pre-processed data from processed_experimental_data.h5 and saves
the summary statistics and temperature records without reloading the raw experimental data.

Inputs:
    - data/processed/processed_experimental_data.h5

Outputs:
    - data/processed/eda_results.h5
"""
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

from creep_model.config import config

DATA_PATH = Path(config.data_output_directory) / "processed_experimental_data.h5"
OUTPUT_PATH = Path(config.data_output_directory) / "eda_results.h5"


def main() -> None:
    # 1. Ensure pre-requisite processed data file exists
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Required processed data file {DATA_PATH} does not exist. "
            "Please run 01_classify_and_trim.py first."
        )

    print(f"Reading processed experimental data from {DATA_PATH}...")

    summary_records = []
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(DATA_PATH, "r") as f_in, h5py.File(OUTPUT_PATH, "w") as f_out:
        tests_out_group = f_out.create_group("tests")

        # Iterate through quality groups (e.g., "High", "Standard")
        for group_name in f_in.keys():
            quality_group = f_in[group_name]

            if not isinstance(quality_group, h5py.Group):
                continue

            for test_id in quality_group.keys():
                test_grp = quality_group[test_id]
                if not isinstance(test_grp, h5py.Group):
                    continue

                # --- FIX: Ensure globally unique test IDs ---
                # If IDs like 'specimen_1' repeat across groups, prefix them.
                unique_test_id = (
                    f"{group_name}_{test_id}" 
                    if not test_id.startswith(group_name) 
                    else test_id
                )

                attrs = test_grp.attrs

                # Extract basic attributes
                applied_stress = float(attrs.get("applied_stress_MPa", 0.0))
                age_days = float(attrs.get("age_days", 0.0))
                print_quality = str(attrs.get("print_quality", group_name))

                # Extract time and strain datasets
                time_s = test_grp["time_s"][:] if "time_s" in test_grp else None
                strain_measured = (
                    test_grp["strain_measured"][:]
                    if "strain_measured" in test_grp
                    else None
                )

                # Extract or compute initial strain (Eps_Tilde_0)
                eps_tilde_0 = attrs.get("eps_tilde_0", attrs.get("Eps_Tilde_0", None))
                if eps_tilde_0 is None:
                    if strain_measured is not None and len(strain_measured) > 0:
                        eps_tilde_0 = float(strain_measured[0])
                    else:
                        eps_tilde_0 = 0.0
                else:
                    eps_tilde_0 = float(eps_tilde_0)

                # Extract or compute secondary creep rate (Eps_Dot_Ss)
                eps_dot_ss = attrs.get("eps_dot_ss", attrs.get("Eps_Dot_Ss", None))
                if eps_dot_ss is None:
                    if (
                        time_s is not None
                        and strain_measured is not None
                        and len(time_s) > 10
                    ):
                        # Linear fit on the secondary creep portion (second half)
                        mid = len(time_s) // 2
                        slope, _ = np.polyfit(time_s[mid:], strain_measured[mid:], 1)
                        eps_dot_ss = float(slope)
                    else:
                        eps_dot_ss = 0.0
                else:
                    eps_dot_ss = float(eps_dot_ss)

                # Extract temperature series if present
                temp_time_s = (
                    test_grp["temp_time_s"][:]
                    if "temp_time_s" in test_grp
                    else (time_s if time_s is not None else np.array([]))
                )
                temp_raw = (
                    test_grp["temperature_raw"][:]
                    if "temperature_raw" in test_grp
                    else (
                        test_grp["temp_array"][:]
                        if "temp_array" in test_grp
                        else np.array([])
                    )
                )

                # Initial Temperature
                if "initial_temp_c" in attrs:
                    initial_temp = float(attrs["initial_temp_c"])
                elif len(temp_raw) > 0:
                    initial_temp = float(temp_raw[0])
                else:
                    initial_temp = float(attrs.get("Initial_Temp_C", 20.0))

                # Secondary Creep Mean Temperature
                if "mean_temp_c_secondary_creep" in attrs:
                    mean_temp_sec = float(attrs["mean_temp_c_secondary_creep"])
                elif len(temp_raw) > 0:
                    mean_temp_sec = float(np.nanmean(temp_raw))
                else:
                    mean_temp_sec = float(
                        attrs.get("Mean_Temp_C_Secondary_Creep", initial_temp)
                    )

                # Save per-test group in eda_results.h5 using the UNIQUE ID
                t_out = tests_out_group.create_group(unique_test_id)
                t_out.attrs["print_quality"] = print_quality
                t_out.attrs["applied_stress_MPa"] = applied_stress
                t_out.attrs["age_days"] = age_days
                t_out.attrs["eps_tilde_0"] = eps_tilde_0
                t_out.attrs["eps_dot_ss"] = eps_dot_ss
                t_out.attrs["initial_temp_c"] = initial_temp
                t_out.attrs["mean_temp_c_secondary_creep"] = mean_temp_sec

                if len(temp_time_s) > 0:
                    t_out.create_dataset("temp_time_s", data=temp_time_s)
                if len(temp_raw) > 0:
                    t_out.create_dataset("temperature_raw", data=temp_raw)
                if time_s is not None:
                    t_out.create_dataset("time_s", data=time_s)
                if strain_measured is not None:
                    t_out.create_dataset("strain_measured", data=strain_measured)

                # Append record for summary table
                summary_records.append(
                    {
                        "Test_ID": unique_test_id,
                        "Print_Quality": print_quality,
                        "Applied_Stress_MPa": applied_stress,
                        "Age_Days": age_days,
                        "Initial_Temp_C": initial_temp,
                        "Mean_Temp_C_Secondary_Creep": mean_temp_sec,
                        "Eps_Tilde_0": eps_tilde_0,
                        "Eps_Dot_Ss": eps_dot_ss,
                    }
                )

        # Build root eda_summary dataset table
        df = pd.DataFrame(summary_records)
        if not df.empty:
            eda_group = f_out.create_group("eda_summary")
            eda_group.create_dataset(
                "Test_ID",
                data=np.array(df["Test_ID"], dtype=h5py.string_dtype()),
            )
            eda_group.create_dataset(
                "Print_Quality",
                data=np.array(df["Print_Quality"], dtype=h5py.string_dtype()),
            )
            for col in [
                "Applied_Stress_MPa",
                "Age_Days",
                "Initial_Temp_C",
                "Mean_Temp_C_Secondary_Creep",
                "Eps_Tilde_0",
                "Eps_Dot_Ss",
            ]:
                eda_group.create_dataset(
                    col, data=df[col].to_numpy(dtype=np.float64)
                )

    print(
        f"Successfully calculated EDA statistics and saved to {OUTPUT_PATH} ({len(summary_records)} test(s))."
    )


if __name__ == "__main__":
    main()