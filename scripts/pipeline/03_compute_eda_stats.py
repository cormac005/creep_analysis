"""
Compute statistics for exploratory data analysis (EDA) and save them to eda_results.h5.
This script extracts the populated EDA statistics, temperature profiles, and 
specimen attributes from tlv_fit_results.h5 and processed_experimental_data.h5.
"""
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

from creep_model.config import config

DATA_PATH = Path(config.data_output_directory) / "processed_experimental_data.h5"
TLV_PATH = Path(config.data_output_directory) / "tlv_fit_results.h5"
OUTPUT_PATH = Path(config.data_output_directory) / "eda_results.h5"


def main() -> None:
    # 1. Ensure prerequisite files exist
    source_h5 = TLV_PATH if TLV_PATH.exists() else DATA_PATH
    if not source_h5.exists():
        raise FileNotFoundError(
            f"Required data file {source_h5} does not exist. "
            "Please run 01_classify_and_trim.py and 02_fit_tlv.py first."
        )

    print(f"Extracting EDA statistics from {source_h5}...")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    summary_records = []

    with h5py.File(source_h5, "r") as f_in, h5py.File(OUTPUT_PATH, "w") as f_out:
        tests_out_group = f_out.create_group("tests")

        # Case A: Reading from tlv_fit_results.h5 (Structured under /tests/<test_id>)
        if "tests" in f_in:
            tests_grp = f_in["tests"]
            for test_id in tests_grp.keys():
                t_grp = tests_grp[test_id]
                attrs = t_grp.attrs

                print_quality = str(attrs.get("print_quality", ""))
                applied_stress = float(attrs.get("applied_stress_MPa", 0.0))
                age_days = float(attrs.get("age_days", 0.0))

                eps_tilde_0 = float(attrs.get("eps_tilde_0", 0.0))
                eps_dot_ss = float(attrs.get("eps_dot_ss", 0.0))
                if np.isnan(eps_dot_ss):
                    eps_dot_ss = 0.0

                initial_temp = float(attrs.get("initial_temp_c", 20.0))
                mean_temp_sec = float(attrs.get("mean_temp_c_secondary_creep", initial_temp))
                if np.isnan(mean_temp_sec):
                    mean_temp_sec = initial_temp

                # Save to eda_results.h5 /tests/<test_id>
                t_out = tests_out_group.create_group(test_id)
                t_out.attrs["print_quality"] = print_quality
                t_out.attrs["applied_stress_MPa"] = applied_stress
                t_out.attrs["age_days"] = age_days
                t_out.attrs["eps_tilde_0"] = eps_tilde_0
                t_out.attrs["eps_dot_ss"] = eps_dot_ss
                t_out.attrs["initial_temp_c"] = initial_temp
                t_out.attrs["mean_temp_c_secondary_creep"] = mean_temp_sec

                # Copy temperature series if available
                for d_name in ["temp_time_s", "temperature_raw", "time_s", "strain_measured"]:
                    if d_name in t_grp:
                        t_out.create_dataset(d_name, data=t_grp[d_name][:])

                summary_records.append(
                    {
                        "Test_ID": test_id,
                        "Print_Quality": print_quality,
                        "Applied_Stress_MPa": applied_stress,
                        "Age_Days": age_days,
                        "Initial_Temp_C": initial_temp,
                        "Mean_Temp_C_Secondary_Creep": mean_temp_sec,
                        "Eps_Tilde_0": eps_tilde_0,
                        "Eps_Dot_Ss": eps_dot_ss,
                    }
                )

        # Case B: Reading from processed_experimental_data.h5 (Structured under /<Quality>/<specimen>)
        else:
            for group_name in f_in.keys():
                q_grp = f_in[group_name]
                if not isinstance(q_grp, h5py.Group):
                    continue

                for spec_id in q_grp.keys():
                    test_grp = q_grp[spec_id]
                    attrs = test_grp.attrs
                    test_id = str(attrs.get("test_id", f"{group_name}_{spec_id}"))

                    print_quality = str(attrs.get("print_quality", group_name))
                    applied_stress = float(attrs.get("applied_stress_MPa", 0.0))
                    age_days = float(attrs.get("age_days", 0.0))

                    strains = test_grp["strain_series"][:] if "strain_series" in test_grp else np.array([0.0])
                    temps = test_grp["temperature_readings"][:] if "temperature_readings" in test_grp else np.array([20.0])

                    eps_tilde_0 = float(strains[0]) if len(strains) > 0 else 0.0
                    initial_temp = float(temps[0]) if len(temps) > 0 else 20.0
                    mean_temp_sec = float(np.nanmean(temps)) if len(temps) > 0 else initial_temp

                    t_out = tests_out_group.create_group(test_id)
                    t_out.attrs["print_quality"] = print_quality
                    t_out.attrs["applied_stress_MPa"] = applied_stress
                    t_out.attrs["age_days"] = age_days
                    t_out.attrs["eps_tilde_0"] = eps_tilde_0
                    t_out.attrs["eps_dot_ss"] = 0.0
                    t_out.attrs["initial_temp_c"] = initial_temp
                    t_out.attrs["mean_temp_c_secondary_creep"] = mean_temp_sec

                    summary_records.append(
                        {
                            "Test_ID": test_id,
                            "Print_Quality": print_quality,
                            "Applied_Stress_MPa": applied_stress,
                            "Age_Days": age_days,
                            "Initial_Temp_C": initial_temp,
                            "Mean_Temp_C_Secondary_Creep": mean_temp_sec,
                            "Eps_Tilde_0": eps_tilde_0,
                            "Eps_Dot_Ss": 0.0,
                        }
                    )

        # Write root eda_summary dataset
        df = pd.DataFrame(summary_records)
        if not df.empty:
            eda_group = f_out.create_group("eda_summary")
            eda_group.create_dataset(
                "Test_ID", data=np.array(df["Test_ID"], dtype=h5py.string_dtype())
            )
            eda_group.create_dataset(
                "Print_Quality", data=np.array(df["Print_Quality"], dtype=h5py.string_dtype())
            )
            for col in [
                "Applied_Stress_MPa",
                "Age_Days",
                "Initial_Temp_C",
                "Mean_Temp_C_Secondary_Creep",
                "Eps_Tilde_0",
                "Eps_Dot_Ss",
            ]:
                eda_group.create_dataset(col, data=df[col].to_numpy(dtype=np.float64))

    print(
        f"Successfully extracted EDA statistics and saved to {OUTPUT_PATH} ({len(summary_records)} test(s))."
    )


if __name__ == "__main__":
    main()