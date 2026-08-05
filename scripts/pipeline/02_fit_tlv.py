"""
Fit the TLV model to experimental creep data, using the pre-processed and partitioned data, and save the fitted parameters and predicted strain data to an HDF5 file.
This script should be run AFTER the data has been processed and partitioned by print quality (see scripts/pipeline/01_classify_and_trim.py).
The output HDF5 file will contain:
- Fitted TLV parameters for each print quality group
- Predicted vs. measured strain data for each test
- Summary statistics like RMSE
"""
from dataclasses import fields
from pathlib import Path

import h5py

from creep_model.config import config
from creep_model.modelling.optimisation.bounds import TLVBounds
from creep_model.modelling.tlv.fit_pipeline import fit_group
from creep_model.modelling.tlv.solver import SolverConvergenceError, solve_tlv

# --- Configuration ---
DATA_PATH = Path(config.data_output_directory) / "processed_experimental_data.h5"
OUTPUT_PATH = Path(config.data_output_directory) / "tlv_fit_results.h5"

# Get DE hyperparameters from config.py
DE_KWARGS = config.DE_KWARGS


def main() -> None:
    # Load the experimental data
    with h5py.File(DATA_PATH, "r") as f:
        groups = {}
        for group_name in f.keys():
            group = f[group_name]
            tests = []
            for test_id in group.keys():
                test_group = group[test_id]
                test_data = {
                    "test_id": test_id,
                    "print_quality": group_name,
                    "applied_stress_MPa": test_group.attrs["applied_stress_MPa"],
                    "age_days": test_group.attrs["age_days"],
                    "time_s": test_group["time_s"][:],
                    "strain_measured": test_group["strain_measured"][:],
                }
                tests.append(test_data)
            groups[group_name] = tests

    # Fit TLV models separately per print-quality group
    fitted_params = {}
    for quality, tests in groups.items():
        if not tests:
            print(f"Warning: no tests found for print quality '{quality}', skipping.")
            continue

        print(f"Fitting TLV model for '{quality}' quality ({len(tests)} tests)...")
        bounds = TLVBounds.from_group_data(tests)
        params = fit_group(tests, bounds, de_kwargs=DE_KWARGS)
        fitted_params[quality] = params
        print(f"  -> {params}")

    # Get all data: predicted vs. measured strain, per test, using each
    # test's own group's fitted parameters
    records = []
    for quality, tests in groups.items():
        params = fitted_params.get(quality)
        if params is None:
            continue

        for test in tests:
            try:
                y_pred = solve_tlv(test, params)
            except SolverConvergenceError as e:
                print(
                    f"Warning: solver did not converge for {test['test_id']} "
                    f"at the final fitted parameters ({e}); skipping."
                )
                continue

            records.append(
                {
                    "test_id": test["test_id"],
                    "print_quality": quality,
                    "applied_stress_MPa": test["applied_stress_MPa"],
                    "age_days": test["age_days"],
                    "time_s": test["time_s"],
                    "strain_measured": test["strain_measured"],
                    "strain_predicted": y_pred,
                }
            )

    # Package data and save to data/processed as an HDF5 file
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(OUTPUT_PATH, "w") as f:
        # One group per print quality, storing the 10 fitted TLV parameters
        # as attributes (named, so they're self-describing when reopened).
        params_group = f.create_group("fitted_parameters")
        for quality, params in fitted_params.items():
            g = params_group.create_group(quality)
            for field, value in zip(fields(params), params.to_array()):
                g.attrs[field.name] = value

        # Group tests under their quality partition to avoid test_id collisions across groups
        tests_group = f.create_group("tests")
        for record in records:
            # Create path tests/<quality>/<test_id>
            quality_grp = tests_group.require_group(record["print_quality"])
            t_group = quality_grp.create_group(record["test_id"])

            t_group.attrs["print_quality"] = record["print_quality"]
            t_group.attrs["applied_stress_MPa"] = record["applied_stress_MPa"]
            t_group.attrs["age_days"] = record["age_days"]
            t_group.create_dataset("time_s", data=record["time_s"])
            t_group.create_dataset("strain_measured", data=record["strain_measured"])
            t_group.create_dataset("strain_predicted", data=record["strain_predicted"])

    print(f"Saved fit results for {len(records)} test(s) to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()