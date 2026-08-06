"""
Fit the TLV model to experimental creep data, using the pre-processed and partitioned data, 
and save the fitted parameters and predicted strain data to an HDF5 file.
"""
from dataclasses import fields
from pathlib import Path

import h5py

from creep_model.config import config
from creep_model.domain import CreepTest
from creep_model.modelling.optimisation.bounds import TLVBounds
from creep_model.modelling.tlv.fit_pipeline import fit_group
from creep_model.modelling.tlv.solver import SolverConvergenceError, solve_tlv

DATA_PATH = Path(config.data_output_directory) / "processed_experimental_data.h5"
OUTPUT_PATH = Path(config.data_output_directory) / "tlv_fit_results.h5"


def _load_creep_test(test_group: h5py.Group, print_quality: str) -> CreepTest:
    return CreepTest(
        test_id=str(test_group.attrs["test_id"]),
        time_series=test_group["time_series"][:],
        strain_series=test_group["strain_series"][:],
        temp_time_series=test_group["temp_time_series"][:],
        temperature_readings=test_group["temperature_readings"][:],
        applied_stress_MPa=float(test_group.attrs["applied_stress_MPa"]),
        age_days=int(test_group.attrs["age_days"]),
        print_quality=print_quality,
    )


def main() -> None:
    with h5py.File(DATA_PATH, "r") as f:
        groups: dict[str, list[CreepTest]] = {}
        for group_name in f.keys():
            group = f[group_name]
            groups[group_name] = [
                _load_creep_test(group[key], print_quality=group_name)
                for key in group.keys()
            ]

    fitted_params = {}
    for quality, tests in groups.items():
        if not tests:
            print(f"Warning: no tests found for print quality '{quality}', skipping.")
            continue

        print(f"Fitting TLV model for '{quality}' quality ({len(tests)} tests)...")
        bounds = TLVBounds.from_group_data(tests)
        params = fit_group(tests, bounds)
        fitted_params[quality] = params
        print(f"  -> {params}")

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
                    f"Warning: solver did not converge for {test.test_id} "
                    f"at the final fitted parameters ({e}); skipping."
                )
                continue

            records.append(
                {
                    "test_id": test.test_id,
                    "print_quality": quality,
                    "applied_stress_MPa": test.applied_stress_MPa,
                    "age_days": test.age_days,
                    "time_s": test.time_series,
                    "strain_measured": test.strain_series,
                    "strain_predicted": y_pred,
                }
            )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(OUTPUT_PATH, "w") as f:
        params_group = f.create_group("fitted_parameters")
        for quality, params in fitted_params.items():
            g = params_group.create_group(quality)
            for field, value in zip(fields(params), params.to_array()):
                g.attrs[field.name] = value

        tests_group = f.create_group("tests")
        for record in records:
            t_group = tests_group.create_group(record["test_id"])

            t_group.attrs["print_quality"] = record["print_quality"]
            t_group.attrs["applied_stress_MPa"] = record["applied_stress_MPa"]
            t_group.attrs["age_days"] = record["age_days"]
            t_group.create_dataset("time_s", data=record["time_s"])
            t_group.create_dataset("strain_measured", data=record["strain_measured"])
            t_group.create_dataset("strain_predicted", data=record["strain_predicted"])

    print(f"Saved fit results for {len(records)} test(s) to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()