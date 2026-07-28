"""
Full pipeline: raw workbook -> trimmed/partitioned tests -> fitted TLV
parameters per print-quality group -> predicted-vs-measured strain for
every test -> persisted to data/processed/ as HDF5.

This is the Phase 4 "run it once, cache the result" script referenced in
the project plan -- DE is slow, so re-run this only when k1/k2, bounds, or
the model itself changes; everything downstream (plotting scripts) should
load the HDF5 output rather than re-fitting.
"""
from dataclasses import fields
from pathlib import Path
import h5py
from creep_model.io.parser import ExcelCreepParser
from creep_model.modelling.tlv.trimming import trim_and_partition
from creep_model.modelling.tlv.fit_pipeline import fit_group
from creep_model.modelling.tlv.solver import solve_tlv, SolverConvergenceError
from creep_model.modelling.optimisation.bounds import TLVBounds

# --- Configuration ---
DATA_PATH = Path("data/raw/CreepData.xlsx")
OUTPUT_PATH = Path("data/processed/tlv_fit_results.h5")

# Locked hyperparameters from EDA tuning (see scripts/exploratory/k1k2_manual_tuning.py)
K1 = 3
K2 = 3

# Passed to scipy.optimize.differential_evolution -- seed pinned for
# reproducibility (a fit you can't reproduce isn't thesis-defensible),
# workers=-1 to parallelise across the population.
DE_KWARGS = {"seed": 42, "workers": -1}


def main() -> None:
    # Load the experimental data
    parser = ExcelCreepParser(DATA_PATH)
    experiment = parser.load_experiment()

    # Trim tertiary creep and partition by print quality in one step --
    # trim_and_partition already returns {"High": [...], "Standard": [...]}
    groups = trim_and_partition(experiment, k1=K1, k2=K2)

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
                print(f"Warning: solver did not converge for {test.test_id} "
                      f"at the final fitted parameters ({e}); skipping.")
                continue

            records.append({
                "test_id": test.test_id,
                "print_quality": quality,
                "applied_stress_MPa": test.applied_stress_MPa,
                "age_days": test.age_days,
                "time_s": test.time_series,
                "strain_measured": test.strain_series,
                "strain_predicted": y_pred,
            })

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

        # One group per test, storing time/measured/predicted strain arrays
        # plus metadata as attributes.
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