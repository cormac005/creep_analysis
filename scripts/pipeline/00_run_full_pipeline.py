"""
00_run_full_pipeline.py

Master pipeline runner that executes all data preparation, model fitting, 
EDA computation, plotting, and LaTeX table generation steps in sequence.

Pipeline Execution Sequence:
    1. 01_classify_and_trim.py           - Parse raw Excel, trim tertiary creep, save to HDF5.
    2. 02_fit_tlv.py                     - Fit TLV model parameters per print quality group.
        WARNING: This step can take up to 60x longer than the other steps combined, depending on the number of tests and DE hyperparameters.
    3. 03_compute_eda_stats.py           - Compute EDA summary statistics & thermal histories.
    4. 04_generate_tlv_plots_and_tables.py - Generate strain-vs-time plots & TLV LaTeX tables.
    5. 05_generate_eda_plots_and_tables.py - Generate EDA plots & summary LaTeX tables.
"""

import sys
import time
import subprocess
from pathlib import Path

# List pipeline scripts in strict execution order
PIPELINE_SCRIPTS = [
    "01_classify_and_trim.py",
    "02_fit_tlv.py",
    "03_compute_eda_stats.py",
    "04_generate_tlv_plots_and_tables.py",
    "05_generate_eda_plots_and_tables.py",
]


def run_script(script_name: str, script_dir: Path) -> float:
    """
    Executes a single pipeline script in an isolated subprocess.

    Args:
        script_name: Filename of the target script.
        script_dir: Directory path containing the script.

    Returns:
        float: Execution duration in seconds.

    Raises:
        RuntimeError: If the script returns a non-zero exit code.
    """
    script_path = script_dir / script_name

    if not script_path.exists():
        raise FileNotFoundError(f"Pipeline script not found: {script_path}")

    print(f"\n{'=' * 75}")
    print(f" Running: {script_name}")
    print(f"{'=' * 75}")

    start_time = time.perf_counter()

    # Execute script using the current Python executable
    result = subprocess.run(
        [sys.executable, str(script_path)],
        check=False,
    )

    elapsed_time = time.perf_counter() - start_time

    if result.returncode != 0:
        raise RuntimeError(
            f"Pipeline failed at step '{script_name}' with exit code {result.returncode}."
        )

    print(f" Finished {script_name} in {elapsed_time:.2f}s")
    return elapsed_time


def main() -> None:
    pipeline_dir = Path(__file__).resolve().parent
    total_start_time = time.perf_counter()
    execution_times: dict[str, float] = {}

    print("\n" + "=" * 75)
    print(" STARTING FULL CREEP MODEL PIPELINE")
    print("=" * 75)

    try:
        for script in PIPELINE_SCRIPTS:
            duration = run_script(script, pipeline_dir)
            execution_times[script] = duration

    except Exception as err:
        print(f"\n[PIPELINE FAILURE] {err}", file=sys.stderr)
        sys.exit(1)

    total_duration = time.perf_counter() - total_start_time

    print("\n" + "=" * 75)
    print(" PIPELINE EXECUTION SUMMARY")
    print("=" * 75)
    for script_name, duration in execution_times.items():
        print(f"  - {script_name:<38} : {duration:>7.2f}s")
    print("-" * 75)
    print(f"  Total Pipeline Time                     : {total_duration:>7.2f}s")
    print("=" * 75 + "\n")


if __name__ == "__main__":
    main()