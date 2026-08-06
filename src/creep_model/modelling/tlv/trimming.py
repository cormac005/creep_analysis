"""
Creep stage trimming & print-quality partitioning.

Reuses eda.stage_classification.classify_stages directly to detect stage boundaries,
trimming secondary and tertiary creep stages as well as any strain measurements
recorded after the final temperature reading.
"""
from dataclasses import replace
import numpy as np

from creep_model.domain import CreepTest, CreepExperiment
from creep_model.eda.stage_classification import classify_stages


def trim_tertiary(test: CreepTest, k1: int, k2: int) -> CreepTest:
    """
    Trims tertiary creep data (if detected) as well as any trailing data points 
    recorded after the final temperature reading.
    """
    cutoff_idx = len(test.time_series) - 1

    # 1. Stage-based trimming: Remove tertiary creep ONLY if detected
    classification = classify_stages(test, k1=k1, k2=k2)

    if classification.has_tertiary and classification.secondary_end_idx is not None:
        # Trim to the end of the secondary stage (inclusive)
        cutoff_idx = min(cutoff_idx, classification.secondary_end_idx)

    # 2. Temperature-based trimming: Remove data past the final temperature reading
    if (
        hasattr(test, "temp_time_series")
        and test.temp_time_series is not None
        and len(test.temp_time_series) > 0
    ):
        max_temp_time = test.temp_time_series[-1]
        valid_temp_indices = np.where(test.time_series <= max_temp_time)[0]
        if len(valid_temp_indices) > 0:
            cutoff_idx = min(cutoff_idx, valid_temp_indices[-1])

    # Apply trimming if a smaller cutoff index was found
    if cutoff_idx < len(test.time_series) - 1:
        return replace(
            test,
            time_series=test.time_series[: cutoff_idx + 1],
            strain_series=test.strain_series[: cutoff_idx + 1],
        )

    return test

# Alias for semantics
trim_test = trim_tertiary


def trim_and_partition(
    experiment: CreepExperiment, k1: int, k2: int
) -> dict[str, list[CreepTest]]:
    """
    Apply trim_tertiary to every non-empty test and group results by
    print_quality, ready for optimization.fit_pipeline.fit_group().

    Returns:
        {"High": [CreepTest, ...], "Standard": [CreepTest, ...]}
    """
    groups: dict[str, list[CreepTest]] = {"High": [], "Standard": []}
    skipped = []
    for test in experiment.tests.values():
        if test.is_empty:
            continue
        try:
            trimmed = trim_tertiary(test, k1=k1, k2=k2)
        except ValueError:
            skipped.append(test.test_id)
            continue
        groups.setdefault(trimmed.print_quality, []).append(trimmed)

    if skipped:
        print(
            f"Warning: skipped {len(skipped)} test(s) with error during trimming "
            f"under k1={k1}, k2={k2}: {skipped}"
        )

    return groups