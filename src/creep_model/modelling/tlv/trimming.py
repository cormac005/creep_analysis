"""
Tertiary-creep trimming & print-quality partitioning (Thesis Sec. 1.3.5).

Reuses eda.stage_classification.classify_stages directly -- the thesis
explicitly states the SAME classification strategy is used here as for the
EDA stage-onset detection, so this module must not reimplement the k1/k2
logic independently.
"""
from dataclasses import replace

from creep_model.domain import CreepTest, CreepExperiment
from creep_model.eda.stage_classification import classify_stages


def trim_tertiary(test: CreepTest, k1: int, k2: int) -> CreepTest:
    """
    Remove tertiary-creep data from a single test, using classify_stages'
    secondary_end_idx as the cutoff.

    Returns a NEW CreepTest (frozen dataclass) truncated at
    classification.secondary_end_idx. If no tertiary creep was detected
    (secondary_end_idx already covers the whole test), returns the test
    unchanged.

    Args:
        test: a non-empty CreepTest.
        k1, k2: the FINAL, locked hyperparameters chosen during EDA tuning
                (see scripts/exploratory/k1_k2_sensitivity.py) -- must match
                whatever was used to produce the EDA statistics, so the same
                test isn't classified two different ways in two parts of
                the thesis.

    Raises:
        ValueError if primary_end_idx is None (no secondary creep detected
        at all under these k1/k2) -- there's no sensible trim point in that
        case. Consider whether such a test should be excluded from the TLV
        fit entirely rather than crashing the whole pipeline; if so, catch
        this in trim_and_partition below and skip with a warning, mirroring
        io/parser.py's "skip missing sheet" pattern.
    """
    classification = classify_stages(test, k1=k1, k2=k2)

    if classification.primary_end_idx is None:
        raise ValueError(
            f"Test {test.test_id}: no secondary creep detected with k1={k1}, k2={k2}; "
            "cannot determine a trim point."
        )

    end_idx = classification.secondary_end_idx
    if end_idx is None:
        return test  # test ended before tertiary creep began -- nothing to trim

    return replace(
        test,
        time_series=test.time_series[: end_idx + 1],
        strain_series=test.strain_series[: end_idx + 1],
        # temp_time_series / temperature_readings are on a DIFFERENT, coarser
        # time base -- do NOT slice them with end_idx (that index refers to
        # the strain/time_series grid). Leave them full-length;
        # interpolate_temperature() resamples onto whatever time_series ends
        # up being, post-trim, automatically.
    )


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
        print(f"Warning: skipped {len(skipped)} test(s) with no detected secondary "
              f"creep under k1={k1}, k2={k2}: {skipped}")

    return groups
