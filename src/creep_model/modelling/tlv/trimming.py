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
    ...
    Raises:
        Nothing anymore for the "primary-only" case -- see below.
    """
    classification = classify_stages(test, k1=k1, k2=k2)

    if classification.primary_end_idx is None:
        # No secondary-creep onset detected at all under this k1/k2 -- the
        # test never left primary creep. Tertiary creep can only begin
        # AFTER secondary onset, so there is nothing to trim here; include
        # the test unchanged rather than discarding it.
        return test

    end_idx = classification.secondary_end_idx
    if end_idx is None:
        return test  # test ended before tertiary creep began -- nothing to trim

    return replace(
        test,
        time_series=test.time_series[: end_idx + 1],
        strain_series=test.strain_series[: end_idx + 1],
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
