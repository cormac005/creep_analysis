"""
Integration test: real CreepData.xlsx -> parser -> classify_stages ->
build_eda_dataframe.

This intentionally asserts STRUCTURAL properties (no crashes, ordering
invariants, known physical expectations) rather than exact numeric values --
exact classification results will shift as k1/k2 are tuned (see
scripts/exploratory/k1_k2_sensitivity.py). Once you've picked final k1/k2
values and visually confirmed the classification looks right, consider
tightening this test with an explicit regression snapshot (e.g. asserting
the exact primary_end_idx/secondary_end_idx per test_id) so a future
refactor of stage_classification.py that silently changes behavior gets
caught immediately.

"""
from pathlib import Path

import pytest

from creep_model.io.parser import ExcelCreepParser
from creep_model.eda.stage_classification import classify_stages
from creep_model.eda.statistics import build_eda_dataframe

K1 = 3
K2 = 3
FAILED_SPECIMEN_IDS: list[str] = ["S.30.2", "H.30.3"]


@pytest.fixture(scope="module")
def experiment(real_data_path: Path):
    if not real_data_path.exists():
        pytest.skip(f"Real data file not found at {real_data_path} (data/ is git-ignored).")
    parser = ExcelCreepParser(real_data_path)
    return parser.load_experiment()


class TestEdaPipelineIntegration:
    def test_all_completed_tests_classify_without_raising(self, experiment):
        completed = [t for t in experiment.tests.values() if not t.is_empty]
        assert len(completed) > 0, "Expected at least one completed test in the workbook."

        classifications = {}
        for test in completed:
            classifications[test.test_id] = classify_stages(test, k1=K1, k2=K2)

        assert len(classifications) == len(completed)

    def test_boundary_ordering_is_consistent(self, experiment):
        """primary_end_idx should never come after secondary_end_idx, for
        every test where both are defined."""
        for test in experiment.tests.values():
            if test.is_empty:
                continue
            result = classify_stages(test, k1=K1, k2=K2)
            if result.primary_end_idx is not None and result.secondary_end_idx is not None:
                assert result.primary_end_idx <= result.secondary_end_idx, (
                    f"{test.test_id}: primary_end_idx={result.primary_end_idx} "
                    f"> secondary_end_idx={result.secondary_end_idx}"
                )

    @pytest.mark.skipif(
        not FAILED_SPECIMEN_IDS,
        reason="FAILED_SPECIMEN_IDS not yet filled in -- see module docstring TODO.",
    )
    def test_failed_specimens_show_tertiary_creep(self, experiment):
        """
        Sanity/anchor check: the two specimens known to have physically
        failed should show detected tertiary creep under the chosen k1/k2.
        If this fails, treat it as a signal the hyperparameters are too
        conservative -- NOT as license to tune k1/k2 specifically until this
        passes (that would be overfitting the hyperparameters to two known
        outcomes rather than genuine visual/manual tuning).
        """
        for test_id in FAILED_SPECIMEN_IDS:
            test = experiment.tests[test_id]
            result = classify_stages(test, k1=K1, k2=K2)
            assert result.has_tertiary, (
                f"Expected failed specimen {test_id} to show tertiary creep "
                f"under k1={K1}, k2={K2}, but it did not."
            )

    def test_build_eda_dataframe_end_to_end(self, experiment):
        completed = {
            tid: t for tid, t in experiment.tests.items() if not t.is_empty
        }
        classifications = {
            tid: classify_stages(t, k1=K1, k2=K2) for tid, t in completed.items()
        }

        df = build_eda_dataframe(completed, classifications)

        assert len(df) == len(completed)
        assert not df["Test_ID"].duplicated().any()