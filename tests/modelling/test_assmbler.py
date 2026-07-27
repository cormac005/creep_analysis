"""
Tests for creep_model.modelling.assembler.DataAssembler.

NOTE ON FILENAME: this file is named test_assmbler.py (missing an 'a') to
match the existing filename already in the repo/CI history -- renaming it
to test_assembler.py would work fine with pytest's rootdir-based collection,
but is a one-line rename Cormac can make separately if he wants it fixed;
not changed here to avoid an unrelated diff in this pass.
"""
import numpy as np
import pandas as pd
import pytest

from creep_model.modelling.assembler import DataAssembler
from creep_model.domain import CreepExperiment

class TestDataAssemblerInitialization:
    def test_default_initialization(self):
        assembler = DataAssembler()
        assert assembler.normalise is False

    def test_custom_initialization(self):
        assembler = DataAssembler(normalise=True)
        assert assembler.normalise is True
        
class TestGetLocalData:
    def test_shapes_and_values(self, make_test):
        test = make_test(
            strain_series=[0.001, 0.002, 0.003],
            time_series=[0.0, 5.0, 10.0],
        )
        X, y = DataAssembler.get_local_data(test)

        assert X.shape == (3, 1)
        assert y.shape == (3,)
        np.testing.assert_allclose(X.flatten(), [0.0, 5.0, 10.0])
        np.testing.assert_allclose(y, [0.001, 0.002, 0.003])


class TestGetGlobalData:
    def test_filters_by_print_quality_and_stacks(self, make_test):
        high_a = make_test(
            strain_series=[0.001, 0.002], time_series=[0.0, 10.0],
            print_quality="High", applied_stress_MPa=20.0, test_id="H.20.1",
        )
        high_b = make_test(
            strain_series=[0.003], time_series=[0.0],
            print_quality="High", applied_stress_MPa=30.0, test_id="H.30.1",
        )
        standard = make_test(
            strain_series=[0.004, 0.005], time_series=[0.0, 10.0],
            print_quality="Standard", applied_stress_MPa=10.0, test_id="S.10.1",
        )
        experiment = CreepExperiment(
            tests={"H.20.1": high_a, "H.30.1": high_b, "S.10.1": standard}
        )

        X, y = DataAssembler.get_global_data(experiment, "High")

        # 2 rows from high_a + 1 row from high_b = 3 rows total, 2 columns [time, stress]
        assert X.shape == (3, 2)
        assert y.shape == (3,)
        # Standard-quality data must be excluded entirely
        assert not np.any(X[:, 1] == 10.0)
        # Stress column correctly broadcast per-test
        np.testing.assert_allclose(sorted(X[:, 1]), [20.0, 20.0, 30.0])

    def test_skips_empty_tests(self, make_test):
        completed = make_test(
            strain_series=[0.001, 0.002], time_series=[0.0, 10.0],
            print_quality="High", applied_stress_MPa=20.0, test_id="H.20.1",
        )
        empty = make_test(
            strain_series=[], print_quality="High", applied_stress_MPa=20.0, test_id="H.20.2",
        )
        experiment = CreepExperiment(tests={"H.20.1": completed, "H.20.2": empty})

        X, y = DataAssembler.get_global_data(experiment, "High")
        assert X.shape == (2, 2)

    def test_raises_when_no_matching_quality(self, make_test):
        standard = make_test(
            strain_series=[0.001], time_series=[0.0],
            print_quality="Standard", test_id="S.10.1",
        )
        experiment = CreepExperiment(tests={"S.10.1": standard})

        with pytest.raises(ValueError, match="No data found for Print Quality"):
            DataAssembler.get_global_data(experiment, "High")


class TestGetSummaryDataframe:
    def test_standardises_to_shortest_completed_test(self, make_test):
        # Shortest completed test duration is 5.0s (test 'short')
        short = make_test(
            strain_series=[0.001, 0.0015], time_series=[0.0, 5.0],
            test_id="short", applied_stress_MPa=10.0, age_days=2,
            print_quality="Standard", temperature_readings=[20.0, 20.0],
        )
        long = make_test(
            strain_series=[0.002, 0.003, 0.005], time_series=[0.0, 5.0, 20.0],
            test_id="long", applied_stress_MPa=20.0, age_days=3,
            print_quality="High", temperature_readings=[21.0, 21.0, 21.0],
        )
        empty = make_test(strain_series=[], test_id="empty")

        experiment = CreepExperiment(tests={"short": short, "long": long, "empty": empty})
        df = DataAssembler.get_summary_dataframe(experiment)

        assert isinstance(df, pd.DataFrame)
        # Empty test should be silently skipped
        assert len(df) == 2
        assert set(df["Test_ID"]) == {"short", "long"}

        expected_columns = {
            "Test_ID", "Applied_Stress_MPa", "Age_Days", "Print_Quality",
            "Mean_Temp_C", "Eps_0", "Eps_Max_Std", "Eps_Creep_Std", "Eval_Time_s",
        }
        assert set(df.columns) == expected_columns

        # All rows evaluated at the shortest completed test's duration (5.0s)
        np.testing.assert_allclose(df["Eval_Time_s"].to_numpy(), 5.0)

        row_short = df.loc[df["Test_ID"] == "short"].iloc[0]
        assert row_short["Eps_0"] == pytest.approx(0.001)
        assert row_short["Eps_Max_Std"] == pytest.approx(0.0015)  # exact match at t=5.0
        assert row_short["Eps_Creep_Std"] == pytest.approx(0.0005)

        row_long = df.loc[df["Test_ID"] == "long"].iloc[0]
        # long test's strain at t=5.0 (exact match) is 0.003
        assert row_long["Eps_Max_Std"] == pytest.approx(0.003)
        assert row_long["Eps_Creep_Std"] == pytest.approx(0.001)