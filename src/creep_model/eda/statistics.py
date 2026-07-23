"""
Creep summary statistics used for exploratory data visualisation

Two statistics are defined:
    eps_tilde_0 : the first valid strain recording in a test.
                  -> This already exists as CreepTest.eps_0; re-exported
                     here just so all "the two thesis statistics" live in
                     one obvious place for whoever reads this module.
    eps_dot_ss  : the estimated steady-state (secondary creep) strain rate,
                  i.e. the slope of a model fit to the strain values
                  classified as secondary creep.

Fits slope by via linear regression.
"""
from turtle import pd

import numpy as np
import numpy.typing as npt

from creep_model.domain import CreepTest
from creep_model.eda.stage_classification import StageClassification


def first_valid_strain(test: CreepTest) -> float:
    eps_tilde_0 = test.eps_0
    return eps_tilde_0


def steady_state_strain_rate(
    test: CreepTest,
    classification: StageClassification,
) -> float:
    """
    eps_dot_ss -- slope of strain vs. time over the secondary-creep region.

    Args:
        test: the CreepTest the classification was computed from.
        classification: result of eda.stage_classification.classify_stages
            for THIS test (caller is responsible for making sure these
            match -- consider adding an assert on test.test_id ==
            classification.test_id if you want a cheap sanity check).

    Returns:
        Estimated slope (strain / second). NaN if no secondary creep
        region was identified (has_tertiary handling aside, a test that
        ends mid-secondary-creep still has a valid, if shorter, region).
        Mean temperature during the secondary creep region

    """
    # Slice the time and strain series for the secondary creep region
    time_series = test.time_series
    strain_series = test.strain_series
    start = classification.primary_end_idx
    end = classification.secondary_end_idx  # may be None -> use test end
    if start is None:
        eps_dot_ss = np.nan
        mean_temp = np.nan
        return eps_dot_ss, mean_temp
    
    if end is None:
       # Exlude the final plateau if secondary_end_idx is None
       end = len(time_series) - 1

    # Prepare the data for linear regression
    x = time_series[start:end + 1]  
    y = strain_series[start:end + 1]

    # Fit linear regression to find the slope (eps_dot_ss)
    if len(x) < 2:
        # Not enough points to fit a line
        eps_dot_ss = np.nan
        mean_temp = np.nan
        return eps_dot_ss, mean_temp

    A = np.vstack([x, np.ones(len(x))]).T
    slope, _ = np.linalg.lstsq(A, y, rcond=None)[0]
    eps_dot_ss = slope

    # Find mean temperature during the secondary creep region
    mean_temp = np.mean(test.temperature_readings[start:end + 1])
    
    return eps_dot_ss, mean_temp


def build_eda_dataframe(
    tests: dict[str, CreepTest],
    classifications: dict[str, StageClassification],
):
    """
    Convenience aggregator: builds a per-test row of
    [Test_ID, Applied_Stress_MPa, Age_Days, Print_Quality, Mean_Temp_C,
     Eps_Tilde_0, Eps_Dot_Ss] for use in the trellis/beeswarm/bubble plots
    already planned for the CSA section.

    TODO(you): implement the loop, calling first_valid_strain and
    steady_state_strain_rate per test, and pd.DataFrame(records) at the end.
    """
    # Loop over tests and classifications to build a list of records
    records = []
    for test_id, test in tests.items():
        # Get stage classification for the current test
        classification = classifications.get(test_id)
        if classification is None:
            raise ValueError(f"No classification found for test ID: {test_id}")

        # Compute the two statistics for the current test
        eps_tilde_0 = first_valid_strain(test)
        eps_dot_ss, mean_temp = steady_state_strain_rate(test, classification)

        record = {
            "Test_ID": test.test_id,
            "Applied_Stress_MPa": test.applied_stress_MPa,
            "Age_Days": test.age_days,
            "Print_Quality": test.print_quality,
            "Mean_Temp_C_Secondary_Creep": mean_temp,
            "Eps_Tilde_0": eps_tilde_0,
            "Eps_Dot_Ss": eps_dot_ss,
        }
        records.append(record)
    return pd.DataFrame(records)