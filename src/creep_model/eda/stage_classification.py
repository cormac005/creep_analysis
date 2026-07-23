"""
Creep stage classification (Thesis Section 1.1.1).

Classifies each CreepTest into primary / secondary / (optionally excluded)
tertiary creep regions, based on the number of quantized data points spent
on each consecutive strain "plateau".

Rule recap:
    - Group strain_series into runs of identical (quantized) value -> plateaus.
    - plateau_lengths[i] = number of samples spent at that strain level.
    - Primary creep:   plateau_lengths is increasing.
    - Secondary creep: begins at the first of k1 CONSECUTIVE plateaus where
                        plateau_lengths is non-increasing.
                        -> primary creep ends at the plateau BEFORE this run.
    - Tertiary creep:  begins after k2 CONSECUTIVE plateaus of STRICTLY
                        decreasing plateau_lengths.
    - The final plateau is always excluded from classification (it ends
      because the test ends, not because of a genuine transition).
"""
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from creep_model.domain import CreepTest


@dataclass(frozen=True)
class Plateau:
    """A single consecutive run of (quantized) equal strain values."""
    strain_value: float
    start_idx: int          # index into the ORIGINAL time/strain series
    end_idx: int             # inclusive
    n_points: int


@dataclass(frozen=True)
class StageClassification:
    """
    Result of classifying a single CreepTest.

    Indices refer to the ORIGINAL (untrimmed) time_series / strain_series
    of the CreepTest that was classified.
    """
    test_id: str
    plateaus: list[Plateau]
    primary_end_idx: int      # last index (in time_series) still primary creep
    secondary_end_idx: int | None   # last index still secondary creep; None if
                                     # tertiary never detected (test may have ended
                                     # in secondary creep)
    k1: int
    k2: int

    @property
    def has_tertiary(self) -> bool:
        return self.secondary_end_idx is not None


def _find_plateaus(strain_series: npt.NDArray[np.float64]) -> list[Plateau]:
    """
    Identify consecutive runs of identical strain values in a CreepTest's strain_series.
    Args:
        strain_series: the strain_series of a single CreepTest.

    Returns:
        A list of Plateau objects representing the consecutive runs of identical strain values.
    """
    # Find the indices where the strain value changes (i.e., the boundaries of plateaus)
    change_points = np.where(np.diff(strain_series) != 0)[0] + 1

    # Include the start and end indices to define the boundaries of plateaus
    boundaries = np.concatenate(([0], change_points, [len(strain_series)]))

    # Build a list of Plateau objects based on the boundaries
    plateaus = []
    for i in range(len(boundaries) - 1):
        start_idx = boundaries[i]
        end_idx = boundaries[i + 1] - 1  # inclusive end index
        strain_value = strain_series[start_idx]
        n_points = end_idx - start_idx + 1
        plateaus.append(Plateau(strain_value, start_idx, end_idx, n_points))
    return plateaus


def classify_stages(test: CreepTest, k1: int, k2: int) -> StageClassification:
    """
    Classify primary / secondary / tertiary creep for a single test.

    Args:
        test: the CreepTest to classify.
        k1: number of consecutive non-increasing plateau lengths required
            to declare the START of secondary creep.
        k2: number of consecutive STRICTLY decreasing plateau lengths
            required to declare the START of tertiary creep.

    Returns:
        StageClassification with boundary indices you can use directly for
        trimming (see modeling/trimming.py) and for computing eps_dot_ss
        (see eda/statistics.py).
    """
    if test.is_empty:
        raise ValueError(f"Cannot classify empty test: {test.test_id}")

    plateaus = _find_plateaus(test.strain_series)

    # Exclude the final plateau -- it ends because the test ends, not because of a real transition.
    plateaus_for_classification = plateaus[:-1]

    # Loop through every combination of k1 consectuve plateaus and check if they are non-increasing
    secondary_start_plateau_idx = None
    for i in range(len(plateaus_for_classification) - k1 + 1):
        if all(plateaus_for_classification[j].n_points >= plateaus_for_classification[j + 1].n_points for j in range(i, i + k1 - 1)):
            secondary_start_plateau_idx = i
            break

    if secondary_start_plateau_idx is None:
        # No secondary creep detected at all within this k1 setting --
        # Set all indices to None and return early.
        return StageClassification(
            test_id=test.test_id,
            plateaus=plateaus,
            primary_end_idx=None,
            secondary_end_idx=None,
            k1=k1,
            k2=k2,
        )
    else:  
        # Set Primary creep end index to the plateau before the first of the k1 run
        primary_end_idx = plateaus_for_classification[secondary_start_plateau_idx - 1].end_idx
        # Now search for tertiary creep after the secondary_start_plateau_idx
        tertiary_start_plateau_idx = None
        for i in range(secondary_start_plateau_idx + k1, len(plateaus_for_classification) - k2 + 1):
            if all(plateaus_for_classification[j].n_points > plateaus_for_classification[j + 1].n_points for j in range(i, i + k2 - 1)):
                tertiary_start_plateau_idx = i
                break

    if tertiary_start_plateau_idx is None:
        # No Tertiary creep detected -- set secondary_end_idx to the last plateau's end index
        secondary_end_idx = plateaus_for_classification[-1].end_idx
    else:
        # Set secondary_end_idx to the plateau before the first of the k2 run
        secondary_end_idx = plateaus_for_classification[tertiary_start_plateau_idx - 1].end_idx

    return StageClassification(
        test_id=test.test_id,
        plateaus=plateaus,
        primary_end_idx=primary_end_idx,
        secondary_end_idx=secondary_end_idx,
        k1=k1,
        k2=k2,
    )