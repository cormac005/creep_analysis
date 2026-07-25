"""
Shared pytest fixtures for the creep_model test suite.

The central idea used throughout tests/eda/: build synthetic CreepTest
objects with EXACTLY known plateau structure, rather than testing against
real CreepData.xlsx data first. Real data gives you no independent ground
truth to check stage-classification boundaries against; synthetic data lets
you assert exact expected indices.
"""
from pathlib import Path

import numpy as np
import pytest

from creep_model.domain import CreepTest


@pytest.fixture
def make_test():
    """
    Factory fixture: builds a minimal CreepTest from an explicit strain
    series, filling in physically-plausible defaults for everything else.

    Usage:
        test = make_test(strain_series=[0, 0, 1, 1, 1, 2])
    """

    def _make(
        strain_series,
        time_series=None,
        temp_time_series=None,
        temperature_readings=None,
        **kwargs,
    ) -> CreepTest:
        strain_series = np.asarray(strain_series, dtype=np.float64)
        n = len(strain_series)

        if time_series is None:
            time_series = np.arange(n, dtype=np.float64)
        else:
            time_series = np.asarray(time_series, dtype=np.float64)

        if temp_time_series is None:
            temp_time_series = time_series.copy()
        else:
            temp_time_series = np.asarray(temp_time_series, dtype=np.float64)

        if temperature_readings is None:
            temperature_readings = np.full(len(temp_time_series), 22.0)
        else:
            temperature_readings = np.asarray(temperature_readings, dtype=np.float64)

        defaults = dict(
            test_id="synthetic",
            time_series=time_series,
            strain_series=strain_series,
            temp_time_series=temp_time_series,
            temperature_readings=temperature_readings,
            applied_stress_MPa=20.0,
            age_days=5,
            print_quality="High",
        )
        defaults.update(kwargs)
        return CreepTest(**defaults)

    return _make


@pytest.fixture
def strain_from_plateau_lengths():
    """
    Builds a strain_series with EXACTLY the given plateau lengths, using
    strictly increasing distinct values per plateau (0, 1, 2, ...) so
    `_find_plateaus` cannot merge adjacent plateaus by accident.

    Usage:
        strain = strain_from_plateau_lengths([1, 2, 3, 5, 5, 5, 5])
        # -> 7 plateaus with those exact n_points, values 0..6
    """

    def _build(lengths: list[int]) -> np.ndarray:
        return np.concatenate(
            [np.full(n, float(value)) for value, n in enumerate(lengths)]
        )

    return _build


@pytest.fixture
def real_data_path() -> Path:
    """
    Path to the real workbook, used only by integration tests. These tests
    should skip (not fail) if the file isn't present, since data/ is
    git-ignored (see .gitignore) and won't exist in a fresh clone or CI.
    """
    return Path("data/raw/CreepData.xlsx")