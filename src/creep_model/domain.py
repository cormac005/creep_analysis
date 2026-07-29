from dataclasses import dataclass
import numpy as np
import numpy.typing as npt
from scipy.interpolate import make_smoothing_spline

@dataclass(frozen=True)
class CreepTest:
    """
    Represents a single physical creep test sheet.
    """
    # Time Series Data
    test_id: str
    time_series: npt.NDArray[np.float64]
    strain_series: npt.NDArray[np.float64]
    temp_time_series: npt.NDArray[np.float64]
    temperature_readings: npt.NDArray[np.float64]
    
    # Metadata
    applied_stress_MPa: float
    age_days: int
    print_quality: str

    def temperature_polynomial(self, degree: int = 2) -> np.poly1d:
        """
        Fits a low-order polynomial (default: quadratic) to the raw
        temp_time_series/temperature_readings via ordinary least squares.

        Chosen after a systematic comparison across method families (linear,
        PCHIP, smoothing spline, isotonic regression + PCHIP, polynomial) and
        a smoothing-strength sweep within the smoothing-spline family (see
        scripts/diagnostics/temperature_method_comparison.py). A quadratic is
        deliberately NOT constrained to be monotonic -- temperature was not
        monotonically increasing or decreasing for all tests, only some, so
        a strictly monotonic method (e.g. isotonic regression) would be too
        restrictive; a quadratic's single curvature change gives enough
        flexibility to represent a genuine warm-then-cool (or vice versa)
        trend while still being far smoother than the raw 0.1C-quantized
        readings.

        NaN entries in temperature_readings are filtered out before fitting
        (this fixes previously-dead code: valid_mask was computed in the old
        implementation but never applied) -- a single NaN would otherwise
        corrupt the ENTIRE least-squares fit, not just nearby points the way
        the old np.interp-based approach's local NaN-poisoning did.

        degree is gracefully reduced if fewer than degree+1 valid readings
        are available (polyfit is underdetermined otherwise) -- relevant for
        short/synthetic test fixtures, not real CreepData.xlsx tests (~24
        raw readings each).
        """
        valid_mask = ~np.isnan(self.temperature_readings)
        t_valid = self.temp_time_series[valid_mask]
        y_valid = self.temperature_readings[valid_mask]

        effective_degree = max(min(degree, len(t_valid) - 1), 0)
        coeffs = np.polyfit(t_valid, y_valid, deg=effective_degree)
        return np.poly1d(coeffs)


    def interpolate_temperature(self) -> npt.NDArray[np.float64]:
        """
        Estimates temperature at every time_series point.

        NOTE: now backed by a quadratic polynomial fit (see
        temperature_polynomial()) rather than an interpolant forced through
        every quantized raw reading. Times outside the recorded temperature
        window [temp_time_series.min(), temp_time_series.max()] are CLAMPED
        to the polynomial's value at the nearest boundary rather than
        extrapolated -- a few tests have temperature logging end before
        strain logging does (see io/parser.py), and an unclamped quadratic
        could otherwise diverge sharply over that trailing region.
        """
        poly = self.temperature_polynomial()
        t_clamped = np.clip(
            self.time_series,
            self.temp_time_series.min(),
            self.temp_time_series.max(),
        )
        return poly(t_clamped)

    @property
    def is_empty(self) -> bool:
        """Checks if the test has data (useful for ongoing experiments)."""
        return len(self.time_series) == 0

    @property
    def final_time(self) -> float:
        """Returns the last recorded time, or 0.0 if empty."""
        return float(self.time_series[-1]) if not self.is_empty else 0.0

    @property
    def eps_0(self) -> float:
        """The initial instantaneous strain."""
        return float(self.strain_series[0]) if not self.is_empty else np.nan

    @property
    def mean_temperature(self) -> float:
        """The average temperature experienced during the test."""
        return float(np.mean(self.temperature_readings)) if not self.is_empty else np.nan

    def get_strain_at_time(self, target_time: float) -> float:
        """
        Calculates the strain at a specific time using linear interpolation.
        This handles the fact that timestamps might not align perfectly across tests.
        """
        if self.is_empty:
            return np.nan
            
        # np.interp(x, xp, fp) -> evaluates target_time using time_series and strain_series
        return float(np.interp(target_time, self.time_series, self.strain_series))

@dataclass(frozen=True)
class CreepExperiment:
    """
    Represents the entire workbook: Metadata constants + 24 Tests.
    """
    tests: dict[str, CreepTest] # Keys are sheet names, values are CreepTest objects

    @property
    def shortest_test_duration(self) -> float:
        """Finds the duration of the shortest COMPLETED test in the workbook."""
        valid_durations = [
            test.final_time for test in self.tests.values() if not test.is_empty
        ]
        
        if not valid_durations:
            raise ValueError("No valid completed tests found in the experiment.")
            
        return min(valid_durations)