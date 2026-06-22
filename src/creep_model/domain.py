from dataclasses import dataclass
import numpy as np
import numpy.typing as npt

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
    
    def interpolate_temperature(self) -> npt.NDArray[np.float64]:
        """
        Replace all NaN values in the temperature_readings array with interpolated values based on the time_series.
        """
        # Create a boolean mask for valid (non-NaN) temperature readings
        valid_mask = ~np.isnan(self.temperature_readings)
        # Interpolate the valid temperature readings to match the time series
        interpolated_temps = np.interp(
            self.time_series,
            self.temp_time_series*60,  # Convert minutes to seconds
            self.temperature_readings
        )
        return interpolated_temps

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