from dataclasses import dataclass
import numpy as np
import numpy.typing as npt

@dataclass(frozen=True)
class CreepTest:
    """
    Represents a single physical creep test sheet.
    frozen=True makes this object immutable. Once created, data cannot be accidentally altered.
    """
    test_id: str
    time_series: npt.NDArray[np.float64]      
    strain_series: npt.NDArray[np.float64]   
    temperature_readings: npt.NDArray[np.float64] 
    
    def interpolate_temperature(self) -> npt.NDArray[np.float64]:
        """
        Replace all NaN values in the temperature_readings array with interpolated values based on the time_series.
        """
        # Create a boolean mask for valid (non-NaN) temperature readings
        valid_mask = ~np.isnan(self.temperature_readings)
        # Interpolate the valid temperature readings to match the time series
        interpolated_temps = np.interp(
            self.time_series,
            self.time_series[valid_mask],
            self.temperature_readings[valid_mask]
        )
        return interpolated_temps

@dataclass(frozen=True)
class CreepExperiment:
    """
    Represents the entire workbook: Metadata constants + 24 Tests.
    """
    applied_stress: float
    material_constant: float  # Add or rename other metadata constants here
    tests: dict[str, CreepTest] # Keys are sheet names, values are CreepTest objects