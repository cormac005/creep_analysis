from dataclasses import dataclass

@dataclass(frozen=True)
class CreepConfig:
    """Centralized configuration for the Creep Modeling package."""

    # File configuration
    data_directory: str = "data/raw"
    metadata_sheet_name: str = "MetaData"

    # Constants for the creep test
    gauge_length_mm: float = 20.0  # Gauge length in millimeters
    
    # Column mapping (Update these strings to exactly match your Excel headers)
    col_time: str = "Time_s"
    col_extension: str = "Extension_mm"
    col_temp_time: str = "TempTime_mins"
    col_temp: str = "Temperature_degC"

# Create a single global instance to be imported by other files
config = CreepConfig()