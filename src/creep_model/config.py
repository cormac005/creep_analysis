from dataclasses import dataclass

@dataclass(frozen=True)
class CreepConfig:
    """Centralized configuration for the Creep Modeling package."""
    
    # File configuration
    metadata_sheet_name: str = "MetaData"

    # Constants for the creep test
    gauge_length_mm: float = 20.0  # Gauge length in millimeters
    
    # Expected data shapes (Crucial for validation later)
    expected_time_points: int = 480
    expected_temp_points: int = 24
    
    # Column mapping (Update these strings to exactly match your Excel headers)
    col_time: str = "Time_s"
    col_extension: str = "Extension_mm"
    col_temp_time: str = "TempTime_mins"
    col_temp: str = "Temperature_degC"
    
    # You can add polynomial degrees or other fixed constants here later

# Create a single global instance to be imported by other files
config = CreepConfig()