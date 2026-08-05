from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class CreepConfig:
    """Centralized configuration for the Creep Modeling package."""

    # File configuration
    data_directory: Path = Path("data/raw")
    metadata_sheet_name: str = "MetaData"
    data_output_directory: Path = Path("data/processed")
    general_output_directory: Path = Path("outputs")

    # Constants for the creep test
    gauge_length_mm: float = 20.0  
    
    # Column mapping 
    col_time: str = "Time_s"
    col_extension: str = "Extension_mm"
    col_temp_time: str = "TempTime_mins"
    col_temp: str = "Temperature_degC"

    # Creep Stage Classification thresholds
    K1 = 3
    K2 = 4

    # DE hyperparameters for TLV fitting 
        # Real run: = {"seed": 42, "workers": -1}
        # Small batch test: = {"maxiter": 20, "popsize": 10, "seed": 42, "workers": -1}
    DE_KWARGS = {"seed": 42, "workers": -1} 


config = CreepConfig()