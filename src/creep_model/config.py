from dataclasses import dataclass, field
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
    K1: int = 3
    K2: int = 3

# DE hyperparameters for TLV fitting
    DE_KWARGS: dict = field(default_factory=lambda: {
        "seed": 42,
        "workers": 1,            # <-- Changed to 1: Disables multiprocessing entirely
        "popsize": 6,            # 6 * 10 = 60 population members
        "maxiter": 40,           # Cap generations (DE locates basin)
        "tol": 0.05,             # Relax tolerance (LM finishes convergence)
        "atol": 1e-3,
        "strategy": "best1bin",  # Fast convergence strategy
        "mutation": (0.5, 1.0),
        "recombination": 0.7,
        "updating": "immediate", # <-- Changed to immediate: Faster convergence on a single core
    })

    # LM hyperparameters for TLV fitting
    LM_KWARGS: dict = field(default_factory=lambda: {
        "method": "lm",
        "max_nfev": 5000,
        "ftol": 1e-8,
        "xtol": 1e-8,
        "gtol": 1e-8,
    })

    # Newton-Raphson hyperparameters for TLV fitting
    NR_KWARGS: dict = field(default_factory=lambda: {
        "tol": 1e-8,
        "max_iter": 100,
    })

    # Optimization Penalty
    NON_CONVERGENCE_PENALTY: float = 1e12


config = CreepConfig()