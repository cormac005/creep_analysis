# Imprort all the tlv modelling functions
from creep_model.modelling.tlv.fit_pipeline import fit_group, 
from pathlib import Path
from creep_model.io.parser import ExcelCreepParser
from creep_model.modelling.assembler import DataAssembler
import numpy as n
import pandas as pd
from creep_model.config import CreepConfig
from creep_model.eda import stage_classification

# Load Creep Data
data_path = CreepConfig.data_directory
creep_data_file = Path(data_path) / "CreepData.xlsx"
Creep_Data = ExcelCreepParser(creep_data_file)

# Classify Creep Stages


# Cut out tertiary creep data


# Split data by print quality


# Fit TLV model to both seperatly


# Generate Fit Summary Statistics


# Visualise fit 