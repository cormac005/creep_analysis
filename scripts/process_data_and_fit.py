from creep_model.io.parser import ExcelCreepParser
from creep_model.modelling.assembler import DataAssembler
from creep_model.modelling.tlv import fit_pipeline

# Load the experimental data
parser = ExcelCreepParser("data/raw/experimental_data.xlsx")
data = parser.parse()

# Trim and partition


# Group the data by print quality


# Fit TLV models


# Get all data


# Package data


# Save to data/processed as a HDF5 file