"""
This script loads the raw data, classifies the creep stages and saves the raw data along with the trimmed data to an HDF5 file. 
"""
from dataclasses import fields
from datetime import datetime
from pathlib import Path

import h5py
import numpy as np

from creep_model.config import config
from creep_model.io.parser import ExcelCreepParser
from creep_model.modelling.tlv.trimming import trim_and_partition

# --- Configuration ---
DATA_PATH = Path(config.data_directory) / "CreepData.xlsx"
OUTPUT_PATH = Path(config.data_output_directory) / "processed_experimental_data.h5"

# Hyperparameters for stage classification (from config.py)
K1 = config.K1
K2 = config.K2


def main() -> None:
    # Load the experimental data
    parser = ExcelCreepParser(DATA_PATH)
    experiment = parser.load_experiment()

    # Trim tertiary creep and partition by print quality in one step --
    # trim_and_partition already returns {"High": [...], "Standard": [...]}
    groups = trim_and_partition(experiment, k1=K1, k2=K2)

    # Ensure output directory exists before writing
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    total_records = 0

    # Package data and save to data/processed as an HDF5 file
    with h5py.File(OUTPUT_PATH, "w") as f:
        f.attrs["k1"] = K1
        f.attrs["k2"] = K2
        f.attrs["generated_at"] = datetime.now().isoformat()

        for group_name, records in groups.items():
            # Create a group for each partition (e.g., "High", "Standard")
            group_h5 = f.create_group(str(group_name))

            for idx, record in enumerate(records):
                total_records += 1

                # Use specimen name/id if available; fallback to formatted index
                record_id = getattr(
                    record, "name", getattr(record, "id", f"specimen_{idx + 1}")
                )
                rec_grp = group_h5.create_group(str(record_id))

                # Iterate through dataclass fields to dynamically store datasets & metadata
                for field_info in fields(record):
                    val = getattr(record, field_info.name)
                    if val is None:
                        continue

                    # Save arrays/sequences as HDF5 Datasets, scalar attributes as HDF5 Attributes
                    if isinstance(val, (np.ndarray, list, tuple)):
                        rec_grp.create_dataset(field_info.name, data=val)
                    else:
                        rec_grp.attrs[field_info.name] = val

    print(f"Saved fit results for {total_records} test(s) to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()