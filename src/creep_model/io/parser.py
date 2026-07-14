from pathlib import Path
import pandas as pd
import numpy as np

from creep_model.config import config
from creep_model.domain import CreepTest, CreepExperiment

class ExcelCreepParser:
    """
    Parses the raw multi-sheet Excel workbook and constructs the pure Domain Model.
    """
    def __init__(self, filepath: Path | str) -> None:
        self.filepath = Path(filepath)
        if not self.filepath.exists():
            raise FileNotFoundError(f"Cannot find data file at: {self.filepath}")

    def _parse_metadata(self, df_meta: pd.DataFrame) -> dict:
        """
        Extracts metadata for each test from the already-loaded DataFrame.
        """
        # Drop any empty rows that Excel sometimes hides at the bottom
        df_meta = df_meta.dropna(subset=['RunCode'])
        
        # Convert the DataFrame to a dictionary for easy access
        metadata_dict = df_meta.set_index('RunCode').to_dict(orient='index')
        return metadata_dict

    def _parse_test_sheet(self, sheet_name: str, df_sheet: pd.DataFrame, sheet_meta: dict) -> CreepTest:
        """
        Processes a single test sheet, handling variable-length columns and NaNs.
        """
        # 1. Process the Extension Data
        df_ext = df_sheet[[config.col_time, config.col_extension]].dropna()

        # Exclude the first extension reading (Pre-load)
        df_ext = df_ext.iloc[1:]
        
        # 2. Isolate the Temperature Data
        df_temp = df_sheet[[config.col_temp_time, config.col_temp]].dropna()
        
        # 3. Convert to NumPy arrays
        time_series = df_ext[config.col_time].to_numpy(dtype=np.float64)
        extension_series = df_ext[config.col_extension].to_numpy(dtype=np.float64)
        temp_time_series = df_temp[config.col_temp_time].to_numpy(dtype=np.float64)
        temperature_readings = df_temp[config.col_temp].to_numpy(dtype=np.float64)

        # 4. Convert extension to strain using the gauge length
        strain_series = extension_series / config.gauge_length_mm

        # 5. Create and return the CreepTest object WITH METADATA
        return CreepTest(
            test_id=sheet_name,
            time_series=time_series,
            strain_series=strain_series,
            temp_time_series=temp_time_series,
            temperature_readings=temperature_readings,
            # Injecting the local metadata variables here:
            applied_stress_MPa=float(sheet_meta["StressPa"]) / 1e6,  # Converted from Pa to MPa
            age_days=int(sheet_meta["AgeDays"]),
            print_quality=str(sheet_meta["PrintQuality"])
        )

    def load_experiment(self) -> CreepExperiment:
        """
        Orchestrates the entire ingestion process.
        """
        # 1. Load all sheets into memory at once
        all_sheets = pd.read_excel(self.filepath, sheet_name=None)

        if "Home" in all_sheets:
            del all_sheets["Home"]

        # 2. Extract Metadata dictionary using the specific sheet
        df_meta = all_sheets[config.metadata_sheet_name]
        metadata_dict = self._parse_metadata(df_meta)
        
        # 3. Loop over the metadata and extract tests
        tests = {}
        for run_code, sheet_meta in metadata_dict.items():
            if run_code not in all_sheets:
                # Switched to a print warning instead of a fatal ValueError. 
                # This allows you to add rows to your MetaData table for upcoming 
                # tests without crashing the code before the sheet actually exists!
                print(f"Warning: Expected sheet '{run_code}' not found. Skipping.")
                continue
                
            df_sheet = all_sheets[run_code]
            
            # Pass BOTH the dataframe and the specific metadata dictionary
            tests[run_code] = self._parse_test_sheet(run_code, df_sheet, sheet_meta)
        
        # 4. Return the assembled CreepExperiment object
        return CreepExperiment(tests=tests)