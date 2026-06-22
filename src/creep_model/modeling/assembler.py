import numpy as np
import numpy.typing as npt
from creep_model.domain import CreepExperiment, CreepTest
import pandas as pd

class DataAssembler:
    """Extracts and formats data from domain objects into modeling matrices."""
    
    @staticmethod
    def get_local_data(test: CreepTest) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """
        Returns X (Time) and y (Strain) for a SINGLE test.
        X shape: (N, 1)
        y shape: (N,)
        """
        X = test.time_series.reshape(-1, 1)
        y = test.strain_series
        return X, y

    @staticmethod
    def get_global_data(experiment: CreepExperiment, target_quality: str) -> tuple[npt.NDArray, npt.NDArray]:
        """
        Extracts a flattened dataset for global modeling, filtered by print quality.
        
        Returns:
            X: 2D array of shape (Total_N, 2) -> Columns: [Time, Applied_Stress_MPa]
            y: 1D array of shape (Total_N,)   -> Strain
        """
        X_list = []
        y_list = []
        
        for test in experiment.tests.values():
            # Skip empty tests or tests that don't match our target quality
            if test.is_empty or test.print_quality != target_quality:
                continue
                
            time = test.time_series
            
            # Create a stress array of the exact same length as the time array
            stress = np.full_like(time, test.applied_stress_MPa)
            
            # Column stack into shape (N, 2)
            X_test = np.column_stack((time, stress))
            
            X_list.append(X_test)
            y_list.append(test.strain_series)
            
        if not X_list:
            raise ValueError(f"No data found for Print Quality: {target_quality}")
            
        # np.vstack stacks the list of 2D matrices vertically into one giant matrix
        return np.vstack(X_list), np.concatenate(y_list)
            
        # Use np.vstack(X_list) and np.concatenate(y_list) to return the final matrices
        X = np.column_stack((
            np.concatenate([test.time_series for test in experiment.tests.values()]),
            np.full(sum(len(test.time_series) for test in experiment.tests.values()), experiment.applied_stress_MPa),
            np.full(sum(len(test.time_series) for test in experiment.tests.values()), experiment.age_days),
            np.full(sum(len(test.time_series) for test in experiment.tests.values()), pq_binary),
            np.concatenate([test.interpolate_temperature() for test in experiment.tests.values()])
        ))
        y = np.concatenate([test.strain_series for test in experiment.tests.values()])
        return X, y
    
    @staticmethod
    def get_summary_dataframe(experiment: CreepExperiment) -> pd.DataFrame:
        """
        Extracts summary statistics for completed tests, standardized to the 
        duration of the shortest test for mathematically valid comparisons.
        """
        records = []
        
        # 1. Determine the global "cutoff" time for this specific experiment
        target_time = experiment.shortest_test_duration
        
        for test_name, test in experiment.tests.items():
            # 2. Silently skip tests that haven't been run yet
            if test.is_empty:
                continue
                
            eps_0 = test.eps_0
            
            # 3. Standardize the max strain at the exact target time
            eps_max_std = test.get_strain_at_time(target_time)
            eps_creep_std = eps_max_std - eps_0
            
            records.append({
                "Test_ID": test.test_id,
                "Applied_Stress_MPa": test.applied_stress_MPa,  
                "Age_Days": test.age_days,                      
                "Print_Quality": test.print_quality,            
                "Mean_Temp_C": test.mean_temperature,
                "Eps_0": eps_0,
                "Eps_Max_Std": eps_max_std,
                "Eps_Creep_Std": eps_creep_std,
                "Eval_Time_s": target_time 
            })
            
        return pd.DataFrame(records)