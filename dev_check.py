# dev_check.py
from pathlib import Path
from creep_model.io.parser import ExcelCreepParser

def main():
    # Update this path to match your actual file name
    data_path = Path("C:\\Y4-Summer\\FYP\\Code\\creep_model\\data\\raw\\CreepData.xlsx") 
    
    print("Initializing Parser...")
    parser = ExcelCreepParser(data_path)
    
    print("Loading Experiment...")
    experiment = parser.load_experiment()
    
    print(f"Successfully loaded! Applied Stress: {experiment.applied_stress_MPa} MPa, Age: {experiment.age_days} days, Run Number: {experiment.run_number}, Print Quality: {experiment.print_quality}")
    print(f"Number of tests loaded: {len(experiment.tests)}")
    
    # Grab the first test to verify shapes
    first_test_name = list(experiment.tests.keys())[0]
    first_test = experiment.tests[first_test_name]
    
    print(f"\nVerifying '{first_test_name}':")
    print(f" - Strain points: {len(first_test.time_series)}")
    print(f" - Temp points: {len(first_test.temperature_readings)}")
    
    # Test the interpolation
    interp_temps = first_test.interpolate_temperature()
    print(f" - Interpolated Temp shape: {interp_temps.shape} (Should match Strain points)")

if __name__ == "__main__":
    main()