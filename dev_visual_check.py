# dev_visual_check.py
from pathlib import Path

from creep_model.io.parser import ExcelCreepParser
from creep_model.modeling.assembler import DataAssembler
from creep_model.modeling.empirical import LocalFindleyModel, QuantizedFindleyModel
from creep_model.viz.plots import plot_local_fit

def main():
    # 1. Configuration (Update with your actual file path)
    data_path = Path("data/raw/CreepData.xlsx") 
    
    print("1. Parsing Excel file...")
    parser = ExcelCreepParser(data_path)
    experiment = parser.load_experiment()
    
    # Grab the first available test sheet dynamically
    test_names = list(experiment.tests.keys())
    first_test_name = test_names[1]
    test = experiment.tests[first_test_name]

    print(f"2. Assembling data for sheet: {first_test_name}")
    X, y = DataAssembler.get_local_data(test)
    
    print(f"   - Shape of X (Time): {X.shape}")
    print(f"   - Shape of y (Strain): {y.shape}")
    # Quick sanity check: Does X start at roughly > 0 since we dropped the pre-load?
    print(f"   - First time reading: {X[0][0]:.4f}") 
    
    print("3. Initializing and Fitting Findley Model...")
    model = QuantizedFindleyModel()
    
    try:
        model.fit(X, y)
        params = model.fitted_params_
        print(f"   - Fit Successful! Optimal Parameters: eps0={params[0]:.4e}, m={params[1]:.4e}, n={params[2]:.4f}")
    except Exception as e:
        print(f"   - Model fitting failed: {e}")
        return

    print("4. Generating Visualization...")
    # This will pop up a window on your screen. 
    # Look closely at the residuals subplot!
    plot_local_fit(test, model)
    
    print("Pipeline check complete.")

if __name__ == "__main__":
    main()