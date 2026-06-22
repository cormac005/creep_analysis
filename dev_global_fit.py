from pathlib import Path
from creep_model.io.parser import ExcelCreepParser
from creep_model.modeling.assembler import DataAssembler
from creep_model.modeling.empirical import GlobalMSEModel

def main():
    parser = ExcelCreepParser(Path("data/raw/CreepData.xlsx"))
    experiment = parser.load_experiment()
    
    for quality in ["High", "Standard"]:
        print(f"\n--- Fitting Global Model for {quality} Prints ---")
        try:
            X, y = DataAssembler.get_global_data(experiment, quality)
            print(f"Data Assembled: {len(y)} total data points across all tests.")
            
            model = GlobalMSEModel()
            model.fit(X, y)
            
            params = model.fitted_params_
            print(f"Fit Successful!")
            print(f" eps_coeff : {params[0]:.4e}")
            print(f" B         : {params[1]:.4e}")
            print(f" n (stress): {params[2]:.4f}")
            print(f" m (time)  : {params[3]:.4f}")
        except Exception as e:
            print(f"Could not fit {quality}: {e}")

if __name__ == "__main__":
    main()