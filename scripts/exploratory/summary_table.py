from pathlib import Path
from creep_model.io.parser import ExcelCreepParser
from creep_model.modelling.assembler import DataAssembler

def main():
    # Load the Excel file and parse the experiment
    data_path = Path("data/raw/CreepData.xlsx") 
    parser = ExcelCreepParser(data_path)
    experiment = parser.load_experiment()
    
    # Create a summary DataFrame for all tests
    df_summary = DataAssembler.get_summary_dataframe(experiment)

    # Display the DataFrame
    print("Summary Data:")
    print(df_summary)

if __name__ == "__main__":
    main()
    
