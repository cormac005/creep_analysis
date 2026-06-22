from pathlib import Path
import seaborn as sns
import matplotlib.pyplot as plt

from creep_model.io.parser import ExcelCreepParser
from creep_model.modeling.assembler import DataAssembler

def main():
    data_path = Path("data/raw/CreepData.xlsx") 
    parser = ExcelCreepParser(data_path)
    experiment = parser.load_experiment()
    
    # 1. Get the Tabular Data
    df_summary = DataAssembler.get_summary_dataframe(experiment)
    print("Summary Data:")
    print(df_summary.head())
    
    # 2. Calculate the Correlation Matrix
    # We drop non-numeric columns like Test_ID and Print_Quality for pure math correlation
    numeric_df = df_summary.select_dtypes(include=['float64', 'int64'])
    corr_matrix = numeric_df.corr(method='pearson') # You can also try 'spearman' for non-linear rank correlation
    
    print("\nPearson Correlation Matrix:")
    print(corr_matrix[['Eps_0', 'Eps_Creep_Std']])
    
    # 3. Visualizing Correlations (The "Pairplot")
    # Seaborn's pairplot is the ultimate tool for this. It plots every variable against 
    # every other variable, and colors them by a category (like Print_Quality).
    sns.pairplot(
        df_summary, 
        vars=["Applied_Stress_MPa", "Mean_Temp_C", "Eps_0", "Eps_Creep_Std"],
        hue="Print_Quality", 
        diag_kind="kde",
        plot_kws={'alpha': 0.7, 's': 100} # Make points transparent and larger
    )
    plt.suptitle("Cross-Sectional Creep Analysis", y=1.02)
    plt.show()

if __name__ == "__main__":
    main()