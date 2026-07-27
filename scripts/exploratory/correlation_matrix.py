from pathlib import Path
import seaborn as sns
import matplotlib.pyplot as plt

from creep_model.io.parser import ExcelCreepParser
from creep_model.modelling.assembler import DataAssembler

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
    corr_matrix = numeric_df.corr(method='spearman') # You can also try 'spearman' for non-linear rank correlation
    
    print("\Spearman Correlation Matrix:")
    print(corr_matrix[['Eps_0', 'Eps_Creep_Std']])
    
    # 3. Visualizing Correlations (The "Pairplot")
    # Seaborn's pairplot is the ultimate tool for this. It plots every variable against 
    # every other variable, and colors them by a category (like Print_Quality).

       # Create a "Nominal_Stress" column for better visualization
    df_summary['Nominal_Stress'] = round(df_summary['Applied_Stress_MPa'], 0)

    # 1. Force the stress values into exactly 10, 20, or 30 using a custom function
    def map_to_nominal(val):
        if val < 15: return 10
        elif val < 25: return 20
        else: return 30

    df_summary['Nominal_Stress'] = df_summary['Applied_Stress_MPa'].apply(map_to_nominal)

    # 2. Define variables and map the explicit 10, 20, 30 values to shapes
    plot_vars = ["Applied_Stress_MPa", "Mean_Temp_C", "Eps_0", "Eps_Creep_Std"]
    marker_mapping = {10: "o", 20: "s", 30: "D"}

    # 3. Set up the Grid with BOTH hue and style specified here
    g = sns.PairGrid(
        df_summary, 
        vars=plot_vars, 
        hue="Print_Quality", 
        hue_kws={"marker": ["o", "s", "D"]}  # Pre-allocates legend slots safely
    )

    # 4. Map the non-diagonal scatter plots using style explicitly mapped to Nominal_Stress
    g.map_offdiag(
        sns.scatterplot, 
        style=df_summary["Nominal_Stress"], 
        markers=marker_mapping,
        alpha=0.7, 
        s=50
    )

    # 5. Map the diagonal plots (KDE)
    g.map_diag(sns.kdeplot, fill=True)

    # 6. Add legends and title
    g.add_legend()
    plt.suptitle("Cross-Sectional Creep Analysis", y=1.02)
    plt.show()


if __name__ == "__main__":
    main()