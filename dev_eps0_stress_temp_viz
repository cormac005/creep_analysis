from pathlib import Path
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from scipy.optimize import curve_fit
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import shapiro
from plotly.subplots import make_subplots
from creep_model.io.parser import ExcelCreepParser


def eps0_model(X, E_20, E_30):
    """
    Theoretical model for initial strain (eps_0).
    X : tuple of (stress in Pa, Temperature in Kelvin)
    E_20, E_30 : Fit parameters for Young's moduli at 20°C and 30°C.
    """
    stress_Pa, T = X
    T_20 = 293.15
    T_30 = 303.15
    
    # Calculate denominator 
    denom = E_20 + (E_30 - E_20) / (T_30 - T_20) * (T - T_20)
    
    return stress_Pa * (0.5 / denom)


def main():
    # Load the Excel file and parse the experiment
    data_path = Path("data/raw/CreepData.xlsx")
    parser = ExcelCreepParser(data_path)
    experiment = parser.load_experiment()

    # 1. Pull out the necessary attributes for each test
    eps0_data = []
    for test in experiment.tests.values():
        if test.is_empty:
            continue
        
        eps0_data.append({
            "Test_ID": test.test_id,
            "Initial_Strain": test.strain_series[0],
            "Initial_Temperature": test.temperature_readings[0],
            "Applied_Stress_MPa": test.applied_stress_MPa,
            "Print_Quality": getattr(test, "print_quality", "Unknown"), 
        })  

    # Convert to DataFrame
    df = pd.DataFrame(eps0_data)
    
    # Generate metric units mapped to Pa and K for correct model calculations
    df["Applied_Stress_Pa"] = df["Applied_Stress_MPa"] * 1e6
    df["Initial_Temperature_K"] = df["Initial_Temperature"] + 273.15

    # 2. Generate the interactive 3D scatter plot
    fig = px.scatter_3d(
        df,
        x="Applied_Stress_MPa",
        y="Initial_Temperature",
        z="Initial_Strain",
        color="Print_Quality", 
        hover_name="Test_ID",
        hover_data={
            "Applied_Stress_MPa": ":.1f",
            "Initial_Temperature": ":.1f",
            "Initial_Strain": ":.5f",
            "Print_Quality": True,
        },
        color_continuous_scale="Turbo",
        labels={
            "Applied_Stress_MPa": "Applied Stress (MPa)",
            "Initial_Temperature": "Temperature (°C)",
            "Initial_Strain": "Initial Strain",
            "Print_Quality": "Print Quality",
        },
        title="Creep Test Curves — Stress / Temperature / Strain",
    )

    # 3. Marker styling — small, semi-transparent
    fig.update_traces(marker=dict(size=5, opacity=0.85, line=dict(width=0)))
    
    # 4. Perform LM curve fitting with an MSE target for each Print Quality 
    # Create evaluation grids for our model rendering
    stress_min, stress_max = df["Applied_Stress_MPa"].min(), df["Applied_Stress_MPa"].max()
    temp_min, temp_max = df["Initial_Temperature"].min(), df["Initial_Temperature"].max()
    
    stress_grid = np.linspace(stress_min, stress_max, 30)
    temp_grid = np.linspace(temp_min, temp_max, 30)
    Stress_Mesh, Temp_Mesh = np.meshgrid(stress_grid, temp_grid)
    
    # Convert mesh grids to Pa and K for math evaluations
    Stress_Pa_Mesh = Stress_Mesh * 1e6
    Temp_K_Mesh = Temp_Mesh + 273.15

    # Fit data separately for each Print Quality
    qualities = df["Print_Quality"].unique()
    surface_colors = ["Blues", "Reds", "Greens", "Purples"] # Used to distinctively map the surfaces
    
    for idx, quality in enumerate(qualities):
        df_sub = df[df["Print_Quality"] == quality]
        
        # We need at least 2 data points for a reliable 2-parameter fit
        if len(df_sub) < 2:
            print(f"Skipping {quality}: not enough data points.")
            continue
            
        X_data = (df_sub["Applied_Stress_Pa"].values, df_sub["Initial_Temperature_K"].values)
        y_data = df_sub["Initial_Strain"].values
        
        # Provide an initial guess for E_20 and E_30 (Assumed here as 1 GPa)
        p0 = [1e9, 1e9]
        
        try:
            # Curve fit utilizes Non-Linear Least Squares. Parameter method='lm' forces 
            # Levenberg-Marquardt. By default it minimizes the Sum of Squared Residuals (MSE target).
            popt, pcov = curve_fit(eps0_model, X_data, y_data, p0=p0, method='lm')
            E_20_fit, E_30_fit = popt
            print(f"Fitted parameters for '{quality}': E_20 = {E_20_fit:.2e} Pa, E_30 = {E_30_fit:.2e} Pa")
            
            # Use fit params to generate Z-values (Strain) for the surface plane
            Strain_Mesh = eps0_model((Stress_Pa_Mesh, Temp_K_Mesh), E_20_fit, E_30_fit)
            
            # Render the fitted function surface mapped over the 3D plane
            cscale = surface_colors[idx % len(surface_colors)]
            fig.add_trace(go.Surface(
                x=Stress_Mesh, 
                y=Temp_Mesh, 
                z=Strain_Mesh,
                name=f"{quality} Fit",
                colorscale=cscale,
                showscale=False,
                opacity=0.6,
                hovertemplate=
                    f"<b>{quality} Model Fit</b><br>" +
                    "Stress: %{x:.1f} MPa<br>" +
                    "Temp: %{y:.1f} °C<br>" +
                    "Fit Strain: %{z:.5f}<extra></extra>"
            ))

            # --- CALCULATE METRICS & RESIDUALS ---
            # 1. Calculate Predicted Strain
            y_pred = eps0_model(X_data, E_20_fit, E_30_fit)
            
            # 2. Calculate Residuals (Actual - Predicted)
            residuals = y_data - y_pred
            
            # 3. Save to dataframe for plotting
            df.loc[df["Print_Quality"] == quality, "Predicted_Strain"] = y_pred
            df.loc[df["Print_Quality"] == quality, "Residuals"] = residuals
            
            # 4. Statistical Metrics
            mse = mean_squared_error(y_data, y_pred)
            rmse = np.sqrt(mse)
            r2 = r2_score(y_data, y_pred)
            
            # Check for normality in residuals (p > 0.05 implies normal distribution)
            stat, p_value = shapiro(residuals)
            
            print(f"--- Stats for '{quality}' ---")
            print(f"R-squared: {r2:.4f}")
            print(f"RMSE:      {rmse:.5f}")
            print(f"Shapiro-Wilk p-value (Residuals): {p_value:.4f}")
            if p_value > 0.05:
                print(" -> Residuals look normally distributed (Random noise, good!)")
            else:
                print(" -> Residuals deviate from normality (Check for non-linear trends at high stress).")
            print("-" * 30)
            
        except Exception as e:
            print(f"Curve fitting failed for {quality}: {e}")
            

    # 5. Layout configuration
    fig.update_layout(
        template="plotly_white",
        title=dict(x=0.5, xanchor="center", font=dict(size=20)),
        font=dict(family="Arial, sans-serif", size=12),
        margin=dict(l=0, r=0, t=60, b=0),
        scene=dict(
            xaxis=dict(title="Applied Stress (MPa)", backgroundcolor="white",
                       gridcolor="lightgrey", showbackground=True),
            yaxis=dict(title="Temperature (°C)", backgroundcolor="white", 
                       gridcolor="lightgrey", showbackground=True),
            zaxis=dict(title="Initial Strain", backgroundcolor="white", 
                       gridcolor="lightgrey", showbackground=True),
            aspectmode="cube",
            camera=dict(eye=dict(x=1.6, y=1.6, z=1.1)),
        ),
        legend=dict(
            title=dict(text="Print Quality"),
            x=1.02, y=0.5, xanchor="left", yanchor="middle",
            bgcolor="rgba(255,255,255,0.85)",
            bordercolor="lightgrey", borderwidth=1,
        ),
        width=1000,
        height=750,
    )

    fig.show()

    # Create a 1x2 subplot: Residuals vs Stress & Predicted vs Actual
    fig_diag = make_subplots(
        rows=1, cols=2, 
        subplot_titles=("Residuals vs. Applied Stress", "Predicted vs. Actual Strain")
    )

    colors = {"High": "blue", "Standard": "red"} # Adjust to match your themes

    for quality in qualities:
        df_sub = df[df["Print_Quality"] == quality]
        if "Residuals" not in df_sub.columns: continue
        
        # Plot 1: Residuals vs Stress (Looking for randomness vs. U-shape)
        fig_diag.add_trace(
            go.Scatter(
                x=df_sub["Applied_Stress_MPa"], 
                y=df_sub["Residuals"],
                mode='markers',
                name=f"{quality} (Residuals)",
                marker=dict(color=colors.get(quality, "black"), size=6)
            ),
            row=1, col=1
        )
        
        # Plot 2: Predicted vs Actual Strain (Looking for a 1:1 diagonal)
        fig_diag.add_trace(
            go.Scatter(
                x=df_sub["Predicted_Strain"], 
                y=df_sub["Initial_Strain"],
                mode='markers',
                name=f"{quality} (Actual)",
                marker=dict(color=colors.get(quality, "black"), size=6),
                showlegend=False
            ),
            row=1, col=2
        )

    # Add a horizontal line at 0 for Residuals
    fig_diag.add_hline(y=0, line_dash="dash", line_color="black", row=1, col=1)
    
    # Add a 1:1 diagonal line for Predicted vs Actual
    min_strain = df["Initial_Strain"].min()
    max_strain = df["Initial_Strain"].max()
    fig_diag.add_trace(
        go.Scatter(
            x=[min_strain, max_strain], y=[min_strain, max_strain],
            mode="lines", line=dict(dash="dash", color="black"), name="1:1 Line"
        ), row=1, col=2
    )

    fig_diag.update_layout(
        title_text="Model Diagnostics: Checking for Yielding",
        template="plotly_white",
        height=500
    )
    
    # Update axes titles
    fig_diag.update_xaxes(title_text="Applied Stress (MPa)", row=1, col=1)
    fig_diag.update_yaxes(title_text="Residual (Actual - Predicted)", row=1, col=1)
    
    fig_diag.update_xaxes(title_text="Predicted Strain", row=1, col=2)
    fig_diag.update_yaxes(title_text="Actual Initial Strain", row=1, col=2)

    # Show the diagnostic figure
    fig_diag.show()


if __name__ == "__main__":
    main()