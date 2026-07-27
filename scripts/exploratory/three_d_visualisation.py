from pathlib import Path

from creep_model.io.parser import ExcelCreepParser
from creep_model.modelling.assembler import DataAssembler
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


def main():
    # Load the Excel file and parse the experiment
    data_path = Path("data/raw/CreepData.xlsx")
    parser = ExcelCreepParser(data_path)
    experiment = parser.load_experiment()

    # 1. Flatten the multi-test data structure into a single tabular format
    rows = []
    for test in experiment.tests.values():
        if test.is_empty:
            continue

        # BUG FIX: test.temperature_readings is sampled on temp_time_series
        # (a coarser, independent time base — 24 points per config.py) while
        # time_series/strain_series have ~480 points. zip()-ing the three
        # arrays directly silently truncates to the shortest one AND pairs
        # temperature sample i with time/strain sample i, which is not the
        # temperature that was actually recorded near that time.
        #
        # test.interpolate_temperature() already exists in domain.py to solve
        # this: it resamples temperature onto the time_series time base via
        # np.interp, so every strain point gets an actual (interpolated)
        # temperature for that same instant.
        interp_temp = test.interpolate_temperature()

        for t, s, T in zip(test.time_series, test.strain_series, interp_temp):
            rows.append({
                "Test_ID": test.test_id,
                "Applied_Stress_MPa": test.applied_stress_MPa,
                "Time_s": t,
                "Strain": s,
                "Print_Quality": test.print_quality,
                "Temperature": T,
            })

    df = pd.DataFrame(rows)

    if df.empty:
        raise ValueError("No completed tests with data were found in the experiment.")

    # 2. Generate the interactive 3D scatter plot
    fig = px.scatter_3d(
        df,
        x="Applied_Stress_MPa",
        y="Time_s",
        z="Strain",
        color="Temperature",
        symbol="Print_Quality",
        hover_name="Test_ID",
        hover_data={
            "Applied_Stress_MPa": ":.1f",
            "Time_s": ":.1f",
            "Strain": ":.5f",
            "Temperature": ":.1f",
            "Print_Quality": True,
        },
        color_continuous_scale="Turbo",
        labels={
            "Applied_Stress_MPa": "Applied Stress (MPa)",
            "Time_s": "Time (s)",
            "Strain": "Strain",
            "Temperature": "Temperature (°C)",
            "Print_Quality": "Print Quality",
        },
        title="Creep Test Curves — Stress / Time / Strain",
    )

    # 3. Marker styling — small, semi-transparent so overlapping tests remain readable
    fig.update_traces(marker=dict(size=3, opacity=0.75, line=dict(width=0)))

    # 4. Layout: separate the continuous colorbar (temperature) from the
    #    discrete symbol legend (print quality) so they don't overlap, and
    #    give the whole figure a clean, professional look.
    fig.update_layout(
        template="plotly_white",
        title=dict(x=0.5, xanchor="center", font=dict(size=20)),
        font=dict(family="Arial, sans-serif", size=12),
        margin=dict(l=0, r=0, t=60, b=0),
        scene=dict(
            xaxis=dict(title="Applied Stress (MPa)", backgroundcolor="white",
                       gridcolor="lightgrey", showbackground=True),
            yaxis=dict(title="Time (s)", backgroundcolor="white",
                       gridcolor="lightgrey", showbackground=True),
            zaxis=dict(title="Strain", backgroundcolor="white",
                       gridcolor="lightgrey", showbackground=True),
            aspectmode="cube",
            camera=dict(eye=dict(x=1.6, y=1.6, z=1.1)),
        ),
        # Discrete "Print Quality" legend: pinned to the right, below the colorbar
        legend=dict(
            title=dict(text="Print Quality"),
            x=1.02, y=0.3, xanchor="left", yanchor="middle",
            bgcolor="rgba(255,255,255,0.85)",
            bordercolor="lightgrey", borderwidth=1,
        ),
        # Continuous "Temperature" colorbar: pinned to the right, above the legend
        coloraxis_colorbar=dict(
            title=dict(text="Temperature (°C)"),
            x=1.02, xanchor="left",
            y=0.78, yanchor="middle", len=0.5,
        ),
        width=1000,
        height=750,
    )

    fig.show()


if __name__ == "__main__":
    main()