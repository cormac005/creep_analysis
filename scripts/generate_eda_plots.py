"""
Generates all thesis-ready EDA figures (docs/methodology.md Sec 1.2) from
the eda_summary table already cached in data/processed/tlv_fit_results.h5.

Saves every figure to docs/figures/eda/, ready to reference directly from
the thesis document.
"""
from pathlib import Path

import h5py
import pandas as pd

from creep_model.viz.eda_plots import (
    EDAStyleConfig,
    plot_eps_tilde_0_vs_stress,
    plot_eps_dot_ss_vs_stress,
    plot_eps_tilde_0_vs_age,
    plot_eps_dot_ss_vs_temperature,
    plot_distribution_by_quality,
    plot_mean_eps_dot_ss_bar,
    plot_pairwise_relationships,
)

H5_PATH = Path("data/processed/tlv_fit_results.h5")
OUTPUT_DIR = Path("docs/figures/eda")

# --- CONFIGURE ALL PLOT STYLING HERE ---
# Modify these parameters to dictate the appearance of all generated plots.
PLOT_STYLE = EDAStyleConfig(
    quality_colors={"High": "#1f77b4", "Standard": "#d62728"},
    quality_markers={"Standard": "o", "High": "s"},
    stress_markers={10: "o", 20: "s", 30: "D"},
    quality_order=["Standard", "High"],
    stress_order=[10.0, 20.0, 30.0],
    cmap="coolwarm",
    dpi=300,
    base_marker_size=75,
    alpha=0.9,
    show_titles=False  # <-- Set to True or False to toggle plot titles
)

# Every EDA figure to generate, in the order they'll be printed/saved.
FIGURE_FUNCTIONS = [
    plot_eps_tilde_0_vs_stress,
    plot_eps_dot_ss_vs_stress,
    plot_eps_tilde_0_vs_age,
    plot_eps_dot_ss_vs_temperature,
    plot_distribution_by_quality,
    plot_mean_eps_dot_ss_bar,
    plot_pairwise_relationships,
]


def load_eda_dataframe(h5_path: Path) -> pd.DataFrame:
    """Loads the eda_summary table written by append_eda_to_h5.py."""
    if not h5_path.exists():
        raise FileNotFoundError(
            f"{h5_path} does not exist -- run process_data_and_fit.py "
            "(and then append_eda_to_h5.py) first."
        )

    with h5py.File(h5_path, "r") as f:
        if "eda_summary" not in f:
            raise KeyError(
                f"'eda_summary' group not found in {h5_path} -- "
                "run scripts/append_eda_to_h5.py first."
            )
        eda_group = f["eda_summary"]
        df = pd.DataFrame({key: eda_group[key][:] for key in eda_group.keys()})

    # Test_ID / Print_Quality are stored as bytes -- decode to str.
    df["Test_ID"] = df["Test_ID"].str.decode("utf-8")
    df["Print_Quality"] = df["Print_Quality"].str.decode("utf-8")
    return df


def main() -> None:
    df = load_eda_dataframe(H5_PATH)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Generating {len(FIGURE_FUNCTIONS)} EDA figures from {len(df)} tests...")
    for fn in FIGURE_FUNCTIONS:
        # Pass the configured style object into each plotting function
        out_path = fn(df, OUTPUT_DIR, PLOT_STYLE)
        print(f"  Saved: {out_path}")

    print(f"\nAll EDA figures saved to {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()