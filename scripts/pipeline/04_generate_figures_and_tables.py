"""
Generates EXACTLY the figures and tables referenced in the results
chapter -- nothing else. Replaces both 04_generate_tlv_plots_and_tables.py
and 05_generate_eda_plots_and_tables.py.

Writes directly into the local Overleaf git-synced project folder using
STABLE filenames (never timestamped), so re-running this script after a
pipeline re-run updates the figures Overleaf displays with no manual
copying or renaming step -- just commit + push the changed files from
your Overleaf repo as usual.

Dependency split (deliberately kept explicit, since 02_fit_tlv.py is the
one genuinely slow step in this pipeline):
  - Fig 1.1, Fig 1.4          : need only 01_classify_and_trim.py's output
                                 (raw/trimmed data) -- regenerate in seconds,
                                 no TLV fit required.
  - Table 1.1, Fig 1.2, Fig 1.3: need eda_results.h5 (03_compute_eda_stats.py)
                                 -- also fast, no TLV fit required.
  - Fig 1.5, Table 1.2        : need tlv_fit_results.h5 (02_fit_tlv.py) --
                                 the only outputs here that actually
                                 require the expensive fit.
"""
from pathlib import Path

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from creep_model.config import config
from creep_model.io.parser import ExcelCreepParser
from creep_model.viz.eda_plots import (
    EDAStyleConfig,
    plot_temperature_fit_example,
    plot_creep_stage_boundaries,
    plot_eps_tilde_0_vs_stress,
    plot_eps_dot_ss_vs_stress,
    plot_pairwise_relationships,
)
from creep_model.viz.tlv_plots import TLVStyleConfig, plot_tlv_fit_summary

# Set output directories for Overleaf figures and tables 
table_and_figures_dir = Path(config.figures_and_tables_directory)

DIR_TEMP_HANDLING =  table_and_figures_dir/ "5.1"   # Fig 1.1
DIR_EXPERIMENTAL =  table_and_figures_dir/ "5.2"    # Table 1.1, Fig 1.2, 1.3, 1.4
DIR_MODELLING =  table_and_figures_dir/ "5.3"       # Fig 1.5, Table 1.2

# --- Which specific tests appear in the chapter -----------------------------
TEMP_FIT_EXAMPLE_TESTS = ["H.10.4", "S.30.1"]   # Fig 1.1 (a), (b)
STAGE_BOUNDARY_EXAMPLE_TESTS = ["S.30.2", "H.30.3"]         # Fig 1.4

DATA_PATH = Path(config.data_directory) / "CreepData.xlsx"
EDA_H5_PATH = Path(config.data_output_directory) / "eda_results.h5"
FIT_H5_PATH = Path(config.data_output_directory) / "tlv_fit_results.h5"

K1 = config.K1
K2 = config.K2


# ============================================================
# Fig 1.1, Fig 1.4 -- fast, raw-data-only (no TLV fit needed)
# ============================================================

def generate_raw_data_figures() -> None:
    parser = ExcelCreepParser(DATA_PATH)
    experiment = parser.load_experiment()
    style = EDAStyleConfig()

    print("Generating Fig 1.1 (temperature-fit examples)...")
    for test_id in TEMP_FIT_EXAMPLE_TESTS:
        out_path = plot_temperature_fit_example(experiment.tests[test_id], DIR_TEMP_HANDLING, style)
        print(f"  Saved: {out_path}")

    print("Generating Fig 1.4 (stage-boundary examples)...")
    for test_id in STAGE_BOUNDARY_EXAMPLE_TESTS:
        test = experiment.tests[test_id]
        out_path = plot_creep_stage_boundaries(test, K1, K2, DIR_EXPERIMENTAL, style)
        print(f"  Saved: {out_path}")


# ============================================================
# Table 1.1, Fig 1.2, Fig 1.3 -- from eda_results.h5 (fast)
# ============================================================

def load_eda_dataframe() -> pd.DataFrame:
    if not EDA_H5_PATH.exists():
        raise FileNotFoundError(f"{EDA_H5_PATH} not found -- run 03_compute_eda_stats.py first.")
    with h5py.File(EDA_H5_PATH, "r") as f:
        eda_group = f["eda_summary"]
        data = {}
        for key in eda_group.keys():
            arr = eda_group[key][:]
            if arr.dtype.kind in ["S", "O"]:
                arr = [x.decode("utf-8") if isinstance(x, bytes) else str(x) for x in arr]
            data[key] = arr
        df = pd.DataFrame(data)
    df["Nominal_Stress_MPa"] = df["Applied_Stress_MPa"].round(-1)
    return df


def generate_eda_figures_and_table(df: pd.DataFrame) -> None:
    style = EDAStyleConfig()

    print("Generating Fig 1.3a (eps_tilde_0 vs stress)...")
    out_path = plot_eps_tilde_0_vs_stress(df, DIR_EXPERIMENTAL, style)
    print(f"  Saved: {out_path}")

    print("Generating Fig 1.3b (eps_dot_ss vs stress)...")
    out_path = plot_eps_dot_ss_vs_stress(df, DIR_EXPERIMENTAL, style)
    print(f"  Saved: {out_path}")

    print("Generating Fig 1.2 (pairwise relationships)...")
    out_path = plot_pairwise_relationships(df, DIR_EXPERIMENTAL, style)
    print(f"  Saved: {out_path}")

    print("Generating Table 1.1 (EDA summary)...")
    generate_eda_latex_table(df, DIR_EXPERIMENTAL)


def generate_eda_latex_table(df: pd.DataFrame, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    tex_path = output_dir / "eda_summary_table.tex"

    def calc_stats(series, scale=1.0, fmt="{:.2f}", filter_positive=False):
        valid = series[series.notna() & (series > 0)] if filter_positive else series.dropna()
        count = len(valid)
        if count == 0:
            return "N/A*", "N/A*"
        mean_str = fmt.format(valid.mean() * scale)
        if count <= 1:
            return mean_str, "N/A"
        v_min, v_max = valid.min() * scale, valid.max() * scale
        range_str = "N/A" if np.isclose(v_min, v_max) else f"{fmt.format(v_min)}--{fmt.format(v_max)}"
        return mean_str, range_str

    latex = [
        r"\begin{table}[H]", r"    \centering",
        r"    \caption{Exploratory Data Analysis (EDA) statistics for creep tests grouped by print quality and stress level.}",
        r"    \label{tab:eda_summary}", r"    \resizebox{\linewidth}{!}{%",
        r"    \begin{tabular}{ccccccccc}", r"        \toprule",
        r"        \textbf{$\sigma$ (MPa)} & \textbf{$N$} & \textbf{$N_{sec}$} & \textbf{$N_{tert}$} & "
        r"\textbf{Statistic} & \textbf{Initial Temp ($^\circ$C)} & \textbf{Mean Temp ($^\circ$C)} & "
        r"\textbf{$\tilde{\varepsilon}_0$ ($\times 10^{-3}$)} & \textbf{$\dot{\varepsilon}_{ss}$ ($\times 10^{-7}$s$^{-1}$)} \\",
        r"        \midrule",
    ]

    for q_idx, quality in enumerate(["High", "Standard"]):
        q_df = df[df["Print_Quality"] == quality]
        if q_df.empty:
            continue
        latex.append(f"        \\multicolumn{{9}}{{l}}{{\\textit{{{quality} Quality}}}} \\\\")
        latex.append(r"        \midrule")
        for stress in sorted(q_df["Nominal_Stress_MPa"].unique()):
            sub = q_df[q_df["Nominal_Stress_MPa"] == stress]
            n_total = len(sub)
            n_sec = int((sub["Eps_Dot_Ss"].notna() & (sub["Eps_Dot_Ss"] > 0)).sum())
            n_tert = 0  # tertiary count not tracked in eda_summary -- fill in manually if needed

            t_init_m, t_init_r = calc_stats(sub["Initial_Temp_C"], fmt="{:.1f}")
            t_sec_m, t_sec_r = calc_stats(sub["Mean_Temp_C_Secondary_Creep"], fmt="{:.1f}")
            eps0_m, eps0_r = calc_stats(sub["Eps_Tilde_0"], scale=1e3, fmt="{:.2f}")
            eps_ss_m, eps_ss_r = calc_stats(sub["Eps_Dot_Ss"], scale=1e7, fmt="{:.2f}", filter_positive=True)

            latex.append(f"        {int(stress)} & {n_total} & {n_sec} & {n_tert} & Mean & "
                          f"{t_init_m} & {t_sec_m} & {eps0_m} & {eps_ss_m} \\\\")
            latex.append(f"        & & & & Range & {t_init_r} & {t_sec_r} & {eps0_r} & {eps_ss_r} \\\\")
            latex.append(r"        \addlinespace")
        if q_idx == 0:
            latex.append(r"        \midrule")

    latex.extend([
        r"        \bottomrule", r"    \end{tabular}%", r"    }",
        r"    \begin{flushleft}",
        r"        \footnotesize{*N/A indicates secondary creep was not detected for any specimens in this test group.}",
        r"    \end{flushleft}", r"\end{table}",
    ])

    with open(tex_path, "w", encoding="utf-8") as f:
        f.write("\n".join(latex))
    print(f"  Saved: {tex_path}")


# ============================================================
# Fig 1.5, Table 1.2 -- from tlv_fit_results.h5 (needs the fit)
# ============================================================

def compute_quality_fit_stats(f: h5py.File, quality: str) -> dict[str, float]:
    """Computes R2, RMSE, MAPE, and SSE across all test predictions of a given print quality."""
    y_true_list = []
    y_pred_list = []
    
    if "tests" not in f:
        return {}
        
    tests_grp = f["tests"]
    for test_id in tests_grp.keys():
        t_grp = tests_grp[test_id]
        q = t_grp.attrs.get("print_quality")
        if q == quality and "strain_measured" in t_grp and "strain_predicted" in t_grp:
            y_t = t_grp["strain_measured"][:]
            y_p = t_grp["strain_predicted"][:]
            if len(y_t) > 0 and len(y_t) == len(y_p):
                y_true_list.append(y_t)
                y_pred_list.append(y_p)
                
    if not y_true_list:
        return {}
        
    y_true = np.concatenate(y_true_list)
    y_pred = np.concatenate(y_pred_list)
    
    sse = float(np.sum((y_true - y_pred) ** 2))
    sst = float(np.sum((y_true - np.mean(y_true)) ** 2))
    r2 = float(1.0 - (sse / sst)) if sst != 0 else np.nan
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    
    nonzero_mask = y_true != 0
    if np.any(nonzero_mask):
        mape = float(np.mean(np.abs((y_true[nonzero_mask] - y_pred[nonzero_mask]) / y_true[nonzero_mask])) * 100.0)
    else:
        mape = np.nan
        
    return {
        "R2": r2,
        "RMSE": rmse,
        "MAPE": mape,
        "SSE": sse,
    }


def generate_modelling_figures_and_table() -> None:
    if not FIT_H5_PATH.exists():
        print(f"WARNING: {FIT_H5_PATH} not found -- skipping Fig 1.5 / Table 1.2. "
              "Run 02_fit_tlv.py first.")
        return

    print("Generating Fig 1.5 (per-quality summary grids)...")
    with h5py.File(FIT_H5_PATH, "r") as f:
        tests_group = f["tests"]
        grouped: dict[tuple, list[dict]] = {}
        for test_id in tests_group.keys():
            g = tests_group[test_id]
            quality = g.attrs["print_quality"]
            stress = float(g.attrs["applied_stress_MPa"])
            nominal = round(stress, -1)
            grouped.setdefault((quality, nominal), []).append({
                "test_id": test_id,
                "time_s": g["time_s"][:],
                "strain_measured": g["strain_measured"][:],
                "strain_predicted": g["strain_predicted"][:],
                "stress": stress,
            })

    style = TLVStyleConfig()
    for quality in ["High", "Standard"]:
        out_path = plot_tlv_fit_summary(quality, grouped, [10.0, 20.0, 30.0], DIR_MODELLING, style)
        print(f"  Saved: {out_path}")

    print("Generating Table 1.2 (TLV Summary Table with Metrics)...")
    generate_parameter_table()


def generate_parameter_table() -> None:
    with h5py.File(FIT_H5_PATH, "r") as f:
        high_attrs = dict(f["fitted_parameters"]["High"].attrs) if "High" in f["fitted_parameters"] else {}
        std_attrs = dict(f["fitted_parameters"]["Standard"].attrs) if "Standard" in f["fitted_parameters"] else {}

        high_stats = compute_quality_fit_stats(f, "High")
        std_stats = compute_quality_fit_stats(f, "Standard")

    key_aliases = {
        "R2": ["R2", "r2", "r_squared", "r2_score", "rsquared", "R_squared"],
        "RMSE": ["RMSE", "rmse", "rmse_val", "root_mean_squared_error"],
        "MAPE": ["MAPE", "mape", "mape_val", "mean_absolute_percentage_error"],
        "SSE": ["SSE", "sse", "sse_val", "sum_squared_error", "sum_squared_errors", "ss_res"],
    }

    def get_val(attrs, computed_stats, key):
        aliases = key_aliases.get(key, [key, key.lower(), key.upper()])
        for a in aliases:
            if a in attrs:
                val = attrs[a]
                if val is not None and not (isinstance(val, float) and np.isnan(val)):
                    return val
        if key in computed_stats:
            return computed_stats[key]
        return None

    param_rows = [
        (r"$A_{20}$", r"$\text{MPa}^{-n}\cdot\text{s}^{-(m+1)}$", "A20", "{:.2e}"),
        (r"$A_{30}$", r"$\text{MPa}^{-n}\cdot\text{s}^{-(m+1)}$", "A30", "{:.2e}"),
        (r"$n_{20}$", r"--", "n20", "{:.3f}"),
        (r"$n_{30}$", r"--", "n30", "{:.3f}"),
        (r"$m_{20}$", r"--", "m20", "{:.3f}"),
        (r"$m_{30}$", r"--", "m30", "{:.3f}"),
        (r"$E_{e,20}$", r"MPa", "Ee20", "{:.1f}"),
        (r"$E_{e,30}$", r"MPa", "Ee30", "{:.1f}"),
        (r"$E_{v,20}$", r"MPa", "Ev20", "{:.1f}"),
        (r"$E_{v,30}$", r"MPa", "Ev30", "{:.1f}"),
        (r"midrule", "", "", ""),
        (r"$R^2$", r"--", "R2", "{:.4f}"),
        (r"RMSE", r"--", "RMSE", "{:.2e}"),
        (r"MAPE", r"\%", "MAPE", "{:.2f}"),
        (r"SSE", r"--", "SSE", "{:.2e}"),
    ]

    def fmt(val, f):
        if val is None or (isinstance(val, float) and np.isnan(val)):
            return "N/A"
        s = f.format(val)
        if "e" in s:
            s = s.replace("e", r" \times 10^{").replace("+0", "").replace("+", "").replace("-0", "-") + "}"
            s = f"${s}$"
        return s

    latex = [
        r"\begin{table}[H]", r"    \centering",
        r"    \caption{Fitted TLV model parameters and fit summary statistics for High and Standard print qualities.}",
        r"    \label{tab:tlv_summary}", r"    \begin{tabular}{llcc}", r"        \toprule",
        r"        \textbf{Parameter / Metric} & \textbf{Units} & \textbf{High Quality} & \textbf{Standard Quality} \\", r"        \midrule",
    ]
    
    for symbol, units, key, f in param_rows:
        if symbol == r"midrule":
            latex.append(r"        \midrule")
            continue
            
        val_h = get_val(high_attrs, high_stats, key)
        val_s = get_val(std_attrs, std_stats, key)
        latex.append(f"        {symbol} & {units} & {fmt(val_h, f)} & {fmt(val_s, f)} \\\\")
        
    latex.extend([r"        \bottomrule", r"    \end{tabular}", r"\end{table}"])

    DIR_MODELLING.mkdir(parents=True, exist_ok=True)
    out_path = DIR_MODELLING / "tlv_summary_table.tex"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(latex))
    print(f"  Saved: {out_path}")


def main() -> None:
    generate_raw_data_figures()
    df = load_eda_dataframe()
    generate_eda_figures_and_table(df)
    generate_modelling_figures_and_table()
    print("\nDone. Commit + push the changed files from your Overleaf repo to update the compiled document.")


if __name__ == "__main__":
    main()