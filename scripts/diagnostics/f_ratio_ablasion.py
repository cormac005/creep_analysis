"""
Tests whether hardcoding f=0.5 in the initial condition (rather than the
current f(T) = Ev(T)/(Ev(T)+Ee(T)) derived from the FITTED moduli) fixes
the systematic initial-strain over-prediction seen in mse_baseline_check.py
-- every single one of 24 tests currently over-predicts eps_tilde_0, by
15-40% in most cases.

No re-fitting: uses the cached fitted parameters and monkeypatches only
f_ratio() as used inside sigma_ep_0() -- A/n/m/Ee/Ev and the rest of the
ODE are untouched. If this alone brings eps_pred[0] much closer to
eps_meas[0], that's strong evidence the initial-condition formula (not
A/n/m, not the solver) is the primary driver of the bad shapes.
"""
from pathlib import Path

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from creep_model.io.parser import ExcelCreepParser
from creep_model.modelling.tlv.parameters import TLVParameters
from creep_model.modelling.tlv.solver import solve_tlv, SolverConvergenceError
import creep_model.modelling.tlv.initial_conditions as ic

DATA_PATH = Path("data/raw/CreepData.xlsx")
H5_PATH = Path("data/processed/tlv_fit_results.h5")
OUTPUT_DIR = Path("diagnostics")

N_TESTS_PER_GROUP = 3


def load_fitted_params(quality: str) -> TLVParameters:
    with h5py.File(H5_PATH, "r") as f:
        attrs = dict(f["fitted_parameters"][quality].attrs)
    return TLVParameters(**{k: float(v) for k, v in attrs.items()})


def main() -> None:
    parser = ExcelCreepParser(DATA_PATH)
    experiment = parser.load_experiment()

    original_f_ratio = ic.f_ratio  # keep a handle so we can always restore it

    print(f"{'Test':<10} {'eps_meas[0]':>12} {'current-f[0]':>13} {'f=0.5[0]':>10}")
    print("-" * 50)

    try:
        for quality in ["High", "Standard"]:
            params = load_fitted_params(quality)
            candidates = [
                t for t in experiment.tests.values()
                if not t.is_empty and t.print_quality == quality
            ][:N_TESTS_PER_GROUP]

            for test in candidates:
                # --- current behaviour: f(T) = Ev/(Ev+Ee), from fitted params ---
                ic.f_ratio = original_f_ratio
                try:
                    eps_current = solve_tlv(test, params)
                except SolverConvergenceError as e:
                    print(f"{test.test_id}: current-f solve failed: {e}")
                    eps_current = None

                # --- ablation: f forced to a constant 0.5 ---
                ic.f_ratio = lambda T, p: 0.5
                try:
                    eps_f05 = solve_tlv(test, params)
                except SolverConvergenceError as e:
                    print(f"{test.test_id}: f=0.5 solve failed: {e}")
                    eps_f05 = None

                eps_meas0 = test.strain_series[0]
                cur0 = eps_current[0] if eps_current is not None else float("nan")
                f05_0 = eps_f05[0] if eps_f05 is not None else float("nan")
                print(f"{test.test_id:<10} {eps_meas0:>12.5f} {cur0:>13.5f} {f05_0:>10.5f}")

                fig, ax = plt.subplots(figsize=(8, 5))
                ax.scatter(test.time_series, test.strain_series, s=8, alpha=0.5, label="Measured")
                if eps_current is not None:
                    ax.plot(test.time_series, eps_current, color="red",
                            label="Current f(T) = Ev/(Ev+Ee)")
                if eps_f05 is not None:
                    ax.plot(test.time_series, eps_f05, color="purple", linestyle="--",
                            label="f forced to 0.5")
                ax.set_xlabel("Time (s)")
                ax.set_ylabel("Strain")
                ax.set_title(f"{test.test_id} -- f=0.5 ablation")
                ax.legend()
                ax.grid(True, alpha=0.3)

                OUTPUT_DIR.mkdir(exist_ok=True)
                out_path = OUTPUT_DIR / f"{test.test_id}_f05_ablation.png"
                fig.savefig(out_path, dpi=150, bbox_inches="tight")
                plt.close(fig)
    finally:
        ic.f_ratio = original_f_ratio  # always restore, even if something raised


if __name__ == "__main__":
    main()