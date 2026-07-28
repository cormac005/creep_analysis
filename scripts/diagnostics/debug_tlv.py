"""
Fast, no-refit diagnostics for pathological (constant/decreasing) TLV
predictions.

Uses ONLY the already-fitted parameters cached in
data/processed/tlv_fit_results.h5 -- nothing here re-runs DE or LM.
solve_tlv() itself runs in milliseconds per test; the expensive part of
process_data_and_fit.py is the parameter SEARCH, not the ODE integration,
so this whole script should complete in a few seconds.

What this checks:
  1. How much Ee/Ev actually change over each test's observed temperature
     range, at the fitted parameters -- a sanity check on whether the
     linear-in-T assumption (Eq. 1.3) is behaving mildly (as intended for
     ~3.5C ambient swings) or has been fit into an implausible regime.
  2. An ABLATION: re-solves the same test with dEe/dT and dEv/dT forced to
     zero (keeping A, n, m, Ee, Ev otherwise untouched, evaluated at the
     test's mean temperature). If the ablated curve is monotonically
     increasing where the full model isn't, that's strong evidence the
     temperature-coupling terms (not A/n/m or the solver itself) are the
     problem.
"""
from pathlib import Path

import h5py
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from creep_model.io.parser import ExcelCreepParser
from creep_model.modelling.tlv.parameters import TLVParameters
from creep_model.modelling.tlv.solver import solve_tlv, SolverConvergenceError

DATA_PATH = Path("data/raw/CreepData.xlsx")
H5_PATH = Path("data/processed/tlv_fit_results.h5")
OUTPUT_DIR = Path("diagnostics")

# How many tests per print-quality group to plot -- start small, widen once
# you've confirmed/refuted the hypothesis on a couple of examples.
N_TESTS_PER_GROUP = 3


def load_fitted_params(quality: str) -> TLVParameters:
    with h5py.File(H5_PATH, "r") as f:
        attrs = dict(f["fitted_parameters"][quality].attrs)
    return TLVParameters(**{k: float(v) for k, v in attrs.items()})


class _ZeroTempCouplingParams:
    """
    Ablation wrapper: same at_temperature() behaviour as the real fitted
    TLVParameters (so A/n/m/Ee/Ev still vary with T as fitted), but
    dEe_dT()/dEv_dT() are forced to 0.

    This isolates whether the *coupling terms* in the ODE (the ones that
    fire only when T changes WITHIN a test, via T_dot) are what's driving
    bad predictions, as opposed to the base A/n/m/Ee/Ev values themselves
    being wrong.
    """
    def __init__(self, params: TLVParameters):
        self._params = params

    def at_temperature(self, T):
        return self._params.at_temperature(T)

    def dEe_dT(self):
        return 0.0

    def dEv_dT(self):
        return 0.0


def diagnose_test(test, params: TLVParameters, quality: str) -> None:
    T = test.interpolate_temperature()
    T_K = T + 273.15
    print(f"\n=== {test.test_id} ({quality}) ===")
    print(f"  Temp range: {T.min():.2f}C to {T.max():.2f}C  (span {T.max() - T.min():.2f}C)")

    p_lo = params.at_temperature(T_K.min())
    p_hi = params.at_temperature(T_K.max())
    print("  Parameter values across THIS test's observed temp range:")
    for key in ["Ee", "Ev", "A", "n", "m"]:
        pct = 100 * (p_hi[key] - p_lo[key]) / p_lo[key] if p_lo[key] != 0 else float("nan")
        print(f"    {key}: {p_lo[key]:.4g} (Tmin) -> {p_hi[key]:.4g} (Tmax)  [{pct:+.1f}%]")

    try:
        eps_full = solve_tlv(test, params)
    except SolverConvergenceError as e:
        print(f"  Full model solver failed: {e}")
        eps_full = None

    ablated_params = _ZeroTempCouplingParams(params)
    try:
        eps_ablated = solve_tlv(test, ablated_params)
    except SolverConvergenceError as e:
        print(f"  Ablated (dEe/dT=dEv/dT=0) solver failed: {e}")
        eps_ablated = None

    # Quick monotonicity check -- cheaper than eyeballing every plot
    if eps_full is not None:
        frac_decreasing = np.mean(np.diff(eps_full) < 0)
        print(f"  Full model: {frac_decreasing:.1%} of steps decreasing")
    if eps_ablated is not None:
        frac_decreasing_ablated = np.mean(np.diff(eps_ablated) < 0)
        print(f"  Ablated model: {frac_decreasing_ablated:.1%} of steps decreasing")

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(test.time_series, test.strain_series, s=8, alpha=0.5, label="Measured")
    if eps_full is not None:
        ax.plot(test.time_series, eps_full, color="red", label="Full TLV fit")
    if eps_ablated is not None:
        ax.plot(test.time_series, eps_ablated, color="green", linestyle="--",
                 label="Ablated (dEe/dT = dEv/dT = 0)")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Strain")
    ax.set_title(f"{test.test_id} -- temperature-coupling ablation")
    ax.legend()
    ax.grid(True, alpha=0.3)

    OUTPUT_DIR.mkdir(exist_ok=True)
    out_path = OUTPUT_DIR / f"{test.test_id}_ablation.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


def main() -> None:
    parser = ExcelCreepParser(DATA_PATH)
    experiment = parser.load_experiment()

    for quality in ["High", "Standard"]:
        params = load_fitted_params(quality)
        print(f"\n{'=' * 60}\n{quality} fitted parameters:\n{params}\n{'=' * 60}")

        candidates = [
            t for t in experiment.tests.values()
            if not t.is_empty and t.print_quality == quality
        ][:N_TESTS_PER_GROUP]

        for test in candidates:
            diagnose_test(test, params, quality)


if __name__ == "__main__":
    main()