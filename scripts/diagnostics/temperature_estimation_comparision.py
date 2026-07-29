"""
Systematic comparison of temperature time-history estimation methods for
a CreepTest's T(t) and T_dot(t), used to feed the TLV ODE's temperature-
coupling terms.

Raw temperature is recorded on a coarse, independent time base
(temp_time_series/temperature_readings), rounded to 0.1C by the logger --
visible as a genuine staircase in the raw data (see H.30.4's recorded
values). The physical prior, based on visual inspection of that raw data,
is that the TRUE underlying signal is smooth and monotonically increasing
throughout each test; the staircase is a sensor-resolution artifact, not
real behaviour.

This script does NOT auto-select a winner -- consistent with this
project's k1/k2 tuning approach (manual tuning, visually justified), it
produces a quantitative metrics table plus plots for several candidate
FAMILIES of method, and a smoothing-strength sweep within the smoothing-
spline family, so the final choice can be made and documented explicitly
rather than left to an automatic criterion (e.g. GCV) that has no
knowledge of the monotonicity prior.

Metrics reported per method:
  RMSE(C)     -- fidelity to the raw (quantized) readings. Not something
                 to minimize blindly -- an s=0 interpolant will "win" this
                 trivially while reproducing every rounding artifact.
  %Tdot<0     -- fraction of the test's actual time grid where the
                 estimated dT/dt is negative. Directly tests the
                 monotonicity prior; large values here contradict the
                 visual evidence that temperature only rises.
  max|Tdot|   -- largest-magnitude derivative produced. A method that's
                 "smooth on average" but has occasional large spikes can
                 still dominate the ODE's temperature-coupling term at
                 those instants.
  Roughness   -- std of consecutive differences in T_dot on the actual
                 time grid; a discrete proxy for how "jumpy" the
                 derivative is (lower = smoother).
"""
from pathlib import Path
from dataclasses import dataclass
from typing import Callable

import numpy as np
from scipy.interpolate import make_smoothing_spline, PchipInterpolator
from sklearn.isotonic import IsotonicRegression
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from creep_model.io.parser import ExcelCreepParser

DATA_PATH = Path("data/raw/CreepData.xlsx")
OUTPUT_DIR = Path("diagnostics/temperature_methods")

# A handful of representative tests -- include the known-wiggly H.30.3 and
# the H.30.4 test whose raw readings you inspected, plus a couple more for
# contrast. Widen this list once a method is chosen, to confirm it behaves
# well across the whole dataset before committing to a re-fit.
TEST_IDS_TO_CHECK = ["H.30.3", "H.30.4", "H.10.1", "S.20.1"]

# Smoothing-spline lambda sweep -- None uses scipy's automatic GCV choice;
# larger explicit values enforce progressively stronger smoothing.
LAMBDA_SWEEP = [None, 1e-4, 1e-3, 1e-2, 1e-1]


@dataclass
class TemperatureMethod:
    name: str
    T_fn: Callable[[np.ndarray], np.ndarray]      # T(t)
    Tdot_fn: Callable[[np.ndarray], np.ndarray]   # dT/dt(t)


def _finite_diff_derivative(T_fn, t, h=1e-2):
    """Fallback for methods without an analytic derivative available."""
    return (T_fn(t + h) - T_fn(t - h)) / (2 * h)


def build_family_methods(test_id: str, temp_time: np.ndarray, temp_vals: np.ndarray) -> list[TemperatureMethod]:
    """One representative method per FAMILY, for a first-pass comparison."""
    methods = []

    def linear_T(t):
        return np.interp(t, temp_time, temp_vals)
    methods.append(TemperatureMethod(
        "Linear (original)", linear_T, lambda t: _finite_diff_derivative(linear_T, t),
    ))

    pchip = PchipInterpolator(temp_time, temp_vals)
    methods.append(TemperatureMethod("PCHIP", pchip, pchip.derivative()))

    spline_gcv = make_smoothing_spline(temp_time, temp_vals)
    methods.append(TemperatureMethod("Smoothing spline (GCV)", spline_gcv, spline_gcv.derivative()))

    # NEW: Smoothing spline with Dithering
    # Adds deterministic, uniformly distributed noise bounded by the sensor's
    # quantization step (+/- 0.05 C) to statistically "un-round" the staircase.
    seed = abs(hash(test_id)) % (2**32)
    rng = np.random.default_rng(seed)
    dithered_vals = temp_vals + rng.uniform(-0.05, 0.05, size=len(temp_vals))
    
    spline_dither = make_smoothing_spline(temp_time, dithered_vals)
    methods.append(TemperatureMethod(
        "Smoothing spline (GCV) + Dithering", 
        spline_dither, 
        spline_dither.derivative()
    ))

    # Isotonic regression (PAVA) enforces monotonic non-decreasing fit
    # directly, then PCHIP over the (now monotonic) fitted values gives a
    # continuously differentiable curve. PCHIP is shape-preserving, so
    # smoothing a monotonic input this way cannot introduce a decrease.
    iso = IsotonicRegression(increasing=True, out_of_bounds="clip")
    iso_vals = iso.fit_transform(temp_time, temp_vals)
    iso_smooth = PchipInterpolator(temp_time, iso_vals)
    methods.append(TemperatureMethod("Isotonic regression + PCHIP", iso_smooth, iso_smooth.derivative()))

    coeffs = np.polyfit(temp_time, temp_vals, deg=2)
    poly = np.poly1d(coeffs)
    methods.append(TemperatureMethod("Quadratic polynomial fit", poly, poly.deriv()))

    return methods


def build_lambda_sweep_methods(temp_time: np.ndarray, temp_vals: np.ndarray) -> list[TemperatureMethod]:
    """Smoothing-spline family only, varying the smoothing strength."""
    methods = []
    for lam in LAMBDA_SWEEP:
        spline = make_smoothing_spline(temp_time, temp_vals, lam=lam)
        label = "GCV (auto)" if lam is None else f"lam={lam:g}"
        methods.append(TemperatureMethod(f"Smoothing spline, {label}", spline, spline.derivative()))
    return methods


def score_method(method: TemperatureMethod, temp_time, temp_vals, fine_t):
    T_at_raw = method.T_fn(temp_time)
    rmse = float(np.sqrt(np.mean((T_at_raw - temp_vals) ** 2)))

    Tdot_fine = method.Tdot_fn(fine_t)
    frac_negative = float(np.mean(Tdot_fine < 0))
    max_abs_tdot = float(np.max(np.abs(Tdot_fine)))
    roughness = float(np.std(np.diff(Tdot_fine)))

    return rmse, frac_negative, max_abs_tdot, roughness


def _run_comparison(test_id, temp_time, temp_vals, fine_t, methods, plot_suffix, header_printed):
    if not header_printed[0]:
        header = f"{'Test':<10} {'Method':<36} {'RMSE(C)':>9} {'%Tdot<0':>9} {'max|Tdot|':>11} {'Roughness':>11}"
        print(header)
        print("-" * len(header))
        header_printed[0] = True

    fig, (ax_T, ax_Tdot) = plt.subplots(2, 1, figsize=(10, 9), sharex=True)
    ax_T.scatter(temp_time, temp_vals, color="black", s=20, zorder=5, label="Raw readings")

    for m in methods:
        rmse, frac_neg, max_tdot, roughness = score_method(m, temp_time, temp_vals, fine_t)
        print(f"{test_id:<10} {m.name:<36} {rmse:>9.4f} {frac_neg * 100:>8.1f}% "
              f"{max_tdot:>11.3e} {roughness:>11.3e}")

        ax_T.plot(fine_t, m.T_fn(fine_t), label=m.name, alpha=0.85)
        ax_Tdot.plot(fine_t, m.Tdot_fn(fine_t), label=m.name, alpha=0.85)

    ax_T.set_ylabel("Temperature (C)")
    ax_T.set_title(f"{test_id} -- T(t) estimation comparison ({plot_suffix})")
    ax_T.legend(fontsize=8, loc='upper left', bbox_to_anchor=(1, 1))
    ax_T.grid(True, alpha=0.3)

    ax_Tdot.axhline(0, color="black", linewidth=0.8)
    ax_Tdot.set_xlabel("Time (s)")
    ax_Tdot.set_ylabel("dT/dt (C/s)")
    ax_Tdot.set_title("Corresponding T_dot(t) -- what actually drives the ODE")
    ax_Tdot.legend(fontsize=8, loc='upper left', bbox_to_anchor=(1, 1))
    ax_Tdot.grid(True, alpha=0.3)

    out_path = OUTPUT_DIR / f"{test_id}_{plot_suffix}.png"
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}\n")


def main() -> None:
    parser = ExcelCreepParser(DATA_PATH)
    experiment = parser.load_experiment()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    header_printed = [False]
    for test_id in TEST_IDS_TO_CHECK:
        test = experiment.tests[test_id]
        
        # Safely filter out NaNs to prevent SciPy interpolators from crashing
        valid_mask = ~np.isnan(test.temperature_readings)
        temp_time = test.temp_time_series[valid_mask]
        temp_vals = test.temperature_readings[valid_mask]
        
        fine_t = test.time_series  # evaluate on the ACTUAL grid solve_tlv uses

        # Pass test_id so the dithering noise can be seeded deterministically
        family_methods = build_family_methods(test_id, temp_time, temp_vals)
        _run_comparison(test_id, temp_time, temp_vals, fine_t, family_methods,
                         "method_families", header_printed)

        sweep_methods = build_lambda_sweep_methods(temp_time, temp_vals)
        _run_comparison(test_id, temp_time, temp_vals, fine_t, sweep_methods,
                         "smoothing_sweep", header_printed)


if __name__ == "__main__":
    main()