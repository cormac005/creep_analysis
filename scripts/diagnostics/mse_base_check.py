"""
Round-2 diagnostics, using ONLY arrays already cached in
tlv_fit_results.h5 -- no re-solving, no re-fitting.

The temperature-coupling ablation showed the ablated (isothermal-
equivalent) curve is essentially flat for every test checked, and is
IDENTICAL to the full model for the Standard group -- so temperature
coupling is not the (main) cause. This checks two more specific
hypotheses:

  A. Initial-condition mismatch: does the model's t=0 strain
     (sigma_ep_0 / Eq. 1.4-1.5) actually match the measured eps_tilde_0?

  B. Is the MSE loss nearly blind to shape here? The total measured
     creep signal is tiny in absolute terms (e.g. Delta ~ 0.0015-0.0075
     strain) -- a flat prediction sitting near the data's mean could
     already achieve a deceptively low MSE. If the TLV fit's MSE is only
     marginally better than (or worse than) a trivial flat baseline,
     that's strong evidence DE/LM settled for a shape-blind minimum
     rather than a genuine fit.
"""
from pathlib import Path

import h5py
import numpy as np

H5_PATH = Path("data/processed/tlv_fit_results.h5")


def main() -> None:
    with h5py.File(H5_PATH, "r") as f:
        tests_group = f["tests"]

        header = (
            f"{'Test':<10} {'quality':<9} {'eps0_meas':>10} {'eps0_pred':>10} "
            f"{'%diff':>7}  {'TLV_MSE':>11} {'Flat_MSE':>11} {'MSE_ratio':>10}"
        )
        print(header)
        print("-" * len(header))

        ratios = []
        for test_id in sorted(tests_group.keys()):
            g = tests_group[test_id]
            if "strain_measured" not in g or "strain_predicted" not in g:
                continue

            measured = g["strain_measured"][:]
            predicted = g["strain_predicted"][:]
            quality = g.attrs.get("print_quality", "?")

            eps0_meas = measured[0]
            eps0_pred = predicted[0]
            pct_diff = 100 * (eps0_pred - eps0_meas) / eps0_meas

            tlv_mse = np.mean((predicted - measured) ** 2)

            # Trivial baseline: a flat line at the measured initial strain
            # -- i.e. "predict no creep happens at all".
            flat_baseline = np.full_like(measured, eps0_meas)
            flat_mse = np.mean((flat_baseline - measured) ** 2)

            ratio = tlv_mse / flat_mse if flat_mse > 0 else float("nan")
            ratios.append(ratio)

            print(
                f"{test_id:<10} {quality:<9} {eps0_meas:>10.5f} {eps0_pred:>10.5f} "
                f"{pct_diff:>6.1f}%  {tlv_mse:>11.3e} {flat_mse:>11.3e} {ratio:>10.3f}"
            )

        print("-" * len(header))
        print(f"Mean MSE ratio (TLV / flat-baseline) across all tests: {np.mean(ratios):.3f}")
        print("(ratio << 1  -> TLV genuinely captures shape better than a flat line)")
        print("(ratio ~ 1   -> TLV barely beats predicting 'no creep at all')")
        print("(ratio > 1   -> TLV is actually WORSE than a flat line)")


if __name__ == "__main__":
    main()