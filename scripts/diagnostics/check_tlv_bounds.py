"""
Fast, synthetic-data check that fit_group()'s LM/TRF refinement stage
actually respects TLVBounds -- runs in seconds with no real data, so you
don't need to wait through another multi-hour fit to find out whether the
bounds mechanism is actually working.

Deliberately uses a TIGHT bounds box, well inside where an unconstrained
fit would naturally wander -- so a violation here is unambiguous, not a
coincidence of a wide box.
"""
import numpy as np

from creep_model.domain import CreepTest
from creep_model.modelling.optimisation.bounds import TLVBounds
from creep_model.modelling.tlv.fit_pipeline import fit_group


def main() -> None:
    time = np.linspace(10, 500, 30)
    strain = 0.01 + 0.001 * np.sqrt(time / 500)  # plausible primary-creep shape
    temp_time = np.array([0.0, 250.0, 500.0])
    temp_vals = np.array([20.0, 20.3, 20.7])

    test = CreepTest(
        test_id="SYNTH.1",
        time_series=time,
        strain_series=strain,
        temp_time_series=temp_time,
        temperature_readings=temp_vals,
        applied_stress_MPa=10.0,
        age_days=5,
        print_quality="High",
    )

    bounds = TLVBounds(
        A_lower=0.0, A_upper=1e-6,
        n_lower=0.5, n_upper=2.0,
        m_lower=-1.0, m_upper=0.0,
        Ee_lower=1.0, Ee_upper=200.0,
        Ev_lower=1.0, Ev_upper=1000.0,
    )

    params = fit_group(
        [test], bounds,
        de_kwargs={"seed": 42, "maxiter": 15, "popsize": 8, "workers": 1},
    )

    x = params.to_array()
    lower, upper = bounds.lower_array(), bounds.upper_array()
    in_bounds = np.all((x >= lower - 1e-6) & (x <= upper + 1e-6))

    print(params)
    print(f"\nAll parameters within bounds: {in_bounds}")
    if not in_bounds:
        violations = (x < lower - 1e-6) | (x > upper + 1e-6)
        print("BOUNDS ARE NOT BEING ENFORCED. Violating entries "
              f"(index -> value, [lower, upper]): "
              f"{[(i, x[i], lower[i], upper[i]) for i in np.where(violations)[0]]}")
        print("-> check fit_pipeline.py's Stage 2 least_squares() call: "
              "is 'bounds=' actually being passed, and is 'method' still 'trf'?")


if __name__ == "__main__":
    main()