"""
Initial condition for the TLV ODE (Thesis Eq. 1.4, 1.5).

sigma_ep_0 is a DIRECT MODEL INPUT (the ODE's starting value), not an
independently-fit validation target -- this was the open question flagged
in earlier planning and is resolved by Eq. 1.4 itself.
"""
from creep_model.modelling.tlv.parameters import TLVParameters


def f_ratio(T_kelvin: float, params: TLVParameters) -> float:
    """Eq. 1.5 -- f(T) = Ev(T) / (Ev(T) + Ee(T))."""
    p = params.at_temperature(T_kelvin)
    return p["Ev"] / (p["Ev"] + p["Ee"])


def sigma_ep_0(applied_stress_MPa: float, T0_kelvin: float, params: TLVParameters) -> float:
    """
    Eq. 1.4 -- sigma_ep_0 = (1 - f(T0)) * sigma.

    Args:
        applied_stress_MPa: constant applied stress for this test.
        T0_kelvin: temperature (Kelvin) at the first recorded time point.
        params: candidate/fitted TLVParameters.

    Returns:
        sigma_ep at t=0, MPa.
    """
    return (1.0 - f_ratio(T0_kelvin, params)) * applied_stress_MPa

def sigma_ep_0_from_measurement(eps_measured_0: float, T0_kelvin: float, params: TLVParameters) -> float:
    """
    Alternative initial condition: instead of deriving sigma_ep_0
    theoretically from Eq. 1.4/1.5 (sigma_ep_0 = (1-f(T0))*sigma), calibrate
    it directly from the first recorded strain measurement by solving
    Eq. 1.2b for sigma_ep:

        sigma_ep(t_0) = eps_measured(t_0) * Ee(T(t_0))

    """
    Ee_T0 = params.at_temperature(T0_kelvin)["Ee"]
    return eps_measured_0 * Ee_T0