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