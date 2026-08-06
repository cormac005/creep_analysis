"""
Residual equation for implicit midpoint integration (Thesis Eq. 1.9-1.11).

R(sigma_ep_{n+1}) = sigma_ep_{n+1} - sigma_ep_n - dt * combined_E(T_mid) * [
        sigma_ep_mid / Ee(T_mid)^2 * dEe/dT * T_dot
      - (sigma - sigma_ep_mid) / Ev(T_mid)^2 * dEv/dT * T_dot
      + A(T_mid) * max(0, sigma - sigma_ep_mid)^n(T_mid) * t_mid^m(T_mid)
    ]

where combined_E(T) = (1/Ee(T) + 1/Ev(T))^-1, matching the prefactor in Eq. 1.2a.
"""
from creep_model.modelling.tlv.parameters import TLVParameters


def _bracket_term(
    sigma_ep_mid: float,
    sigma: float,
    T_mid: float,
    T_dot: float,
    t_mid: float,
    params: TLVParameters,
) -> float:
    """The full bracketed RHS of Eq. 1.2a, evaluated at the midpoint state."""
    p_mid = params.at_temperature(T_mid)
    Ee, Ev, A, n, m = p_mid["Ee"], p_mid["Ev"], p_mid["A"], p_mid["n"], p_mid["m"]

    combined_E = 1.0 / (1.0 / Ee + 1.0 / Ev)

    elastic_temp_term = (sigma_ep_mid / Ee ** 2) * params.dEe_dT() * T_dot
    viscous_temp_term = ((sigma - sigma_ep_mid) / Ev ** 2) * params.dEv_dT() * T_dot

    # Smooth non-negative saturation: if sigma_ep_mid >= sigma, effective stress diff is 0
    stress_diff = max(0.0, sigma - sigma_ep_mid)
    norton_hoff_term = A * (stress_diff ** n) * (t_mid ** m)

    bracket = elastic_temp_term - viscous_temp_term + norton_hoff_term
    return combined_E * bracket


def residual(
    sigma_ep_next: float,
    sigma_ep_n: float,
    sigma: float,
    T_n: float,
    T_next: float,
    t_n: float,
    dt: float,
    params: TLVParameters,
    T_dot: float = None,
) -> float:
    """R(sigma_ep_{n+1}) = sigma_ep_{n+1} - sigma_ep_n - dt * bracket(sigma_ep_mid, ...)."""
    sigma_ep_mid = (sigma_ep_n + sigma_ep_next) / 2.0
    T_mid = (T_n + T_next) / 2.0
    t_mid = t_n + dt / 2.0
    
    if T_dot is None:
        T_dot = (T_next - T_n) / dt if dt > 0 else 0.0

    bracket = _bracket_term(sigma_ep_mid, sigma, T_mid, T_dot, t_mid, params)
    return sigma_ep_next - sigma_ep_n - dt * bracket


def residual_derivative(
    sigma_ep_next: float,
    sigma_ep_n: float,
    sigma: float,
    T_n: float,
    T_next: float,
    t_n: float,
    dt: float,
    params: TLVParameters,
    T_dot: float = None,
) -> float:
    """
    Exact analytical derivative of the residual function with respect to sigma_ep_next,
    using smooth non-negative saturation for stress_diff.
    """
    sigma_ep_mid = (sigma_ep_n + sigma_ep_next) / 2.0
    T_mid = (T_n + T_next) / 2.0
    t_mid = t_n + dt / 2.0
    
    if T_dot is None:
        T_dot = (T_next - T_n) / dt if dt > 0 else 0.0

    p_mid = params.at_temperature(T_mid)
    Ee, Ev, A, n, m = p_mid["Ee"], p_mid["Ev"], p_mid["A"], p_mid["n"], p_mid["m"]
    
    combined_E = 1.0 / (1.0 / Ee + 1.0 / Ev)
    elastic_temp_deriv_term = (1.0 / Ee ** 2) * params.dEe_dT() * T_dot
    viscous_temp_deriv_term = (1.0 / Ev ** 2) * params.dEv_dT() * T_dot
    
    stress_diff = max(0.0, sigma - sigma_ep_mid)
    if stress_diff <= 0.0:
        norton_hoff_deriv_term = 0.0
    else:
        norton_hoff_deriv_term = A * n * (stress_diff ** (n - 1.0)) * (t_mid ** m)
        
    bracket_deriv = elastic_temp_deriv_term + viscous_temp_deriv_term - norton_hoff_deriv_term
    return 1.0 - 0.5 * dt * combined_E * bracket_deriv