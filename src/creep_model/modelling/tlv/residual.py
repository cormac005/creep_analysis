"""
Residual equation for implicit midpoint integration (Thesis Eq. 1.9-1.11).

IMPORTANT -- read before trusting this file:
The bracket term below is derived directly from the GENERAL Eq. 1.2a (which
keeps Ee(T) and Ev(T) distinct), NOT transcribed verbatim from the Eq. 1.10
text extracted from the PDF. That extracted text uses a single undifferentiated
"E(T)" with no e/v subscript, which -- when checked algebraically -- matches
Eq. 1.2a's bracket ONLY under the substitution Ee(T) = Ev(T). Since Table 1.1
fits Ee20/Ee30 and Ev20/Ev30 as four independent parameters (not a shared
E20/E30 pair), that substitution contradicts the rest of the chapter. The
most likely explanation is that Eq. 1.10's inline subscripts (Ee vs Ev) were
lost during PDF text extraction.

ACTION REQUIRED: open the actual PDF and confirm whether Eq. 1.10 shows two
distinct moduli (Ee, Ev) or one (E). This file assumes two distinct moduli,
consistent with Eq. 1.2a and Table 1.1. If the real equation intentionally
uses Ee=Ev for the solver step only, that's a small, easy edit to
_bracket_term below -- but it needs to be a deliberate choice, not an
artifact of a bad PDF extraction.

R(sigma_ep_{n+1}) = sigma_ep_{n+1} - sigma_ep_n - dt * combined_E(T_mid) * [
        sigma_ep_mid / Ee(T_mid)^2 * dEe/dT * T_dot
      - (sigma - sigma_ep_mid) / Ev(T_mid)^2 * dEv/dT * T_dot
      + A(T_mid) * (sigma - sigma_ep_mid)^n(T_mid) * t_mid^m(T_mid)
    ]

where combined_E(T) = (1/Ee(T) + 1/Ev(T))^-1, matching the prefactor in
Eq. 1.2a exactly.
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

    combined_E = 1.0 / (1.0 / Ee + 1.0 / Ev)  #(1/Ee + 1/Ev)^-1

    elastic_temp_term = (sigma_ep_mid / Ee ** 2) * params.dEe_dT() * T_dot
    viscous_temp_term = ((sigma - sigma_ep_mid) / Ev ** 2) * params.dEv_dT() * T_dot

    # Ensure all stress components make physical sense
    stress_diff = sigma - sigma_ep_mid
    if stress_diff < 0:
        stress_diff = 0.0  
        raise ValueError("Invalid state: sigma_ep_mid > sigma, producing a negative base for the (stress_diff)^n term.")

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
    T_dot: float = None,  # <-- NEW: Optional exact derivative
) -> float:
    """R(sigma_ep_{n+1}) = sigma_ep_{n+1} - sigma_ep_n - dt * bracket(sigma_ep_mid, ...)."""
    sigma_ep_mid = (sigma_ep_n + sigma_ep_next) / 2.0
    T_mid = (T_n + T_next) / 2.0
    t_mid = t_n + dt / 2.0
    
    # <-- NEW: Use exact T_dot if provided, otherwise fallback to finite difference
    if T_dot is None:
        T_dot = (T_next - T_n) / dt

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
    T_dot: float = None,  # <-- NEW: Optional exact derivative
) -> float:
    """
    Analytical Solution for the exact derivative of the residual function 
    with respect to sigma_ep_next (the Jacobian).
    """
    sigma_ep_mid = (sigma_ep_n + sigma_ep_next) / 2.0
    T_mid = (T_n + T_next) / 2.0
    t_mid = t_n + dt / 2.0
    
    # <-- NEW: Use exact T_dot if provided, otherwise fallback to finite difference
    if T_dot is None:
        T_dot = (T_next - T_n) / dt

    p_mid = params.at_temperature(T_mid)
    Ee, Ev, A, n, m = p_mid["Ee"], p_mid["Ev"], p_mid["A"], p_mid["n"], p_mid["m"]
    
    combined_E = 1.0 / (1.0 / Ee + 1.0 / Ev)
    
    elastic_temp_deriv_term = (1.0 / Ee ** 2) * params.dEe_dT() * T_dot
    viscous_temp_deriv_term = (1.0 / Ev ** 2) * params.dEv_dT() * T_dot
    
    # Ensure all stress components make physical sense before taking fractional powers
    stress_diff = sigma - sigma_ep_mid
    if stress_diff < 0:
        raise ValueError("Invalid state: sigma_ep_mid > sigma, producing a negative base for the (stress_diff)^(n-1) term.")
        
    # Guard against division by zero if n < 1 and stress_diff == 0
    if stress_diff == 0.0:
        if n < 1.0:
            raise ZeroDivisionError("stress_diff is 0 and n < 1, causing division by zero in the derivative.")
        else:
            norton_hoff_deriv_term = 0.0
    else:
        norton_hoff_deriv_term = A * n * (stress_diff ** (n - 1.0)) * (t_mid ** m)
        
    # The derivative of the bracket term with respect to sigma_ep_mid
    bracket_deriv = elastic_temp_deriv_term + viscous_temp_deriv_term - norton_hoff_deriv_term
    
    # Apply the chain rule multiplier (0.5) arising from d(sigma_ep_mid)/d(sigma_ep_next)
    return 1.0 - 0.5 * dt * combined_E * bracket_deriv