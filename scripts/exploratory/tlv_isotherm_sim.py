import math
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots  # Imported for multi-panel plotting
from creep_model.modelling.tlv.solver import solve_tlv, SolverConvergenceError
import creep_model.modelling.tlv.solver as tlv_solver  

# Set Example Material Properties and Initial Condition
E_e = 10000.0   # 10 GPa
E_v = 10000.0   # 10 GPa
E_eff = E_e * E_v / (E_e + E_v)
f = E_v / (E_e + E_v)
sigma = 30.0 #MPa
A = 3 * 10**-10
m = -0.5 
n = 2.5

# Initial condition
sigma_ep_0 = sigma * (1 - f) 

# Put into params object for solve_tlv function
class MockTLVParameters:
    def at_temperature(self, T):
        if np.isscalar(T):
            return {"Ee": E_e, "Ev": E_v, "A": A, "n": n, "m": m}
        else:
            return {
                "Ee": np.full_like(T, E_e, dtype=float),
                "Ev": np.full_like(T, E_v, dtype=float),
                "A": np.full_like(T, A, dtype=float),
                "n": np.full_like(T, n, dtype=float),
                "m": np.full_like(T, m, dtype=float)
            }
    def dEe_dT(self): 
        return 0.0
    def dEv_dT(self): 
        return 0.0

params = MockTLVParameters()

# Initialise time vector
first_time = 0.0
last_time = 7200
dt = 15 # seconds
t = np.linspace(first_time, last_time, num=int((last_time - first_time) / dt) + 1) 

# Create mock test object for solve_tlv function
class MockCreepTest:
    def __init__(self, time_series, strain_series):
        self.time_series = time_series
        self.strain_series = strain_series
        self.applied_stress_MPa = sigma  

    def interpolate_temperature(self):
        return np.full_like(self.time_series, 293.15)

test_obj = MockCreepTest(time_series=t, strain_series=np.zeros_like(t))

# Monkey-patch the solver's imported sigma_ep_0 function
tlv_solver.sigma_ep_0 = lambda sig, T, p: sigma_ep_0

# Compute closed form isothermal solution
t_safe = np.maximum(t, 1e-12) if (m + 1.0) < 0 else t

if n == 1.0:
    eps_t = (1.0 / E_e) * (sigma - (sigma - sigma_ep_0) * np.exp(-E_eff * A / (m + 1.0) * t_safe**(m + 1.0)))
else:
    base = (sigma - sigma_ep_0)**(1.0 - n) - ((1.0 - n) / (m + 1.0)) * E_eff * A * t_safe**(m + 1.0)
    
    if np.any(base < 0):
        raise ValueError("Invalid state: base for fractional exponentiation is negative.")
        
    eps_t = (1.0 / E_e) * (sigma - base**(1.0 / (1.0 - n)))

if (m + 1.0) < 0:
    eps_t[0] = sigma_ep_0 / E_e

# Compute numerical solution using tlv pipeline
try:
    eps_num = solve_tlv(test=test_obj, params=params)
except SolverConvergenceError as e:
    print(f"Solver failed to converge: {e}")
    eps_num = np.zeros_like(t)

# Calculate solver residuals (Analytical minus Numerical)
residuals = eps_t - eps_num

# ---------------------------------------------------------
# INTERACTIVE PLOTTING SECTION WITH SUBPLOTS
# ---------------------------------------------------------
# Create a 2-row layout sharing the same X-axis (Time)
fig = make_subplots(
    rows=2, cols=1, 
    shared_xaxes=True, 
    vertical_spacing=0.12,
    subplot_titles=(f'TLV ODE Solver Verification (n={n}, m={m})', 'Solver Residuals')
)

# Row 1: Analytical as a solid line
fig.add_trace(go.Scatter(
    x=t, 
    y=eps_t, 
    mode='lines', 
    name='Analytical Solution',
    line=dict(color='black', width=2),
    hovertemplate='Strain: %{y:.6e}'
), row=1, col=1)

# Row 1: Numerical as scattered points
fig.add_trace(go.Scatter(
    x=t, 
    y=eps_num, 
    mode='markers', 
    name='Numerical Solution (Implicit Midpoint)',
    marker=dict(color='red', size=6, opacity=0.7),
    hovertemplate='Strain: %{y:.6e}'
), row=1, col=1)

# Row 2: Residuals as a colored area chart or line
fig.add_trace(go.Scatter(
    x=t, 
    y=residuals, 
    mode='lines', 
    name='Residual (Analytical - Numerical)',
    line=dict(color='blue', width=1.5),
    fill='tozeroy',  # Shades the region under the error curve for visual clarity
    hovertemplate='Residual: %{y:.6e}'
), row=2, col=1)

# Configure layout, titles, and shared hover behavior
fig.update_layout(
    height=700,  # Tall layout to cleanly accommodate both plots
    hovermode='x unified',  
    template='plotly_white',
    legend=dict(x=0.02, y=0.98, bgcolor='rgba(255,255,255,0.6)') 
)

# Axis labels configuration
fig.update_xaxes(title_text='Time (s)', row=2, col=1)
fig.update_yaxes(title_text='Strain (ε)', row=1, col=1)
fig.update_yaxes(title_text='Residual Error (Δε)', row=2, col=1)

# Render the interactive plot
fig.show()
