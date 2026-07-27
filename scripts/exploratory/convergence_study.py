import math
import numpy as np
import time
import plotly.graph_objects as go
from plotly.subplots import make_subplots
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

# Create mock test object for solve_tlv function
class MockCreepTest:
    def __init__(self, time_series, strain_series):
        self.time_series = time_series
        self.strain_series = strain_series
        self.applied_stress_MPa = sigma  

    def interpolate_temperature(self):
        return np.full_like(self.time_series, 293.15)
    
tlv_solver.sigma_ep_0 = lambda sig, T, p: sigma_ep_0

# ---------------------------------------------------------
# CONVERGENCE STUDY AUTOMATION
# ---------------------------------------------------------
# We will test resolutions from 10 points up to 10,000 points
num_points_list = [10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000]

dt_values = []
max_errors = []
compute_times = []

for n_pts in num_points_list:
    # 1. Setup the grid for this iteration
    t_grid = np.linspace(0, 1000, n_pts)
    dt = 1000.0 / (n_pts - 1)
    
    test_obj = MockCreepTest(time_series=t_grid, strain_series=np.zeros_like(t_grid))
    
    # 2. Compute Exact Analytical Solution on this grid
    t_safe = np.maximum(t_grid, 1e-12) if (m + 1.0) < 0 else t_grid
    base = (sigma - sigma_ep_0)**(1.0 - n) - ((1.0 - n) / (m + 1.0)) * E_eff * A * t_safe**(m + 1.0)
    eps_exact = (1.0 / E_e) * (sigma - base**(1.0 / (1.0 - n)))
    if (m + 1.0) < 0:
        eps_exact[0] = sigma_ep_0 / E_e
        
    # 3. Compute Numerical Solution and time it
    start_time = time.perf_counter()
    try:
        eps_num = solve_tlv(test=test_obj, params=params)
    except SolverConvergenceError:
        eps_num = np.full_like(t_grid, np.nan) # Mark as failed
    end_time = time.perf_counter()
    
    # 4. Calculate Metrics
    # Max absolute error across the whole time history
    error = np.max(np.abs(eps_exact - eps_num)) 
    
    dt_values.append(dt)
    max_errors.append(error)
    compute_times.append(end_time - start_time)
    
    print(f"Points: {n_pts:5d} | dt: {dt:6.2f}s | Error: {error:8.2e} | Time: {end_time - start_time:6.4f}s")


# ---------------------------------------------------------
# PLOT CONVERGENCE RESULTS
# ---------------------------------------------------------
fig = make_subplots(rows=1, cols=2, subplot_titles=("Error vs. Time Step (dt)", "Error vs. Compute Time"))

# Subplot 1: Convergence Rate (Log-Log)
fig.add_trace(go.Scatter(
    x=dt_values, y=max_errors, mode='lines+markers',
    name='Max Error', marker=dict(color='blue', size=8)
), row=1, col=1)

fig.update_xaxes(title_text="Time Step dt (s) [Log Scale]", type="log", autorange="reversed", row=1, col=1)
fig.update_yaxes(title_text="Max Strain Error [Log Scale]", type="log", row=1, col=1)

# Subplot 2: Computational Trade-off
fig.add_trace(go.Scatter(
    x=compute_times, y=max_errors, mode='lines+markers',
    name='Compute Time', marker=dict(color='red', size=8)
), row=1, col=2)

fig.update_xaxes(title_text="Solver Time (s) [Log Scale]", type="log", row=1, col=2)
fig.update_yaxes(title_text="Max Strain Error [Log Scale]", type="log", row=1, col=2)

fig.update_layout(title_text=f"Solver Convergence Analysis (n={n}, m={m})", template="plotly_white")
fig.show()