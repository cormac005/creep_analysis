# Methodology

This document summarizes the theoretical approach implemented (or being implemented) by this package, mirroring Chapter 3 of the thesis. It's intended as the "why" companion to the code — the README explains what to run, this explains what it's doing and why each choice was made.

---

## 1. Exploratory Data Analysis & Pre-Processing

### 1.1 Creep stage classification

Strain data is recorded at fixed time intervals but is **quantized** — the sensor can only report discrete strain levels, so the raw signal looks like a staircase of flat "plateaus" rather than a smooth curve. This package turns that quantization from a nuisance into a diagnostic tool.

Because strain increases monotonically with time under constant applied stress, the *number of consecutive data points spent on each plateau* is directly related to the local strain rate at that point in the test:

- **Primary creep**: time spent on each successive plateau is *increasing* → strain rate is decreasing.
- **Secondary creep**: time spent on plateaus is roughly *constant* → strain rate has stabilized.
- **Tertiary creep**: time spent on plateaus is *decreasing* → strain rate is accelerating toward failure.

Two hyperparameters control the transition detection:

- `k1` — secondary creep is considered to begin at the first of `k1` consecutive plateaus where the point-count does not increase. Primary creep ends at the plateau immediately before this run.
- `k2` — tertiary creep is considered to begin after `k2` consecutive plateaus where the point-count is *strictly* decreasing.

Both are tuned manually via visual inspection and applied consistently across all tests. The **final plateau of every test is excluded** from classification, since it ends because the experiment was stopped, not because of a genuine physical transition — including it would bias detection near the end of every curve.

*(Implemented in `eda/stage_classification.py`.)*

### 1.2 Initial analysis

Two summary statistics characterize each test's creep behavior:

- **ϵ̃₀** — the first valid strain recording (the instantaneous elastic response).
- **ϵ̂̇ₛₛ** — the estimated steady-state (secondary creep) strain rate, computed as the slope of a linear fit to all data classified as secondary creep.

The factors expected to influence these statistics — print quality, applied stress, specimen age, and temperature — are investigated through **exploratory visualization only** (histograms, bar charts, pair plots, scatter plots, bubble plots), rather than formal inferential statistics (e.g. ANCOVA). Given the sample size (24 specimens, two failed) and project timeline, this is a deliberate, explicitly-justified downgrade from a confirmatory to an exploratory analysis. Relationships that show a visually clear effect are carried forward into the modeling stage; the rest are documented as inconclusive given the available data.

*(Implemented in `eda/statistics.py`, visualized via `scripts/exploratory/`.)*

---

## 2. Constitutive Creep Model

### 2.1 The TLV model

The **TLV (Three-Layer Viscoelastic) model** with a **Norton-Hoff creep law** is used to model the material's viscoelastic response. This model has been shown to accurately capture viscoelastic effects in FFF-printed PLA (Tanner et al. 2026), and — practically — it's already implemented in Abaqus, making parameters fit here directly reusable in future FEA work.

Every parameter in the governing equations is modeled as a function of temperature (converted to Kelvin so all fitted parameters carry consistent SI units):

$$
\dot{\sigma}^{ep} = A(T)\,(\sigma - \sigma^{ep})^{n(T)}\, t^{m(T)} \left(\frac{1}{E_e(T)} + \frac{\Theta(\sigma^{ep} - Y_0(T))}{H(T)} + \frac{1}{E_v(T)}\right),
$$

$$
\dot{\epsilon} = \left(\frac{1}{E_e(T)} + \frac{\Theta(\sigma^{ep} - Y_0(T))}{H(T)}\right)\dot{\sigma}^{ep}.
$$

### 2.2 Model simplifications

Two properties of the experimental setup justify simplifying the equations above:

1. **Constant, positive applied stress** for the duration of each test.
2. **Yield stress exceeds 30 MPa** for FFF PLA printed at 0°/45° to the filament direction (Luo et al. 2022), and all applied stresses in this study stay below that. This means **yield hardening effects can be ignored entirely** — the `Θ(σ^ep − Y₀(T))/H(T)` terms drop out.

Applying this gives the simplified governing equations actually used for fitting:

$$
\dot{\sigma}^{ep} = \left(\frac{1}{E_e(T)} + \frac{1}{E_v(T)}\right)^{-1}\left[\frac{\sigma^{ep}}{E_e(T)^2}\frac{dE_e}{dT}\dot{T} - \frac{\sigma - \sigma^{ep}}{E_v(T)^2}\frac{dE_v}{dT}\dot{T} + A(T)(\sigma - \sigma^{ep})^{n(T)} t^{m(T)}\right],
$$

$$
\epsilon = \frac{\sigma^{ep}}{E_e(T)}.
$$

All temperature variation is a natural, mild fluctuation in ambient temperature (up to ~3.5°C within a single test) — small enough that every temperature-dependent material property is modeled as **linear in temperature**, parametrized between values measured at 20°C and 30°C:

$$
x(T) = x_{20} + \frac{x_{30} - x_{20}}{T_{30} - T_{20}}(T - T_{20}).
$$

This means the quantities actually estimated by the fitting routine are the material properties *at 20°C and at 30°C* for each parameter (e.g. `A₂₀`, `A₃₀`, `Ee₂₀`, `Ee₃₀`, ...) — the linear interpolation reconstructs the full temperature-dependent function from those two anchor values.

**Initial condition:** assuming the instantaneous initial strain is purely elastic,

$$
\sigma^{ep}_0 = (1 - f(T_0))\,\sigma, \qquad f(T) = \frac{E_v(T)}{E_v(T) + E_e(T)}.
$$

This settles the earlier open question about how ε₀(σ,T) is used: **σ^ep₀ is a direct model input** — the initial condition needed to start the ODE integration — not a separate validation target fit independently.

### 2.3 Numerical solver

Equation 1.2a has no closed-form solution under non-isothermal conditions, so it's integrated numerically, following an approach similar to Kossa & Horváth (2021). The measured temperature history is linearly interpolated to build a continuous `T(t)`.

The equation is solved using **implicit midpoint integration**: writing the ODE in the general form `σ̇^ep = f(t, σ^ep)`, each time step produces a nonlinear residual equation

$$
R(\sigma^{ep}_{n+1}) = \sigma^{ep}_{n+1} - \sigma^{ep}_n - \Delta t \cdot f\!\left(t_{n+1/2},\ \sigma^{ep}_{n+1/2}\right),
$$

which is solved at every time step via **Newton-Raphson iteration**, updating the estimate as

$$
\sigma^{ep}_{n+1,k+1} = \sigma^{ep}_{n+1,k} - \frac{R(\sigma^{ep}_{n+1,k})}{\partial R / \partial \sigma^{ep}_{n+1}},
$$

until the residual magnitude falls below 10⁻⁸ MPa. This is implemented directly using SciPy, stepping sequentially through each test's time series. Every call to the solver requires the model parameters, applied stress, time points, and temperature measurements for that specific test.

> **Note:** this replaces an earlier planned approach using `scipy.integrate.solve_ivp` with an explicit method — the coupled elastic/viscous system in Eq. 1.2a needs the implicit-midpoint + Newton-Raphson scheme described above, since it isn't a simple decoupled ODE.

---

## 3. Parameter Estimation

### 3.1 Loss function

The solver produces a predicted strain ϵ̂ᵢ for each recorded (time, stress, temperature) triple, compared against the measured strain ϵ̃ᵢ. Consistent with prior polymer-creep literature, a **mean squared error (MSE)** loss is used:

$$
L = (\hat{\epsilon}_i - \tilde{\epsilon}_i)^2.
$$

This loss is smooth, and its quadratic penalty produces consistent, well-behaved fits.

### 3.2 Optimization algorithm

A **two-stage global-then-local** optimization strategy is used to find a robust minimum:

1. **Differential Evolution (DE)** — a global, population-based search over the full bounded parameter domain (see below). This avoids getting stuck in a poor local minimum, which is a real risk given the number of parameters and their widely different physical scales.
2. **Levenberg-Marquardt (LM)** — the best parameter set found by DE is used as the initial guess for LM, which then converges precisely to the nearest exact minimum of the loss function.

> **Note:** this two-stage DE→LM scheme is the current fitting strategy, replacing an earlier plan based on penalized regression (LASSO/ridge) with leave-one-specimen-out cross-validation for parameter selection. That approach doesn't appear in the current chapter and should be treated as superseded.

### 3.3 Parameter bounds & scaling

Because TLV model parameters span vastly different physical scales (e.g. `A` ~ 10⁻⁵, versus elastic moduli in the hundreds or thousands of MPa), both **bounding** and **scaling** are used together:

- Each parameter's physically-valid range (Table 1.1 in the thesis) is linearly transformed to `[0, 1]` before being passed to DE. The inverse transform recovers physical units afterward.
- Before the LM stage, each parameter is additionally rescaled so all parameters sit at a comparable order of magnitude — necessary because LM's convergence behavior is sensitive to scale mismatches between parameters. The final LM result is un-scaled to report physical parameter estimates.

Parameter bounds are chosen to be physically plausible (e.g. `Ee`, `Ev` bounded above by `max(σ/ϵ)`) and are informed by Tanner et al. (2026) where available, but are kept intentionally wide so the true global minimum isn't excluded by an overly tight bound.

### 3.4 Data trimming & partitioning

Since the TLV model only predicts primary and secondary creep, **tertiary creep data is trimmed** from each curve before fitting, using the same stage-classification boundaries from Section 1.1.

Data is also **partitioned by print quality** — Standard and High quality prints are fit **separately**, producing two independent sets of TLV parameters rather than one pooled model. This avoids conflating a print-quality effect with the material behavior the model is meant to capture, at the cost of halving the effective sample size available to each fit.

---

## References

- Kossa, A. & Horváth, R. (2021).
- Luo et al. (2022) — yield stress of FFF PLA at 0°/45° filament orientation.
- Tanner et al. (2026) — TLV model validation for FFF PLA; prior parameter bounds reference.
- Virtanen et al. (2020) — SciPy.