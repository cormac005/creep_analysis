# Methodology

This document summarizes the theoretical approach implemented by this package, mirroring Chapter 3 of the thesis. It's intended as the "why" companion to the code — the README explains what to run, this explains what it's doing and why each choice was made.

---

## 1. Exploratory Data Analysis & Pre-Processing

### 1.1 Creep stage classification

Strain data is recorded at fixed time intervals but is **quantized** — the sensor can only report discrete strain levels, so the raw signal looks like a staircase of flat "plateaus" rather than a smooth curve. This package turns that quantization from a nuisance into a diagnostic tool.

Because strain increases monotonically with time under constant applied stress, the *number of consecutive data points spent on each plateau* is directly related to the local strain rate at that point in the test:

- **Primary creep**: time spent on each successive plateau is *increasing* → strain rate is decreasing.
- **Secondary creep**: time spent on plateaus is roughly *constant* → strain rate has stabilized.
- **Tertiary creep**: time spent on plateaus is *decreasing* → strain rate is accelerating toward failure.

Two hyperparameters control transition detection in `classify_stages(test, k1, k2)`:

- `k1` — secondary creep is considered to begin at the first of `k1` consecutive plateaus where the point-count does not increase. Primary creep ends at the plateau immediately before this run.
- `k2` — tertiary creep is considered to begin after `k2` consecutive plateaus where the point-count is *strictly* decreasing relative to the primary-end plateau length.

Both are configured via `config.py` and applied consistently across all tests. The **final plateau of every test is excluded** from classification, since it ends because the experiment was stopped, not because of a genuine physical transition — including it would bias detection near the end of every curve.

For exploratory visualization (`plot_creep_stage_boundaries`), stage boundaries are evaluated on the **untrimmed raw time series** parsed directly from `CreepData.xlsx` via `ExcelCreepParser`. This ensures that all three stages (Primary, Secondary, and Tertiary) and their rate fits remain fully visible on plots for specimens where failure occurred, whereas downstream material parameter fitting consumes the trimmed series.

*(Implemented in `eda/stage_classification.py` and `viz/eda_plots.py`.)*

### 1.2 Initial analysis

Two summary statistics characterize each test's creep behavior:

- **$\tilde{\epsilon}_0$** — the first valid strain recording (the instantaneous elastic response).
- **$\dot{\epsilon}_{ss}$** — the estimated steady-state (secondary creep) strain rate, computed as the slope of a linear fit to all data classified as secondary creep.

The factors expected to influence these statistics — print quality, applied stress, specimen age, and temperature — are investigated through **exploratory visualization only** (histograms, bar charts, pair plots, scatter plots, bubble plots), rather than formal inferential statistics (e.g. ANCOVA). Given the sample size (24 specimens, two failed) and project timeline, this is a deliberate, explicitly-justified downgrade from a confirmatory to an exploratory analysis. Relationships that show a visually clear effect are carried forward into the modeling stage; the rest are documented as inconclusive given the available data.

*(Implemented in `eda/statistics.py`, visualized via `scripts/pipeline/05_generate_eda_plots_and_tables.py`.)*

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
2. **Yield stress exceeds 30 MPa** for FFF PLA printed at 0°/45° to the filament direction (Luo et al. 2022), and all applied stresses in this study stay below that. This means **yield hardening effects can be ignored entirely** — the $\frac{\Theta(\sigma^{ep} - Y_0(T))}{H(T)}$ terms drop out.

Applying this gives the simplified governing equations actually used for fitting:

$$
\dot{\sigma}^{ep} = \left(\frac{1}{E_e(T)} + \frac{1}{E_v(T)}\right)^{-1}\left[\frac{\sigma^{ep}}{E_e(T)^2}\frac{dE_e}{dT}\dot{T} - \frac{\sigma - \sigma^{ep}}{E_v(T)^2}\frac{dE_v}{dT}\dot{T} + A(T)(\sigma - \sigma^{ep})^{n(T)} t^{m(T)}\right],
$$

$$
\epsilon = \frac{\sigma^{ep}}{E_e(T)}.
$$

All temperature variation is a natural, mild fluctuation in ambient temperature (up to ~3.5°C within a single test) — small enough that every temperature-dependent material property is modeled as **linear in temperature**, parametrized between values measured at anchor constants $T_{20} = 293.15\text{ K}$ (`T_20_KELVIN`) and $T_{30} = 303.15\text{ K}$ (`T_30_KELVIN`) defined centrally in `parameters.py`:

$$
x(T) = x_{20} + \frac{x_{30} - x_{20}}{T_{30} - T_{20}}(T - T_{20}).
$$

This means the quantities actually estimated by the fitting routine are the material properties *at 20°C and at 30°C* for each parameter (e.g. `A₂₀`, `A₃₀`, `Ee₂₀`, `Ee₃₀`, ...) — the linear interpolation reconstructs the full temperature-dependent function from those two anchor values.

**Initial condition:** assuming the instantaneous initial strain is purely elastic,

$$
\sigma^{ep}_0 = (1 - f(T_0))\,\sigma, \qquad f(T) = \frac{E_v(T)}{E_v(T) + E_e(T)}.
$$

Alternatively, when `use_measured_initial_condition=True`, $\sigma^{ep}_0 = \epsilon_{measured,0} \cdot E_e(T_0)$. **$\sigma^{ep}_0$ is a direct model input** — the initial condition needed to start the ODE integration — not a separate validation target fit independently.

### 2.3 Numerical solver architecture & Numba JIT compilation

Equation 1.2a has no closed-form solution under non-isothermal conditions and must be integrated numerically. To achieve the performance required for global optimization across dozens of specimens, the numerical solver combines **pre-cached array profiles** with a **Numba JIT-compiled C-kernel**:

1. **Pre-calculated Data Profile (`prepare_test_data`)**: Before optimization loops begin, time series, midpoint times $t_{\text{mid}}$, midpoint temperatures $T_{\text{mid}}$, thermal rates $\dot{T}_{\text{mid}}$, and time step sizes $\Delta t$ are interpolated once and cached in a frozen `PreparedTest` object. This eliminates redundant spline and polynomial interpolation overhead during repeated objective evaluations.
2. **Implicit Midpoint Integration**: Writing the ODE as $\dot{\sigma}^{ep} = f(t, \sigma^{ep})$, each time step forms a non-linear residual equation:

$$
R(\sigma^{ep}_{n+1}) = \sigma^{ep}_{n+1} - \sigma^{ep}_n - \Delta t \cdot f\!\left(t_{n+1/2},\ \sigma^{ep}_{n+1/2}\right) = 0.
$$

3. **Newton-Raphson Kernel (`_solve_tlv_numba_kernel`)**: The residual $R(\sigma^{ep}_{n+1})$ is solved iteratively at each time step using Newton-Raphson updates:

$$
\sigma^{ep}_{n+1,k+1} = \sigma^{ep}_{n+1,k} - \frac{R(\sigma^{ep}_{n+1,k})}{\partial R / \partial \sigma^{ep}_{n+1}},
$$

until $|R| < 10^{-8}\text{ MPa}$ within `max_iter` iterations. The entire stepping loop is compiled to native machine code via `@nb.njit(fastmath=True, error_model="numpy")`, enabling zero-overhead execution.

#### Smooth Effective Stress Saturation & Exact Jacobian Derivation

As the material relaxes, $\sigma^{ep} \to \sigma$, and effective stress $x = \sigma - \sigma^{ep}$ approaches zero. For shear-thinning or non-linear creep exponents ($n < 1$), the naive term $x^n$ has a derivative $\frac{d}{dx} x^n = n x^{n-1}$ that **diverges to $+\infty$ as $x \to 0^+$**. In naive implementations, clamping $x \le 0$ to zero creates a discontinuous "cliff" in the Newton-Raphson Jacobian $R'$, causing step-size explosions, solver stalling, or artificial downward bias in fitted $n$ values.

To ensure $C^\infty$ smoothness and exact gradient consistency, effective stress is saturated using a **Softplus function**:

$$
S(x) = \frac{1}{\beta} \ln\left(1 + e^{\beta x}\right) \quad (\beta = 50.0).
$$

The exact analytical derivative of Softplus is the logistic sigmoid function:

$$
S'(x) = \frac{d}{dx} S(x) = \frac{1}{1 + e^{-\beta x}}.
$$

Using the chain rule, the exact derivative of the Norton-Hoff term with respect to $\sigma^{ep}_{n+1}$ is evaluated as:

$$
\frac{\partial}{\partial \sigma^{ep}_{n+1}} \left[ A(T) \cdot S(\sigma - \sigma^{ep}_{\text{mid}})^n \cdot t_{\text{mid}}^m \right] = -A(T) \cdot n \cdot \left(S(\sigma - \sigma^{ep}_{\text{mid}}) + \epsilon\right)^{n-1} \cdot t_{\text{mid}}^m \cdot S'(\sigma - \sigma^{ep}_{\text{mid}}),
$$

where $\epsilon = 10^{-12}$ acts as a base regularization. This exact matching between $R$ and $R'$ guarantees quadratic Newton-Raphson convergence, prevents solver breakdown near saturation, and eliminates numerical bias during parameter optimization.

*(Implemented in `modelling/tlv/solver.py` and `modelling/tlv/residual.py`.)*

---

## 3. Parameter Estimation

### 3.1 Loss function

The solver produces a predicted strain $\hat{\epsilon}_i$ for each recorded (time, stress, temperature) triple, compared against the measured strain $\tilde{\epsilon}_i$. Consistent with prior polymer-creep literature, a **mean squared error (MSE)** loss is used:

$$
L = \frac{1}{N} \sum_{i=1}^N (\hat{\epsilon}_i - \tilde{\epsilon}_i)^2.
$$

This loss is smooth, and its quadratic penalty produces consistent, well-behaved fits.

### 3.2 Optimization algorithm

A **two-stage global-then-local** optimization strategy is used to find a robust minimum:

1. **Differential Evolution (DE)** — a global, population-based search over the full bounded parameter domain. This avoids getting stuck in a poor local minimum, which is a real risk given the number of parameters and their widely different physical scales.
2. **Levenberg-Marquardt (LM)** — the best parameter set found by DE is used as the initial guess for LM, which then converges precisely to the nearest exact minimum of the loss function.

> **Note:** this two-stage DE→LM scheme is the current fitting strategy, replacing an earlier plan based on penalized regression (LASSO/ridge) with leave-one-specimen-out cross-validation for parameter selection. That approach doesn't appear in the current chapter and should be treated as superseded.

### 3.3 Parameter bounds & scaling

Because TLV model parameters span vastly different physical scales (e.g. `A` $\sim 10^{-5}$, versus elastic moduli in the hundreds or thousands of MPa), both **bounding** and **scaling** are used together:

- Each parameter's physically-valid range (Table 1.1 in the thesis) is linearly transformed to $[0, 1]$ before being passed to DE. The inverse transform recovers physical units afterward.
- Before the LM stage, each parameter is additionally rescaled so all parameters sit at a comparable order of magnitude — necessary because LM's convergence behavior is sensitive to scale mismatches between parameters. The final LM result is un-scaled to report physical parameter estimates.

Parameter bounds are chosen to be physically plausible (e.g. `Ee`, `Ev` bounded above by $\max(\sigma/\epsilon)$) and are informed by Tanner et al. (2026) where available, but are kept intentionally wide so the true global minimum isn't excluded by an overly tight bound.

### 3.4 Data trimming & partitioning

Since the TLV model only predicts primary and secondary creep, **tertiary creep data is trimmed** from each curve before fitting, using the same stage-classification boundaries from Section 1.1.

Data is also **partitioned by print quality** — Standard and High quality prints are fit **separately**, producing two independent sets of TLV parameters rather than one pooled model. This avoids conflating a print-quality effect with the material behavior the model is meant to capture, at the cost of halving the effective sample size available to each fit.

---

## References

- Kossa, A. & Horváth, R. (2021).
- Luo et al. (2022) — yield stress of FFF PLA at 0°/45° filament orientation.
- Tanner et al. (2026) — TLV model validation for FFF PLA; prior parameter bounds reference.
- Virtanen et al. (2020) — SciPy.