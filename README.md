# creep_model

Python package for modeling **creep behavior in FFF-printed PLA specimens**, developed as part of a final-year engineering thesis (supervised by Tanner) investigating how print quality, applied stress, specimen age, and ambient temperature fluctuations affect long-term deformation.

This repository contains the full data pipeline: parsing raw experimental Excel workbooks, exploratory data analysis, constitutive model fitting, and visualization.

> **Status:** active thesis project. APIs, especially in `modeling/`, are still evolving as the constitutive model implementation catches up to the current theory chapter (see [Implementation Status](#implementation-status) below).

---

## What this package does

Twenty-four PLA specimens (two print-quality groups, three nominal stress levels, varying specimen ages) were creep-tested under mild, naturally-fluctuating ambient temperature. This package:

1. **Parses** the raw multi-sheet Excel workbook into a clean, typed domain model (`io/`, `domain.py`).
2. **Classifies** each creep curve into primary / secondary / tertiary stages using a quantization-based plateau-detection method (`eda/stage_classification.py`).
3. **Explores** the data visually — trellis plots, bubble charts, pair plots — to identify which factors (print quality, stress, age, temperature) actually influence creep behavior (`eda/statistics.py`).
4. **Fits** a physically-motivated constitutive model (the TLV model with a Norton-Hoff creep law) to the trimmed primary/secondary creep data, with model parameters expressed as functions of temperature (`modeling/`).
5. **Visualizes** raw data, fitted curves, and residuals for both the thesis document and general model diagnostics (`viz/`).

Full theoretical background — the creep-stage classification rule, the constitutive equations, the numerical solver, and the parameter estimation strategy — lives in **[`docs/methodology.md`](docs/methodology.md)**, which mirrors Chapter 1 of the thesis.

---

## Repository layout

```
src/creep_model/
├── config.py           # centralized paths, column names, physical constants
├── domain.py            # CreepTest / CreepExperiment — the core data model
├── io/                  # Excel → domain model parsing
├── modeling/             # constitutive model, solver, parameter fitting
├── eda/                  # creep-stage classification & summary statistics
└── viz/                  # matplotlib / Plotly plotting utilities

scripts/                 # exploratory & one-off analysis scripts (not part of the package)
tests/                   # pytest suite, mirrors src/creep_model/
docs/                    # methodology write-up and exported figures
data/raw/                # raw Excel workbook (not tracked by git)
```

---

## Installation

Requires Python ≥ 3.11.

```bash
git clone <repo-url>
cd creep_model
pip install -e ".[dev]"
```

## Quick start

```python
from pathlib import Path
from creep_model.io.parser import ExcelCreepParser
from creep_model.modeling.assembler import DataAssembler

parser = ExcelCreepParser(Path("data/raw/CreepData.xlsx"))
experiment = parser.load_experiment()

# Summary statistics across all 24 tests
df_summary = DataAssembler.get_summary_dataframe(experiment)
print(df_summary.head())
```

See `scripts/exploratory/` for runnable examples of stage classification, correlation analysis, and 3D creep-curve visualization.

## Running the tests

```bash
pytest
```

Coverage is enforced at 90% (`pyproject.toml`); see `tests/` for the current suite.

---

## Implementation status

The theory chapter (`docs/methodology.md`) specifies the **TLV model with a Norton-Hoff creep law**, solved via implicit midpoint integration + Newton-Raphson iteration, with parameters estimated using a two-stage differential evolution → Levenberg-Marquardt optimization.

The current `modeling/empirical.py` module implements a family of **simpler exploratory models** (Findley and modified-Findley laws, a pooled global MSE model) that were used to validate the data pipeline and get early fits — these are stepping stones, not the final thesis model. The TLV model, its residual-equation solver, and the DE→LM fitting routine still need to be implemented as a new `modeling/tlv/` submodule. This README will be updated once that lands.

## License

See [`LICENSE`](LICENSE).