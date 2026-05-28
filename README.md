# AwesomePKF

This repository contains a set of programs illustrating the **Pairwise Kalman Filter (PKF)**, a generalization of the classical Kalman Filter, extended to non-linear models. It includes several variants of non-linear filters:

- **Extended Pairwise Kalman Filter (EPKF)**
- **Unscented Pairwise Kalman Filter (UPKF)**, with multiple variants depending on the choice of sigma points
- **Pairwise Particle Filter (PPF)**

and **pairwise Kalman smoothers** for offline post-processing:

- **Linear Pairwise Kalman Smoother (PKS)** — RTS-style backward pass on the joint `(X, Y)` Markov chain.
- **Extended Pairwise Kalman Smoother (EPKS)** — same recursion with the per-step Jacobian replacing the constant transition matrix.
- **Unscented Pairwise Kalman Smoother (UPKS)** — sigma-point cross-covariance in the backward pass; supports the same sigma-point sets as the UPKF (`wan2000`, `cpkf`, `lerner2002`, `ito2000`).
- **Unscented Kalman Smoother (UKS)** — classical (non-pairwise) sigma-point smoother for FxHx models with Markov-in-X assumption; gain `(dim_x, dim_x)`.
- **Pairwise Particle Smoother (PPS)** — FFBSm (Forward Filtering, Backward Smoothing) on top of the PPF; reweights the forward particle cloud via a backward weight recursion. No closed-form gain.

---

## Table of Contents

- [AwesomePKF](#awesomepkf)
    - [Table of Contents](#table-of-contents)
    - [Installation](#installation)
    - [Models and Simulations](#models-and-simulations)
    - [Filters](#filters)
        - [Pairwise Kalman Filter (PKF)](#pairwise-kalman-filter-pkf)
        - [Extended Pairwise Kalman Filter (EPKF)](#extended-pairwise-kalman-filter-epkf)
        - [Unscented Pairwise Kalman Filter (UPKF)](#unscented-pairwise-kalman-filter-upkf)
        - [Pairwise Particle Filter (PPF)](#pairwise-particle-filter-ppf)
    - [Smoothers](#smoothers)
        - [Linear Pairwise Kalman Smoother (PKS)](#linear-pairwise-kalman-smoother-pks)
        - [Extended Pairwise Kalman Smoother (EPKS)](#extended-pairwise-kalman-smoother-epks)
        - [Unscented Pairwise Kalman Smoother (UPKS)](#unscented-pairwise-kalman-smoother-upks)
        - [Unscented Kalman Smoother (UKS)](#unscented-kalman-smoother-uks)
        - [Pairwise Particle Smoother (PPS)](#pairwise-particle-smoother-pps)
    - [Tutorials](#tutorials)
    - [Usage Examples](#usage-examples)
        - [Simulate Linear Data and Filter with PKF](#simulate-linear-data-and-filter-with-pkf)
        - [Simulate Non-Linear Data and Filter with EPKF, UPKF and PPF](#simulate-non-linear-data-and-filter-with-epkf-upkf-and-ppf)
    - [Folders structure](#folders-structure)

---

## Installation

### From PyPI (recommended)

```bash
pip install awesomepkf
```

### From source

```bash
git clone https://github.com/sderrode/awesomepkf.git
cd awesomepkf
pip install .
```

### Development install

```bash
pip install -e ".[dev]"
```

### Requirements

- Python >= 3.10
- numpy, scipy, matplotlib, pandas, rich, sympy

---

## Quick Start

```python
from prg.classes.linear_pkf import Linear_PKF
from prg.models.linear import ModelFactoryLinear

model = ModelFactoryLinear.create("model_x1_y1_AQ_pairwise")
pkf = Linear_PKF(model)
# ... run the filter step by step
```

Or use the CLI entry points installed with the package:

```bash
awesomepkf-simulate --N 2000 --linear-model-name "model_x1_y1_AQ_pairwise" --data-filename "testL.csv" --s-key 303
awesomepkf-pkf      --linear-model-name "model_x1_y1_AQ_pairwise" --data-filename "testL.csv" --plot
```

---

## Tutorials

Interactive Jupyter notebooks are available in the [`notebooks/`](notebooks/) directory:

| # | Notebook | Description |
|---|----------|-------------|
| 01 | [`tutorial_01_getting_started.ipynb`](notebooks/tutorial_01_getting_started.ipynb) | Introduction to the PKF framework: linear models, running the filter, visualizing estimates, error metrics (MSE, NEES, NIS), comparing PKF / EPKF / UPKF |
| 02 | [`tutorial_02_nonlinear_models.ipynb`](notebooks/tutorial_02_nonlinear_models.ipynb) | Nonlinear models: EPKF, UPKF, PPF and PF — classic vs pairwise, sigma-point sets, particle count impact, filter comparison |
| 03 | [`tutorial_03_sigma_points.ipynb`](notebooks/tutorial_03_sigma_points.ipynb) | Sigma-point sets for the UPKF: wan2000, cpkf, lerner2002, ito2000 — impact on estimation accuracy |
| 04 | [`tutorial_04_particle_filters.ipynb`](notebooks/tutorial_04_particle_filters.ipynb) | Particle filters (PPF and PF): tuning the number of particles, resampling, comparison with EPKF/UPKF |
| 05 | [`tutorial_05_new_model_lotkavolterra.ipynb`](notebooks/tutorial_05_new_model_lotkavolterra.ipynb) | How to add a new nonlinear pairwise model: Lotka-Volterra prey-predator (dim_x=1, dim_y=1), augmented version, filtering with EPKF/UPKF/PPF |
| 06 | [`tutorial_06_filter_runner_and_config.ipynb`](notebooks/tutorial_06_filter_runner_and_config.ipynb) | High-level orchestration with `FilterRunner` and `RunOptions`; parameter sweeps via `model_kwargs`; saving and replaying experiments through TOML session configs |
| 07 | [`tutorial_07_smoothers.ipynb`](notebooks/tutorial_07_smoothers.ipynb) | The 5 smoothers (`Linear_PKS`, `NonLinear_EPKS`/`UPKS`/`UKS`/`PPS`) — RTS-style backward recursion and FFBSm; ±2σ envelope shrinkage, Joseph form, Monte-Carlo convergence of PPS to PKS, decision rule for choosing among the five |

---

## Models and Simulations

The repository provides a program called **run_simulator.py** to simulate data according to **linear and non-linear models**.

---

## Filters

Each filter has two types of programs:

1. Simulate data **and filter it directly**  
2. Filter data **from a previously saved file**  

### Pairwise Kalman Filter (PKF)

- **run_linear_pkf.py** – filter linear data either from simulated data or from a previously saved file (e.g., generated with `run_simulator.py`)  

### Extended Pairwise Kalman Filter (EPKF)

- **run_nonlinear_epkf.py** – filter non-linear data either from simulated data or from a previously saved file (e.g., generated with `run_simulator.py`)  

### Unscented Pairwise Kalman Filter (UPKF)

- **run_nonlinear_upkf.py** – filter non-linear data either from simulated data or from a previously saved file (e.g., generated with `run_simulator.py`)  

### Pairwise Particle Filter (PPF)

- **run_nonlinear_ppf.py** – filter non-linear data either from simulated data or from a previously saved file (e.g., generated with `run_simulator.py`)  

---

## Smoothers

Smoothers are **two-pass, offline** estimators that condition on the *entire* observation sequence `y_{1:N}`. They produce posterior means and covariances `p(X_n | y_{1:N})` that are at least as good (in PSD sense) as the corresponding forward filter outputs `p(X_n | y_{1:n})`.

### Linear Pairwise Kalman Smoother (PKS)

The linear PKS runs the [PKF forward](#pairwise-kalman-filter-pkf), then a backward Rauch-Tung-Striebel recursion at the **joint** `(X, Y)` level. The pairwise model is Markov in `Z = (X, Y)` (not in `X` alone), so the smoothing gain `G_n` has shape `(dim_x, dim_x + dim_y)`. Equivalently, the linear PKS is the classical RTS smoother applied to the augmented state `Z' = (X, Y)` with degenerate observation `Y_n = (0, I) Z'_n` (`R^aug = 0`).

```python
from prg.classes.linear_pks import Linear_PKS
from prg.classes.param_linear import ParamLinear
from prg.models.linear import ModelFactoryLinear

model  = ModelFactoryLinear.create("model_x1_y1_AQ_pairwise")
params = model.get_params().copy()
dim_x  = params.pop("dim_x");  dim_y = params.pop("dim_y")
param  = ParamLinear(0, dim_x, dim_y, **params)

pks = Linear_PKS(param, sKey=42, joseph=False)
# joseph=True selects the explicitly PSD-preserving Joseph form
# (mathematically equivalent at the optimal gain, useful for ill-conditioned cases).

results = pks.process_N_data_smoother(N=500)
# each tuple: (k, x_true, y_obs, X_predict, X_update, X_smooth)
```

Implementation: [`prg/classes/linear_pks.py`](prg/classes/linear_pks.py). Tests: [`prg/tests/test_linear_pks.py`](prg/tests/test_linear_pks.py) (40 tests, including PSD shrinkage, Joseph equivalence, augmented-state RTS equivalence, and full exception/logging coverage).

### Extended Pairwise Kalman Smoother (EPKS)

The EPKS extends the linear PKS to non-linear pairwise models via first-order linearisation. The forward pass is the standard [EPKF](#extended-pairwise-kalman-filter-epkf); the backward pass uses the *per-step* Jacobian `F_{n+1}` (evaluated at the filtered state) in place of the constant `A` matrix. PSD shrinkage of the linearised covariance is preserved; on average the MSE also shrinks, but the linearisation bias breaks the per-trajectory guarantee of the linear case.

```python
from prg.classes.nonlinear_epks import NonLinear_EPKS
from prg.classes.param_nonlinear import ParamNonLinear
from prg.models.nonlinear import ModelFactoryNonLinear

model  = ModelFactoryNonLinear.create("model_x2_y1_pairwise")
params = model.get_params().copy()
dim_x  = params.pop("dim_x");  dim_y = params.pop("dim_y")
param  = ParamNonLinear(0, dim_x, dim_y, **params)

epks = NonLinear_EPKS(param, sKey=42, joseph=False)
results = epks.process_N_data_smoother(N=300)
```

Implementation: [`prg/classes/nonlinear_epks.py`](prg/classes/nonlinear_epks.py). Tests: [`prg/tests/test_nonlinear_epks.py`](prg/tests/test_nonlinear_epks.py) (25 tests). Note: not suitable for augmented models (rank-deficient predicted covariance fails the backward Cholesky).

### Unscented Pairwise Kalman Smoother (UPKS)

The UPKS replaces the first-order linearisation of the EPKS with **sigma-point propagation**: the cross-covariance `Cov(X_n, Z_{n+1} | y_{1:n})` and the predicted joint covariance `P^{ZZ}_{n+1|n}` are estimated from sigma points regenerated at the filtered state. Supports the same sigma-point sets as the UPKF (`wan2000`, `cpkf`, `lerner2002`, `ito2000`), selected via the `sigmaSet` constructor argument.

```python
from prg.classes.nonlinear_upks import NonLinear_UPKS
from prg.classes.param_nonlinear import ParamNonLinear
from prg.models.nonlinear import ModelFactoryNonLinear

model  = ModelFactoryNonLinear.create("model_x2_y1_pairwise")
params = model.get_params().copy()
dim_x  = params.pop("dim_x");  dim_y = params.pop("dim_y")
param  = ParamNonLinear(0, dim_x, dim_y, **params)

upks = NonLinear_UPKS(param, sigmaSet="wan2000", sKey=42, joseph=False)
results = upks.process_N_data_smoother(N=300)
```

Implementation: [`prg/classes/nonlinear_upks.py`](prg/classes/nonlinear_upks.py). Tests: [`prg/tests/test_nonlinear_upks.py`](prg/tests/test_nonlinear_upks.py) (29 tests, parametrised over all sigma-point sets). Same caveats as the EPKS regarding augmented models.

### Unscented Kalman Smoother (UKS)

The UKS is the classical (non-pairwise) sigma-point smoother — equivalent to the UPKS when applied to a model that is Markov in X alone (FxHx structure). The gain is `(dim_x, dim_x)` (vs `(dim_x, dim_x + dim_y)` for the pairwise variants), and the sigma-point dimension is `dim_x` only. The smoother refuses pairwise models at construction time (FilterError).

```python
from prg.classes.nonlinear_uks import NonLinear_UKS
from prg.classes.param_nonlinear import ParamNonLinear
from prg.models.nonlinear import ModelFactoryNonLinear

model  = ModelFactoryNonLinear.create("model_x1_y1_Sinus_classic")
params = model.get_params().copy()
dim_x  = params.pop("dim_x");  dim_y = params.pop("dim_y")
param  = ParamNonLinear(0, dim_x, dim_y, **params)

uks = NonLinear_UKS(param, sigmaSet="wan2000", sKey=42, joseph=False)
results = uks.process_N_data_smoother(N=300)
```

Implementation: [`prg/classes/nonlinear_uks.py`](prg/classes/nonlinear_uks.py). Tests: [`prg/tests/test_nonlinear_uks.py`](prg/tests/test_nonlinear_uks.py) (29 tests).

### Pairwise Particle Smoother (PPS)

The PPS implements **FFBSm** (Forward Filtering, Backward Smoothing) on top of the PPF. The forward pass runs the standard PPF with particle clouds stored at every step; the backward pass reweights those forward particles via:

```
ŵ_{i,n} = w_{i,n} · Σ_j ŵ_{j,n+1} · p(ξ_{j,n+1} | ξ_{i,n}, y_n) / Σ_l w_{l,n}·p(ξ_{j,n+1} | ξ_{l,n}, y_n)
```

The smoothed mean and covariance are weighted statistics of the forward particle cloud with the smoothed weights. Complexity is O(N·n_p²) per smoother run (vs O(N·n_p) for the forward).

```python
from prg.classes.nonlinear_pps import NonLinear_PPS
from prg.classes.param_nonlinear import ParamNonLinear
from prg.models.nonlinear import ModelFactoryNonLinear

model  = ModelFactoryNonLinear.create("model_x2_y1_pairwise")
params = model.get_params().copy()
dim_x  = params.pop("dim_x");  dim_y = params.pop("dim_y")
param  = ParamNonLinear(0, dim_x, dim_y, **params)

pps = NonLinear_PPS(param, n_particles=300, sKey=42)
results = pps.process_N_data_smoother(N=200)
```

Implementation: [`prg/classes/nonlinear_pps.py`](prg/classes/nonlinear_pps.py). Tests: [`prg/tests/test_nonlinear_pps.py`](prg/tests/test_nonlinear_pps.py) (19 tests, including a Monte-Carlo convergence test that verifies the PPS converges to the exact Linear_PKS as `n_particles` grows on a linear-Gaussian pairwise model).

---

## Paper Reproducibility Scripts

The following scripts reproduce all figures and tables from the article
*"Non-linear extensions to Gaussian pairwise Kalman filter"*.
Each script can be run independently from the repository root.

### Section 4 — Simulation Results

| Script | Figures generated |
|--------|------------------|
| `run_paper_section4.py` | `epkf_observations_x1_y1_Retroactions.png`, `epkf_x1_y1_Retroactions.png`, `upkf_x1_y1_Retroactions.png`, `ppf_x1_y1_Retroactions.png` + Tables 1 & 2 |
| `run_paper_section4_backaction.py` | `backaction_mse_nees_vs_b.png` |
| `run_paper_section4_multip.py` | `multip_mse_nees_vs_sigma.png` |
| `run_paper_section4_sensitivity.py` | console output — mean ± std of MSE over 30 seeds |

```bash
python3 -m prg.run_paper_section4
python3 -m prg.run_paper_section4_backaction
python3 -m prg.run_paper_section4_multip
python3 -m prg.run_paper_section4_sensitivity
```

### Section 5 — Real Data Experiment (S&P 500 Stochastic Volatility)

| Script | Figures generated |
|--------|------------------|
| `run_paper_section5.py` | `nn_gx_gy_sv.png`, `epkf_sv.png`, `upkf_sv.png`, `ppf_sv.png` |
| `run_paper_section5_enso.py` | archived ENSO experiment (Niño 3.4 / SOI), kept for reference |

```bash
python3 -m prg.run_paper_section5       # requires: pip install yfinance
python3 -m prg.run_paper_section5_enso  # archived version
```

> **Note:** all figures are saved in `papier_NonLinearPKF/figures/`.

---

## Usage Examples

### Simulate Linear Data and Filter with PKF

```bash
awesomepkf-simulate --N 2000 --linear-model-name "model_x1_y1_AQ_pairwise" --data-filename "testL.csv" --verbose 1 --s-key 303
awesomepkf-pkf      --linear-model-name "model_x1_y1_AQ_pairwise" --data-filename "testL.csv" --verbose 1 --save-history --plot
```

### Simulate Non-Linear Data and Filter with EPKF, UPKF and PPF

```bash
awesomepkf-simulate --N 1000 --nonlinear-model-name "model_x2_y1_pairwise" --data-filename "testNL.csv" --verbose 1 --s-key 303

awesomepkf-epkf --nonlinear-model-name "model_x2_y1_pairwise" --data-filename "testNL.csv"                      --verbose 1 --save-history --plot
awesomepkf-upkf --nonlinear-model-name "model_x2_y1_pairwise" --data-filename "testNL.csv" --sigma-set "wan2000"  --verbose 1 --save-history --plot
awesomepkf-ppf  --nonlinear-model-name "model_x2_y1_pairwise" --data-filename "testNL.csv" --n-particles 300      --verbose 1 --save-history --plot
```

---

## Folders structure

<!-- PROJECT_STRUCTURE_START -->
```text
./
|-- data/
|   |-- datafile/
|   |-- historyTracker/
|   |-- plot/
|   `-- clean_dirs.sh
|-- notebooks/
|-- prg/
|   |-- base_classes/
|   |   |-- __init__.py
|   |   |-- filter_runner.py
|   |   |-- filter_specs.py
|   |   |-- runner_base.py
|   |   |-- simulator_base.py
|   |   |-- simulator_linear.py
|   |   `-- simulator_nonlinear.py
|   |-- classes/
|   |   |-- history_tracker/
|   |   |   |-- __init__.py
|   |   |   |-- _core.py
|   |   |   |-- _demo.py
|   |   |   |-- _metrics_mixin.py
|   |   |   `-- _plot_mixin.py
|   |   |-- matrix_diagnostics/
|   |   |   |-- __init__.py
|   |   |   |-- base.py
|   |   |   |-- covariance.py
|   |   |   |-- invertible.py
|   |   |   |-- results.py
|   |   |   |-- stability.py
|   |   |   |-- status.py
|   |   |   `-- tolerances.py
|   |   |-- __init__.py
|   |   |-- _base_particle_filter.py
|   |   |-- linear_pkf.py
|   |   |-- linear_pks.py
|   |   |-- nonlinear_epkf.py
|   |   |-- nonlinear_epks.py
|   |   |-- nonlinear_pf.py
|   |   |-- nonlinear_ppf.py
|   |   |-- nonlinear_pps.py
|   |   |-- nonlinear_ukf.py
|   |   |-- nonlinear_uks.py
|   |   |-- nonlinear_upkf.py
|   |   |-- nonlinear_upks.py
|   |   |-- param_linear.py
|   |   |-- param_nonlinear.py
|   |   |-- pkf.py
|   |   |-- seed_generator.py
|   |   `-- sigma_points_set.py
|   |-- models/
|   |   |-- linear/
|   |   |   |-- __init__.py
|   |   |   |-- _amq.py
|   |   |   |-- _base.py
|   |   |   |-- _dynamics.py
|   |   |   |-- _plotting.py
|   |   |   |-- _sigma.py
|   |   |   |-- _symbolic.py
|   |   |   |-- base_model_linear.py
|   |   |   `-- configs.py
|   |   |-- nonlinear/
|   |   |   |-- __init__.py
|   |   |   |-- _latex_helpers.py
|   |   |   |-- base_model_fxhx.py
|   |   |   |-- base_model_gxgy.py
|   |   |   |-- base_model_nonlinear.py
|   |   |   |-- configs.py
|   |   |   |-- model_x1_y1_augmented.py
|   |   |   |-- model_x1_y1_lotkavolterra_augmented.py
|   |   |   |-- model_x1_y1_markov_naive.py
|   |   |   |-- model_x1_y1_multiplicative.py
|   |   |   |-- model_x1_y1_multiplicative_augmented.py
|   |   |   |-- model_x1_y1_pairwise_param.py
|   |   |   |-- model_x2_y1_augmented.py
|   |   |   |-- model_x2_y1_classic.py
|   |   |   `-- model_x2_y2_augmented.py
|   |   `-- __init__.py
|   |-- tests/
|   |   |-- __init__.py
|   |   |-- conftest.py
|   |   |-- test_cli.py
|   |   |-- test_linear_pkf.py
|   |   |-- test_linear_pks.py
|   |   |-- test_models.py
|   |   |-- test_nonlinear_epks.py
|   |   |-- test_nonlinear_filters.py
|   |   |-- test_nonlinear_pps.py
|   |   |-- test_nonlinear_uks.py
|   |   |-- test_nonlinear_upks.py
|   |   |-- test_particle_filters.py
|   |   `-- test_simulators.py
|   |-- utils/
|   |   |-- __init__.py
|   |   |-- csv_to_parquet.py
|   |   |-- display.py
|   |   |-- exceptions.py
|   |   |-- generate_matrix_cov.py
|   |   |-- io.py
|   |   |-- metrics.py
|   |   |-- nn_model.py
|   |   |-- numerics.py
|   |   |-- parser.py
|   |   `-- plot_settings.py
|   |-- __init__.py
|   |-- run_filter.py
|   |-- run_linear_pkf.py
|   |-- run_nonlinear_epkf.py
|   |-- run_nonlinear_pf.py
|   |-- run_nonlinear_ppf.py
|   |-- run_nonlinear_ukf.py
|   |-- run_nonlinear_upkf.py
|   |-- run_paper_section4.py
|   |-- run_paper_section4_backaction.py
|   |-- run_paper_section4_multip.py
|   |-- run_paper_section4_sensitivity.py
|   |-- run_paper_section5.py
|   |-- run_paper_section5_enso.py
|   `-- run_simulator.py
|-- .gitignore
|-- CHANGELOG.md
|-- LICENSE
|-- README.md
|-- pyproject.toml
|-- requirements.txt
`-- update_readme_structure.sh

```
<!-- PROJECT_STRUCTURE_END -->
