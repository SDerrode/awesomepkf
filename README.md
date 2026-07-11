# AwesomePKF

This repository contains a set of programs illustrating the **Pairwise Kalman Filter (PKF)**, a generalization of the classical Kalman Filter, extended to non-linear models. It includes several variants of non-linear filters:

- **Extended Pairwise Kalman Filter (EPKF)**
- **Unscented Pairwise Kalman Filter (UPKF)**, with multiple variants depending on the choice of sigma points
- **Pairwise Particle Filter (PPF)**

together with the classic (non-pairwise) **Unscented Kalman Filter (UKF)** and **Particle Filter (PF)** as baselines.

For offline post-processing it provides the **Linear Pairwise Kalman Smoother (PKS)** — a backward pass on the joint `(X, Y)` Markov chain, available in **six equivalent variants** selectable via `method=`: `RTS` (default), `BF` (Bryson-Frazier), `MBF` (Modified Bryson-Frazier), `MF` (Mayne-Fraser two-filter, alias `2F`), `DWY` (Desai-Weinert-Yusypchuk backward filter) and `VAR` (variational / lifted block-tridiagonal). All return the same estimate on the linear-Gaussian model (Geng et al., 2023); `VAR` additionally exposes the lag-one cross-covariance `Mk_smooth`.

---

## Papers

This library is the reference implementation for two companion papers on Gaussian pairwise Markov models:

1. **Smoothing, Learning, and Testing the Gaussian Pairwise Markov Model** — the six equivalent linear smoothers, learning the couple coefficients (back-action `A_xy`, observation memory `A_yy`) by partial EM, and a likelihood-ratio test for measurement back-action. *Reproduce it with* [`experiments/`](experiments/) (Figs. 1–3, Tables II–III) and tutorials [`07`](notebooks/tutorial_07_smoothers.ipynb), [`09`](notebooks/tutorial_09_linear_smoothers.ipynb), [`10`](notebooks/tutorial_10_learning_and_testing.ipynb).
2. **Nonlinear Extensions of the Gaussian Pairwise Kalman Filter** (EPKF / UPKF; [hal-05645741](https://hal.science/hal-05645741)). *Reproduce it with* the `run_nonlinear_{epkf,upkf,ppf}.py` scripts and tutorials [`02`](notebooks/tutorial_02_nonlinear_models.ipynb), [`05`](notebooks/tutorial_05_new_model_lotkavolterra.ipynb).

See [`experiments/README.md`](experiments/README.md) for exact commands, expected numbers, and runtimes.

---

## Table of Contents

- [AwesomePKF](#awesomepkf)
    - [Table of Contents](#table-of-contents)
    - [Papers](#papers)
    - [Installation](#installation)
    - [Models and Simulations](#models-and-simulations)
    - [Filters](#filters)
        - [Pairwise Kalman Filter (PKF)](#pairwise-kalman-filter-pkf)
        - [Extended Pairwise Kalman Filter (EPKF)](#extended-pairwise-kalman-filter-epkf)
        - [Unscented Pairwise Kalman Filter (UPKF)](#unscented-pairwise-kalman-filter-upkf)
        - [Pairwise Particle Filter (PPF)](#pairwise-particle-filter-ppf)
        - [Unscented Kalman Filter (UKF)](#unscented-kalman-filter-ukf)
        - [Particle Filter (PF)](#particle-filter-pf)
    - [Smoothers — Linear PKS (six variants)](#smoothers)
        - [Linear Pairwise Kalman Smoother (PKS)](#linear-pairwise-kalman-smoother-pks)
    - [Tutorials](#tutorials)
    - [Parameter learning from data](#parameter-learning-from-data)
    - [Usage Examples](#usage-examples)
        - [Simulate Linear Data and Filter with PKF](#simulate-linear-data-and-filter-with-pkf)
        - [Simulate Non-Linear Data and Filter with EPKF, UPKF and PPF](#simulate-non-linear-data-and-filter-with-epkf-upkf-and-ppf)

---

## Installation

### From PyPI (recommended)

```bash
pip install awesomepkf
```

### From source

```bash
git clone https://github.com/SDerrode/awesomepkf.git
cd awesomepkf
pip install .
```

### Development install

```bash
pip install -e ".[dev]"
```

### Requirements

- Python >= 3.10
- numpy, scipy, matplotlib, pandas, rich, chardet, sympy (plus tomli on Python < 3.11)

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
| 06 | [`tutorial_06_filter_runner_and_config.ipynb`](notebooks/tutorial_06_filter_runner_and_config.ipynb) | High-level orchestration with `FilterRunner` and `RunOptions`; parameter sweeps via `model_kwargs`; saving and replaying experiments through a reproducible JSON spec |
| 07 | [`tutorial_07_smoothers.ipynb`](notebooks/tutorial_07_smoothers.ipynb) | Pairwise smoothing — backward recursion and forward-filtering / backward-smoothing; ±2σ envelope shrinkage, the Joseph form, and a decision rule for choosing among the available smoothers |
| 08 | [`tutorial_08_real_data_pkf_learning.ipynb`](notebooks/tutorial_08_real_data_pkf_learning.ipynb) | Estimating the 1D linear PMM parameters `(a, b, c, d, e)` from a real two-column time series (wind-farm active power vs wind speed); PMM vs HMM/Kalman projection; converting to `LinearAmQ` kwargs |
| 09 | [`tutorial_09_linear_smoothers.ipynb`](notebooks/tutorial_09_linear_smoothers.ipynb) | The six linear pairwise smoothers behind one façade `Linear_PKS(param, method=...)` — RTS, BF, MBF, MF, DWY and the variational/lifted `VAR` form; shows all six return the same smoothed mean/covariance to round-off (~1e-15), companion to the `python -m prg.run_dwy_equivalence` non-regression check |
| 10 | [`tutorial_10_learning_and_testing.ipynb`](notebooks/tutorial_10_learning_and_testing.ipynb) | Learning and testing the couple structure: partial-EM recovery of the back-action `A_xy` and observation-memory `A_yy` from the classical initialisation, the `A_yy`-tighter-than-`A_xy` identifiability, and a likelihood-ratio test for back-action (`prg.learning.em_partial_dynamics`) |

---

## Parameter learning from data

The [`prg.learning`](prg/learning/) module offers three estimators for the
linear-Gaussian case: a **method-of-moments** fit of the scalar PMM, a **partial EM**
for the joint noise covariance, and a **partial EM** for the couple dynamics blocks
(back-action and observation memory) together with a likelihood-ratio test for
back-action.

### Method of moments (scalar PMM)

For the **linear, scalar** case (`dim_x = dim_y = 1`), it estimates the five PMM
parameters `(a, b, c, d, e)` from a two-column time series by the method of moments.

```bash
awesomepkf-fit-pkf \
    --data-filename data/samples/windfarms/site1_202210_Month_586_norm.csv \
    --x-col ActivePower_KWh --y-col WindSpeed \
    --output learned_params.npz --verbose 1
```

On the embedded WindFarms series the fit lands clearly outside the HMM
(classical Kalman) submanifold — the off-HMM gap `Δc ≈ 0.53` between the
estimated `c` and its HMM projection `a·b²` means a pairwise model tracks the
state more tightly than the classical KF. See
[`tutorial_08`](notebooks/tutorial_08_real_data_pkf_learning.ipynb) for the
full load → fit → compare → convert workflow.

A small WindFarms sample is shipped under [`data/samples/`](data/samples/);
the full dataset (BuildingTemp, SeattleTemp, multiple WindFarms sites and
granularities) is kept outside the repository — point `--data-filename` at a
local copy if needed.

### Partial EM for the noise covariance (linear-Gaussian)

For a linear-Gaussian pairwise couple `Z = (X, Y)` with the transition `A`
**known**, [`prg.learning.estimate_noise_em`](prg/learning/em_partial_noise.py)
runs a partial EM that estimates **only** the joint process-noise covariance `Q`
(with `B = I`, so `Q` is the effective process covariance). The E-step uses the
variational linear smoother (`method="VAR"`, the only backward pass exposing the
lag-one cross-covariance `Mk_smooth`); the M-step is a unique closed form given
`A`, and the observed-data log-likelihood it returns is monotone non-decreasing.

```python
from prg.learning import estimate_noise_em

result = estimate_noise_em(param, data, block_diagonal=True)
result.Q          # estimated (dim_xy, dim_xy) joint noise covariance, PSD
result.loglik     # per-iteration observed-data log-likelihood
result.converged  # bool
```

**Identifiability.** Fixing `A` removes the `(A, Q)` gauge freedom, but the
*full* joint `Q` is still not identifiable from the hidden state: the cross-noise
block `Q_xy` lies on a near-flat likelihood ridge and the EM estimate of `Q_xy`
tracks the initialisation rather than converging to the truth as `N` grows. The
X- and Y-noise *blocks* remain well identified, so pass `block_diagonal=True` to
fit the well-conditioned block-diagonal sub-model (`Q_xy = 0`), which recovers
cleanly at modest `N`. This is a **library function** (returning an
`EMNoiseResult`) — unlike the PMM estimator it has **no CLI**.

### Partial EM for the couple dynamics — back-action and observation memory (linear-Gaussian)

Dually, [`prg.learning.estimate_dynamics_em`](prg/learning/em_partial_dynamics.py)
holds `A_xx`, `A_yx`, `Q` fixed and learns the two **couple-defining** transition
blocks — the back-action `A_xy` (Y → X) and the observation memory `A_yy` (Y → Y),
both zero in a classical model — starting from the classical initialisation
`A_xy = A_yy = 0`. The E-step is one `VAR` smoothing pass; the M-step regresses each
residual on the observed `y`. Because `A_yy` couples observed coordinates it is fully
identifiable, whereas the latent-mediated `A_xy` is identifiable only up to the latent
gauge — so `A_yy` recovers more tightly.

Whether back-action is present at all is a hypothesis test:
[`back_action_lrt`](prg/learning/em_partial_dynamics.py) fits `A_xy` free vs.
`A_xy = 0` (with `A_yy` a free nuisance in both) and returns the likelihood-ratio
statistic, asymptotically `χ²` with `dim_x·dim_y` degrees of freedom under `H0`.

```python
from prg.learning import estimate_dynamics_em, back_action_lrt

res = estimate_dynamics_em(classical_init_param, data)   # learns A_xy, A_yy
res.A_xy, res.A_yy, res.loglik, res.converged

lrt = back_action_lrt(classical_init_param, data)        # test H0: A_xy = 0
lrt.stat, lrt.dof, lrt.pvalue
```

See [`tutorial_10`](notebooks/tutorial_10_learning_and_testing.ipynb) and the
paper-reproduction scripts in [`experiments/`](experiments/)
(`em_identification.py`, `em_lrt.py`).

Parameter identification for **nonlinear** EPKF/UPKF models is *not* covered
by this estimator — those require a separate procedure (e.g. a neural
network).

---

## Models and Simulations

The repository provides a program called **run_simulator.py** to simulate data according to **linear and non-linear models**.

---

## Filters

Each filter is exposed through a single `run_<filter>.py` script (a thin wrapper over `prg.run_filter`). The same script does both jobs: it **simulates and filters** when given `--N`, or **filters a previously saved file** when given `--data-filename` (the two are mutually exclusive).

### Pairwise Kalman Filter (PKF)

- **run_linear_pkf.py** – filter linear data either from simulated data or from a previously saved file (e.g., generated with `run_simulator.py`)  

### Extended Pairwise Kalman Filter (EPKF)

- **run_nonlinear_epkf.py** – filter non-linear data either from simulated data or from a previously saved file (e.g., generated with `run_simulator.py`)  

### Unscented Pairwise Kalman Filter (UPKF)

- **run_nonlinear_upkf.py** – filter non-linear data either from simulated data or from a previously saved file (e.g., generated with `run_simulator.py`)  

### Pairwise Particle Filter (PPF)

- **run_nonlinear_ppf.py** – filter non-linear data either from simulated data or from a previously saved file (e.g., generated with `run_simulator.py`)  

### Unscented Kalman Filter (UKF)

- **run_nonlinear_ukf.py** – the classic (non-pairwise) sigma-point filter for models Markov in `X` alone; takes `--sigma-set`.

### Particle Filter (PF)

- **run_nonlinear_pf.py** – the classic (non-pairwise) bootstrap particle filter; takes `--n-particles`.

---

## Smoothers

Smoothers are **two-pass, offline** estimators that condition on the *entire* observation sequence `y_{1:N}`. They produce posterior means and covariances `p(X_n | y_{1:N})` that are at least as good (in PSD sense) as the corresponding forward filter outputs `p(X_n | y_{1:n})`.

### Linear Pairwise Kalman Smoother (PKS)

The linear PKS runs the [PKF forward](#pairwise-kalman-filter-pkf), then a backward Rauch-Tung-Striebel recursion at the **joint** `(X, Y)` level. The pairwise model is Markov in `Z = (X, Y)` (not in `X` alone), so the smoothing gain `G_n` has shape `(dim_x, dim_x + dim_y)`. Equivalently, the linear PKS is the classical RTS smoother applied to the augmented state `Z' = (X, Y)` with degenerate observation `Y_n = (0, I) Z'_n` (`R^aug = 0`).

`Linear_PKS` is a façade that selects the backward pass via `method=` — one of six: `"RTS"` (default), `"BF"` (Bryson-Frazier), `"MBF"` (Modified Bryson-Frazier), `"MF"` (Mayne-Fraser two-filter, alias `"2F"`), `"DWY"` (Desai-Weinert-Yusypchuk backward filter) and `"VAR"` (variational / lifted). All return the same smoothed mean and covariance as RTS to machine precision on the linear-Gaussian model (cf. Geng et al., 2023; verified by `test_variant_equals_rts` / `test_dwy_equals_rts`), and the matching explicit classes `Linear_PKS_RTS`, `Linear_PKS_BF`, `Linear_PKS_MBF`, `Linear_PKS_MF`, `Linear_PKS_DWY`, `Linear_PKS_VAR` are all exported. `VAR` solves for the whole smoothed trajectory as a single **block-tridiagonal** linear system (lifted / quadratic-program form, requiring full-rank process noise `R = B Q Bᵀ ≻ 0`) and is the only variant that also writes the lag-one cross-covariance `Mk_smooth[n] = Cov(X_{n+1}, X_n | y_{0:N})` — the cross-moment the EM noise estimator consumes.

The six backward passes (all write `Xkp1_smooth` / `PXXkp1_smooth`; only `VAR` also writes `Mk_smooth`):

| `method=` | Smoother | Backward pass |
|-----------|----------|---------------|
| `"RTS"` *(default)* | Rauch–Tung–Striebel | classical forward–backward smoothing gain `G_n` (shape `dim_x × dim_xy`) coupling `X_n` to the whole couple `Z_{n+1}` |
| `"BF"` | Bryson–Frazier | adjoint smoother at the **couple** level: propagates the predicted-couple adjoint `(μ_n, N_n)` |
| `"MBF"` | Modified Bryson–Frazier (Bierman) | adjoint smoother at the **filtered `X`** level (`dim_x`), via the adjoint pair `(λ_n, Λ_n)` |
| `"MF"` *(alias `"2F"`)* | Mayne–Fraser (two-filter) | fuses, in **information** form, the forward posterior with a backward information filter |
| `"DWY"` | Desai–Weinert–Yusypchuk | dual of RTS: a **backward** pairwise filter on the time-reversed (complementary) couple model |
| `"VAR"` | Variational / lifted | one **block-tridiagonal** solve (quadratic program in the latent trajectory); the only variant exposing `Mk_smooth` |

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

Implementation: [`prg/classes/linear_pks.py`](prg/classes/linear_pks.py). Tests: [`prg/tests/test_linear_pks.py`](prg/tests/test_linear_pks.py) (62 tests, including PSD shrinkage, Joseph equivalence, augmented-state RTS equivalence, the BF/MBF/MF/DWY/VAR ≡ RTS equivalence, the VAR lag-one cross-covariance, and full exception/logging coverage).

#### Deterministic control (consigne)

All six variants accept an optional known control input `u_n`: the couple obeys
`Z_{n+1} = A Z_n + G u_n + B W_n` with a control matrix `G` (shape `(dim_xy, dim_u)`).
Pass `G` to `ParamLinear(..., G=G)` and the control sequence `u` to
`simulate_N_data(N, u=...)` and `process_N_data_smoother(N, ..., u=...)`. A
deterministic control shifts only the means (never the covariances), so it is
applied by **exact mean-trajectory superposition** and behaves identically across
all six backward passes (which stay equivalent to ~1e-15). `u=None` / `G=None`
keeps the autonomous model — fully backward compatible.

```python
import numpy as np
G   = np.array([[0.6], [0.3]])                    # control on the (X, Y) couple
param = ParamLinear(0, dim_x, dim_y, **params, G=G)
sim = Linear_PKF(param, sKey=42).simulate_N_data(N, u=u)
res = Linear_PKS(param, method="VAR").process_N_data_smoother(
    N=None, data_generator=iter(sim), u=u)
```

Tests: [`prg/tests/test_linear_pks_control.py`](prg/tests/test_linear_pks_control.py) (4 tests).

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
