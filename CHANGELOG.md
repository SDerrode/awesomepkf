# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [Unreleased]

---

## [0.2.0] - 2026-03-18

### Added

- **Lotka-Volterra nonlinear pairwise model** (`model_x1_y1_LotkaVolterra_pairwise`)
  - Symplectic Suris integrator for long-term stability
  - Additive noise formulation in both standard and augmented variants
  - New tutorial `tutorial_05` demonstrating PKF/EPKF/UPKF/PPF on a prey-predator system
- **Lotka-Volterra parameter estimation script** (`estimate_lotka_volterra.py`)
- **Real data pipeline** for Lotka-Volterra CSV files
  - Cleaned CSV output `*_clean_xy.csv` with columns `(t, X0, Y0)`
  - Glob restricted to `[Cc][0-9]*.csv` to avoid processing generated files
  - `realdata/` subdirectory preserved by `clean_dirs.sh`

### Fixed

- `latex_model()` now handles multiplicative noise correctly
- `NonLinear_PF` no longer crashes on pairwise models (tutorial_04)
- Augmented LV model: Q scaled by equilibrium² to prevent Cholesky failure
- Augmented LV model: Euler scheme used to prevent exp overflow

---

## [0.1.0] - 2026-03-16

### Added

- **Pairwise Kalman Filter (PKF)** for linear state-space models
  - Support for classic, augmented and pairwise model formulations
  - Linear models: `x1_y1`, `x2_y2`, `x3_y1` with `AQ` and `Sigma` parameterizations
- **Extended Pairwise Kalman Filter (EPKF)** for nonlinear models
- **Unscented Pairwise Kalman Filter (UPKF)** with multiple sigma-point sets
  - `wan2000` (Wan & Van der Merwe), `cubature`, and custom sets via `SigmaPointsSet`
- **Unscented Kalman Filter (UKF)** adapted for pairwise model parameterization
  - Correct noise extraction from `B @ mQ @ B^T`
  - H-recovery from pairwise `A` matrix (`H = A_yx @ inv(F)`)
  - Cross-covariance M correction in the update step
- **Pairwise Particle Filter (PPF)** and **Bootstrap Particle Filter (PF)**
  - Sequential importance resampling
  - Configurable number of particles
- **Data simulator** (`run_simulator.py`) for linear and nonlinear models
- **CLI entry points**: `awesomepkf-simulate`, `awesomepkf-pkf`, `awesomepkf-epkf`,
  `awesomepkf-upkf`, `awesomepkf-ukf`, `awesomepkf-ppf`, `awesomepkf-pf`
- NEES and NIS calibration metrics with history tracking
- Rich terminal output and matplotlib plots

[Unreleased]: https://github.com/sderrode/awesomepkf/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/sderrode/awesomepkf/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/sderrode/awesomepkf/releases/tag/v0.1.0
