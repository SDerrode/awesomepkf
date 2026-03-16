# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [Unreleased]

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

[Unreleased]: https://github.com/sderrode/awesomepkf/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/sderrode/awesomepkf/releases/tag/v0.1.0
