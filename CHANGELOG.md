# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [Unreleased]

---

## [2.3.0] - 2026-05-15

### Added

- **`NonLinear_EPKS` — Extended Pairwise Kalman Smoother** ([`prg/classes/nonlinear_epks.py`](prg/classes/nonlinear_epks.py)). Two-pass smoother extending `NonLinear_EPKF`. The backward pass reuses the linear PKS recursion with the per-step Jacobian `F_{n+1}` (evaluated at the filtered point `(X_{n|n}, y_n)`) replacing the constant `A` matrix. Jacobians are recomputed in the backward pass rather than stored — adds one extra `param.jacobiens_g` call per step, but requires no API change on the parent `NonLinear_EPKF`. Joseph form available via the `joseph=True` flag, mathematically equivalent (~1e-16 in double precision) and more useful here than in the linear case (per-step Jacobians can be locally ill-conditioned).
- **EPKS report section** (`Report/NonLinearSmoothingReport/Sections/Section3_EPKS.tex`) with derivation, Joseph form, augmented-state EKF equivalence remark, and a dedicated §3.6 explaining why PSD shrinkage of the *linearised* covariance does not translate to systematic per-trajectory MSE reduction (linearisation bias).
- **EPKS figure generator** (`Report/.../Figures/generate_epks_figure.py`) with the same auto-detected repo-root mechanism as the linear PKS script.

### Caveats
- **Not suitable for augmented nonlinear models.** When `param.augmented=True`, the joint predicted covariance `P^{ZZ}_{n+1|n}` is structurally rank-deficient (Y is a noise-free function of X), and the backward Cholesky fails with `CovarianceError`. The EPKS in this release targets the pairwise (non-augmented) family. Documented in the §3.5 of the report.

### Tests
- **+25 tests in `prg/tests/test_nonlinear_epks.py`**: shapes, terminal equality, PSD shrinkage on linearised covariance, Joseph form equivalence and shrinkage, edge cases (N=1, generator semantics, double-call, external `data_generator`, missing ground truth), exception policy (`ParamError`, `PKFError` root), `caplog`-based INFO/DEBUG emission assertions, and a regression test that the EPKS MSE does not significantly degrade vs the EPKF (ratio < 1.02 on 20 seeds × N=300 of `model_x2_y1_pairwise`).
- Total: 158 tests pass (up from 133).

---

## [2.2.0] - 2026-05-15

### Added

- **`Linear_PKS` — linear pairwise Kalman smoother** ([`prg/classes/linear_pks.py`](prg/classes/linear_pks.py)). Two-pass RTS-style smoother extending `Linear_PKF`. The backward pass operates at the **joint** `(X, Y)` level: the pairwise model is Markov in `Z = (X, Y)` (not in `X` alone — the `A^{yx}` block couples `Y_{n+1}` directly to `X_n`), so the smoothing gain `G_n` is a `(dim_x, dim_xy)` matrix mapping the joint `Z`-residual to the `X`-correction. New API: `process_smoother(N, data_generator)` generator and `process_N_data_smoother(N, ...)` eager wrapper, both yielding `(k, x_true, y, X_predict, X_update, X_smooth)`.
- **Joseph form of the smoother covariance update** (`joseph=True` flag). Explicitly symmetric and PSD-preserving form `P^{xx}_{n|N} = (I, -G_n) Omega_n (I, -G_n)^T + G_n^x P^{xx}_{n+1|N} (G_n^x)^T`, analog of the paper's `PKFJoseph` proposition for the forward step. Agrees with the standard form to machine precision (`~1e-15`) on the linear case; becomes useful for the upcoming nonlinear extensions (EPKF / UPKF smoothers).
- **`HistoryTracker` public API for indexed access and in-place updates** — `__getitem__`, `__iter__`, `update_record(idx, **fields)`. Two-pass algorithms (smoothers) can now augment forward-pass records without touching the private `_history` list. Backward-compatible: existing callers using `record()` / `last()` / `as_dataframe()` are unaffected.
- **Linear smoother report** (in a separate doc tree, see `Report/NonLinearSmoothingReport/`) with derivation, Joseph form proof, and reproducibility script for the comparison figures.

### Logging & verbose
- Module logger `prg.classes.linear_pks` emits `INFO` at backward-pass entry/exit (with `N_records` and `joseph` mode) and `DEBUG` per step (gain norm + traces, gated by `isEnabledFor` to skip formatting cost when off).
- `verbose > 1` calls `rich_show_fields` on each smoothed record, mirroring the forward-pass display.

### Exception policy
- `CovarianceError` is raised with `(step, matrix_name)` attributes on backward Cholesky failure of `P^{ZZ}_{n+1|n}` or PSD violation of the smoothed covariance. The standard project taxonomy (`ParamError`, `InvertibilityError`, `NumericalError`, `StepValidationError`, `FilterError`) propagates unwrapped through `process_smoother` and `process_N_data_smoother`.

### Tests
- **+40 tests in `prg/tests/test_linear_pks.py`** covering shapes, terminal equality, PSD shrinkage of `P^{xx}_{n|n} - P^{xx}_{n|N}`, Joseph form equivalence and shrinkage, augmented-state RTS equivalence (with a standalone reference smoother implementation in the test file), MSE regression on `x1y1` and `x2y2` pairwise models, edge cases (`N=1`, generator semantics, double-call, external `data_generator`, missing ground truth, singular predicted covariance), `HistoryTracker` public-API contracts, full exception taxonomy (`PKFError` root, `ParamError` for invalid `N`), and `caplog`-based assertions on `INFO`/`DEBUG` log emission.
- Total: 133 tests pass (up from 93).

---

## [2.1.2] - 2026-05-05

### Fixed

- **CI lint** — replaced the ``zero_noise = {s: 0 for s in ...}`` dict comprehension introduced in v2.1.1 with ``dict.fromkeys(..., 0)`` (ruff C420).
- **CI tests on every Python version** — moved ``pytest-qt`` from the ``[dev]`` extras group to ``[gui]``. ``pytest-qt`` requires a Qt binding (PyQt6 / PySide6) at *import time*; with the binding only available under ``[gui]``, ``pip install -e .[dev]`` (used by CI) was triggering ``ERROR: pytest-qt requires either PySide6, PyQt5 or PyQt6 installed`` and aborting collection. ``[dev]`` is now self-sufficient for the public test suite; GUI test collection requires ``[gui,dev]`` (or ``[all]``).

---

## [2.1.1] - 2026-05-05

### Fixed

- **EPKF crash on `BaseModelFxHx` models with multiplicative noise** — the symbolic Jacobians ``df/dx`` and ``dh/dx`` were lambdified over state variables only. For additive-noise models the Jacobians contain no noise symbols, so this happened to work; for multiplicative-noise models such as ``model_x1_y1_multiplicative_augmented`` the noise symbols survived in the Jacobian and ``np.array(..., dtype=float)`` raised ``TypeError: Cannot convert expression to float`` at the first prediction step. The Jacobians are now substituted at the linearization point ``noise = 0`` (the noise mean) before lambdify, which is the correct linearization for the EKF/EPKF and a no-op for additive-noise models. **No paper result is impacted**: this combination is not exercised by any of the ``run_paper_section*.py`` scripts (UKF runs on the augmented model, sigma-point based; EPKF/UPKF run on the pairwise version which uses ``BaseModelGxGy``).

### Changed

- **Soft post-construction kwargs in both factories** — ``ModelFactoryLinear.create`` and ``ModelFactoryNonLinear.create`` no longer raise on extra kwargs. Class-based nonlinear models receive the constructor-accepted subset; everything else (including kwargs that survive on config-driven models) becomes a post-construction ``setattr`` for attributes that already exist on the instance. Lets the Sensitivity tab sweep universal UPKF / UKF tuning knobs (``alpha`` / ``beta`` / ``kappa``) on any model without surfacing the config/class distinction.

---

## [2.1.0] - 2026-05-05

### Added

- **`model_kwargs` on `FilterRunner`** — constructor-time scalar overrides forwarded to the model factory. Required for parameter sweeps to genuinely vary `q_x`, `q_y`, etc. (the previous post-construction `setattr` path was silently no-op for matrices already baked into `param.mQ`).
- **Factories accept `**kwargs`** — `ModelFactoryNonLinear.create(name, **kwargs)` forwards to class-discovered constructors; `ModelFactoryLinear.create(name, **kwargs)` raises if any are passed (linear models are config-driven).
- **Tutorial 06** — [`tutorial_06_filter_runner_and_config.ipynb`](notebooks/tutorial_06_filter_runner_and_config.ipynb): high-level orchestration with `FilterRunner` + `RunOptions`, parameter sweeps via `model_kwargs`, and TOML session-config replay through `awesomepkf-<filter> --config foo.toml`.

### Changed

- **`ipynb/` → `notebooks/`** — directory renamed to the conventional name. README, `pyproject.toml` (`tool.coverage.exclude`), and internal notebook references updated. Notebook `sys.path` insertions are unchanged (depth-relative).
- **Tutorials 01 and 02** "Going Further" tables list the unified dispatcher entry points (`awesomepkf-<filter> --config session.toml`, `python -m prg.run_<filter>`).
- **`.gitignore`** — added `/prg/gui` (no trailing slash) so symlinks pointing at the canonical GUI directory are also ignored across worktrees.

### Fixed

- **Tutorial 05** — `prg/models/nonLinear/` (camel-case path used as `Path(...).write_text` target) corrected to `prg/models/nonlinear/`. On case-sensitive filesystems the previous path silently created a parallel directory the factory never scanned.
- **Tutorials 02 and 04** — model name `model_x1_y1_gordon_classic` corrected to `model_x1_y1_Gordon_classic`.

---

## [0.4.0] - 2026-04-21

### Changed

- **Section 5 experiment replaced**: the real-data experiment (Section 5 of the paper) now uses S&P 500 stochastic volatility instead of ENSO climate data
  - Latent state: log-variance $x_n = \log(\mathrm{RV}_n^{\mathrm{Park}})$ (Parkinson range-based estimator)
  - Observation: log-squared return $\tilde{y}_n = \log r_n^2 - \mu_w$, with log-χ² noise (σ ≈ 2.22)
  - Training period: 2000–2015 (~4 000 daily observations); test period: 2016–2023 (~2 000 daily observations)
  - EPKF achieves MSE = 1.26 on test set, beating linear regression (MSE = 1.43) and the single-shot estimator (MSE = 6.53)
- **`run_paper_section5.py`** rewritten for the S&P 500 SV experiment (downloads data via `yfinance`, Parkinson estimator as ground-truth proxy, generates `nn_gx_gy_sv.png`, `epkf_sv.png`, `upkf_sv.png`, `ppf_sv.png`)
- **`run_paper_section5_enso.py`** — original ENSO script archived under this new name for reference

### Added

- **Bibliography entries** in the paper: `Parkinson1980` (Parkinson range-based variance estimator) and `Taylor1994SV` (Taylor stochastic volatility review)
- **New figures**: `nn_gx_gy_sv.png`, `epkf_sv.png`, `upkf_sv.png`, `ppf_sv.png` (generated by `run_paper_section5.py`)

---

## [0.3.0] - 2026-04-21

### Added

- **Paper reproducibility scripts** for the article "Non-linear extensions to Gaussian pairwise Kalman filter"
  - `run_paper_section4.py` — synthetic experiment (EPKF / UPKF / PPF / EKF-aug / UKF-aug) with timing
  - `run_paper_section5.py` — real ENSO data experiment (downloads data, trains NNModel, runs filters)
- **NNModel** (`prg/utils/nn_model.py`) — MLP-based transition model (2×64, tanh) for data-driven dynamics estimation; exposes `g()` and `jacobians_g()` via PyTorch autograd
- **`--list-models` flag** in `run_simulator.py` to print all available linear and nonlinear models then exit

### Fixed

- **UPKF lambda bug** (`SigmaPointsSet.py`): `lambda_` is now recomputed locally from the actual sigma-point dimension `dim` (= `2p+q` for UPKF, `p` for UKF) rather than read from `param.lambda_` which was always computed from `dim_x` alone — this caused incorrect sigma-point spreads and weight values when `p ≠ q`

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

[Unreleased]: https://github.com/sderrode/awesomepkf/compare/v0.4.0...HEAD
[0.4.0]: https://github.com/sderrode/awesomepkf/compare/v0.3.0...v0.4.0
[0.3.0]: https://github.com/sderrode/awesomepkf/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/sderrode/awesomepkf/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/sderrode/awesomepkf/releases/tag/v0.1.0
