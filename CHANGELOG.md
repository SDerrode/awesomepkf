# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [Unreleased]

Documentation, paper-reproduction scripts and tutorials only: **no change to `prg/`**,
so the installable package is unchanged. Follows a cross-document consistency audit of the
companion paper; every published number is now produced by a released script.

### Added
- `experiments/identifiability_frozen.py` — checks that freezing `A^xx,A^yx,Q` restores
  identifiability of `(A^xy,A^yy)` iff `(A^xx,A^yx)` is observable (rank O = p), across
  (p,q) = (1,1),(2,2),(2,1); shows `A^yx != 0` alone is not enough when p > q.
- `experiments/em_observability.py` — partial-EM recovery on two p=2,q=1 couples sharing
  the same nonzero `A^yx`, differing only in observability rank.
- `experiments/em_realdata.py` — Diebold–Mariano and Clark–West held-out predictive tests
  (`dm_cw`, Newey–West SE), so the chemostat t-statistics (0.80 / 2.90 / 2.56) are computed,
  not quoted.
- `experiments/em_lrt.py` — caches the H0 null draws to `em_lrt_null.npz`.
- `notebooks/tutorial_10_learning_and_testing.ipynb` — enriched with the χ²_pq(λ)=2N·KL_rate
  law (cached MC null + analytic power curve) and a surrogate-calibrated real-data test;
  runs in ~25 s under nbmake. Bundled inputs in `notebooks/data/`.

### Fixed
- `experiments/backaction_tradeoff.py` — the estimability panel (Fig. 9b) is now recomputed
  from the spectra (state-optimal classical smoother) instead of hard-coded constants.
- `experiments/lrt_vector.py` — docstring corrected: the χ²_pq count is full under
  observability (rank O = p), not under q ≥ p.
- Figure legends/labels: 2F vs DWY made distinguishable (`discriminating_models.py`); NEES
  normalised to 1 (`classical_vs_couple.py`); shared α line and "DM conservative"
  (`rmse_vs_lrt.py`); "spectral KL_rate" axis label and no "gauge" for a frozen-block
  regime (`petetin_kl_comparison.py`); poles-figure legend moved off the true-pole marker
  (`backaction_oscillator.py`).

---

## [2.14.1] - 2026-07-20

Documentation and paper-reproduction scripts only: **no change to `prg/`**, so the
installable package is byte-identical to 2.14.0 (hence a patch bump, not a minor one).
The whole set stems from an audit of the companion paper; every number below was
re-derived rather than taken on trust.

### Added
- **Two paper-reproduction experiments (`experiments/`).**
  `petetin_kl_comparison.py` (Fig. 10) contrasts the estimability-side
  Kullback–Leibler divergence of Petetin–Desbouvries (IEEE TSP 62(14), 2014,
  Eq. 71) with the paper's spectral `KL_rate` testability gauge, and shows the
  back-action becomes testable only once a forward path `A_yx` opens a
  `Y`-footprint. `rmse_vs_lrt.py` (Fig. 11) shows that an *uncalibrated* RMSE
  comparison is not a valid back-action test — the nested couple wins the
  in-sample RMSE with probability 1 (held-out false-positive rate ≈ 0.32,
  ~6× nominal) — whereas a Diebold–Mariano / likelihood-ratio test is
  calibrated, with the identity `Λ ≈ N·log(MSE0/MSE1)` (corr. 0.997). Both are
  self-contained (numpy/scipy/matplotlib); `rmse_vs_lrt.py` caches its
  Monte-Carlo summary for a fast `--replot`.

### Changed
- **Experiment-figure polish.** `em_lrt.py` legend now reads `λ = 2N·KL_rate`
  (was `2N·KL`); `discriminating_models.py` labels the two-filter smoother `2F`
  (was `MF`), including its `Σ_n (2F/DWY)` legend; `pmm_schematic.py` draws the
  within-slice correlated-noise link `R_xy` at both time slices (the couple no
  longer appears to drop the vertical `X`–`Y` link). `experiments/README.md`
  renumbered and extended for Figs. 1–11.
- `.gitignore` now excludes the whole generated `figures/` directory (and
  `experiments/*.pdf`, `*.png`) rather than an enumerated list, so new
  experiment outputs are ignored automatically.

### Fixed
- **`em_identification.py` did not reproduce its own published figure.** The
  `--iters` default was 25, while the paper's coefficients
  (`A_xy = 0.395 ± 0.041`, `A_yy = 0.403 ± 0.022`) need 100 EM iterations; the
  bare command now reproduces them. Its runtime is documented honestly as
  **~40 min** (measured), not the "~3 min" previously claimed — the E-step is the
  library smoother, deliberately not replaced by a hand-rolled one.
- **`discriminating_models.py`: `cond(S_n) = 1` was a tautology.** With `q = 1`,
  `S_n` is a scalar, so that column carried no information. A new block **R3**
  reruns the starvation at `p = q = 2`, where `cond(S_n)` saturates near **8.7**
  while `cond(P)` reaches **1.2e6** — and where, below `eps = 1e-8`, the filter
  aborts on a singular `S_n`, showing MBF's advantage to be large but *bounded*.
- **`em_realdata.py`: the negative control promised in the docstring was never
  run.** It now is — circularly shifting the driver preserves each marginal and its
  autocorrelation while destroying the cross-coupling, giving `Λ = 0.74`
  (`p_surr = 0.53`, keep `H0`). The S&P rows are relabelled `return→volat.` /
  `volat.→return` to match the paper's table, and the per-system bar panel, which
  merely restated that table, was dropped.
- `petetin_kl_comparison.py` cited a hardcoded bibliography number in the figure
  legend, which went stale as soon as references were renumbered; it now cites by
  author. `backaction_tradeoff.py` writes `KL_rate`, not a bare `KL`.
- `experiments/README.md`: corrected three table numbers, both stale runtimes, and
  the pointer for the "one estimate" check (`prg.run_dwy_equivalence`).

---

## [2.14.0] - 2026-07-11

### Added
- **Partial EM for the couple dynamics + back-action test
  (`prg/learning/em_partial_dynamics.py`).** `estimate_dynamics_em` learns the two
  couple-defining transition blocks — the back-action `A_xy` (Y→X) and the
  observation memory `A_yy` (Y→Y) — by partial EM from the classical initialisation
  `A_xy = A_yy = 0`, with `A_xx`, `A_yx` and `Q` held **fixed** (the dynamics
  counterpart of the 2.12.0 noise estimator). `back_action_lrt` forms the
  likelihood-ratio statistic `Λ = 2[ℓ(A_xy free) − ℓ(A_xy = 0)]`, asymptotically
  `χ²` with `dim_x·dim_y` degrees of freedom, for testing `H0: A_xy = 0` (no
  back-action). Reproduces Figs. 2–3 of the companion paper; walked through in
  `notebooks/tutorial_10_learning_and_testing.ipynb`.
- **`'2F'` alias for the Mayne–Fraser two-filter smoother.** `Linear_PKS(param,
  method='2F')` is accepted as a synonym of `method='MF'`; both dispatch to the same
  two-filter backward pass.
- **Paper-reproduction scripts (`experiments/`).** Self-contained, deterministic
  drivers for the linear-smoothers paper — `classical_vs_couple.py` (Fig. 1),
  `em_identification.py` (Fig. 2), `em_lrt.py` (Fig. 3) and `lrt_vector.py`
  (vector-data `χ²_pq` check) — with `experiments/README.md` documenting the
  expected numbers, runtimes and outputs.

### Fixed
- Ruff lint on `prg/` (`F841` unused variable, `RUF022` unsorted `__all__`) and
  pinned `ruff==0.15.21` in CI so the lint job is reproducible across ruff releases.

---

## [2.13.0] - 2026-06-28

### Added
- **Deterministic control input (*consigne*) for the linear pairwise smoothers.**
  `ParamLinear` gains an optional control matrix `G` (shape `(dim_xy, dim_u)`);
  the couple then obeys `Z_{n+1} = A Z_n + G u_n + B W_n`. The control sequence
  `u` is passed to `Linear_PKF.simulate_N_data(N, u=...)` and
  `Linear_PKS.process_N_data_smoother(N, ..., u=...)`. A deterministic control
  shifts only the means (never the covariances), so it is applied by **exact
  mean-trajectory superposition** — uniform across all six backward passes
  (RTS/BF/MBF/MF/DWY/VAR), which stay equivalent to machine precision
  (~6e-15 with control). `u=None` / `G=None` is fully backward compatible
  (the autonomous model is unchanged). `prg/tests/test_linear_pks_control.py`
  (4 tests: cross-variant equivalence, control-is-used MSE check, covariance
  invariance, backward-compatibility).

---

## [2.12.0] - 2026-06-20

### Added
- **Partial EM noise estimator (`prg/learning/em_partial_noise.py`).**
  `estimate_noise_em` performs *partial* maximum-likelihood estimation by EM for
  the linear-Gaussian pairwise couple: the transition `A` is held **known**, and
  only the joint process-noise covariance `Q` is estimated. The E-step runs the
  variational smoother (`method="VAR"`, the only variant exposing the lag-one
  cross-covariance `Mk_smooth`); the M-step is the closed form
  `Q̂ = (1/T) Σ E[(Z_{n+1} − A Z_n)(Z_{n+1} − A Z_n)ᵀ | y_{1:N}]`, exploiting that
  the observed Y-block collapses the joint posterior onto the hidden X-block. A
  `block_diagonal=True` flag estimates the well-conditioned `Q_xy = 0` sub-model.
  Returns an `EMNoiseResult` (`Q`, per-iteration log-likelihood, `n_iter`,
  `converged`). `prg/tests/test_em_partial_noise.py`: 4 tests (block-diagonal
  recovery, full-joint likelihood maximisation, two validation cases).
- **Identifiability (documented + numerically verified).** Fixing `A` removes the
  EM *gauge* non-uniqueness but does **not** make the full joint `Q` identifiable:
  the cross-noise `Q_xy` is effectively non-identified from the hidden state (a
  near-flat likelihood ridge — the joint-`Q` Fisher information is numerically
  singular along it, and the EM endpoint tracks the initialisation rather than
  converging to the truth as `N` grows). Conditional on the diagonal blocks (or
  with X observed) `Q_xy` is sharply identified, and the `block_diagonal`
  sub-model recovers cleanly — hence the flag. The M-step algebra, the
  `Mk_smooth` transpose convention, the Y-collapse and the block-diagonal
  constraint were verified to machine precision by an independent adversarial
  review.

### Fixed
- **Particle-filter reproducibility.** `_BaseParticleFilter` now derives its
  particle RNG (initialisation / propagation / resampling) from `sKey` via an
  independent `numpy.SeedSequence` subsequence, so PF / PPF runs are
  reproducible at a fixed seed (previously a fresh `SeedGenerator()` made them
  non-reproducible even with `sKey`). The subsequence is decoupled from the
  trajectory-simulation RNG (no correlation between true-state noise and filter
  randomness); `sKey=None` keeps the non-reproducible draw for Monte-Carlo runs.

### Also shipped since 2.11.0 (previously committed, untagged)
- **Linear pairwise smoothers BF, MBF, MF** — complete the linear PKS family;
  each agrees with RTS to machine precision.
- **Variational (block-tridiagonal) pairwise smoother** (`method="VAR"`).
- **VAR lag-one cross-covariances `M_{n|N}`** (`Mk_smooth`) — the E-step input
  consumed by the partial EM above.
- **FFBSm backward-kernel fix** (pyproject `2.11.2`) — joint couple transition
  density, so PPS converges to PKS.

---

## [2.11.0] - 2026-05-29

### Added
- **Linear DWY smoother.** `prg/classes/linear_pks.py` is refactored into a
  multi-variant backward-smoother framework: `Linear_PKS` becomes a façade
  selecting the backward pass via `method=` (default `"RTS"` — fully backward
  compatible). The new `method="DWY"` runs the **Desai-Weinert-Yusypchuk**
  backward-filter recursion on the time-reversed complementary couple model
  (cf. Geng et al., 2023); on the linear-Gaussian model it returns the same
  smoothed mean and covariance as RTS to machine precision
  (`test_dwy_equals_rts`). Explicit `Linear_PKS_RTS` / `Linear_PKS_DWY`
  classes are exported, and the `_SMOOTHING_PASSES` registry is set up for
  further variants (BF / MBF / MF). `test_linear_pks.py`: 40 → 44 tests.

### Also shipped since 2.10.0 (previously committed but untagged)
- **`--config` CLI decoupled from the gitignored GUI** — `SessionConfig` +
  `load_config` / `save_config` moved to the tracked, PyQt-free
  `prg/utils/session.py` (`prg.gui.session` re-exports them), so
  `awesomepkf-<filter> --config foo.toml` works on a fresh clone / PyPI
  install. TOML read via stdlib `tomllib` (`tomli` fallback on 3.10).
- **nbmake notebook CI** (`pytest --nbmake notebooks/`) + a `notebooks`
  optional-dependency group.
- **SinCoupling demo** in tutorial_02; **tutorial_05** modernised to the
  `configs.py` registry API (it was broken); **tutorial_06** decoupled from
  `prg.gui`; notebook kernels normalised.

### Tests
- 321 tests pass (`prg/tests/`); ruff clean.

---

## [2.10.0] - 2026-05-29

### Added

- **`prg/learning/` — method-of-moments estimator for the 1D linear PMM.** Recovers the five parameters `(a, b, c, d, e)` from a two-column time series by computing the empirical 4×4 covariance of the lagged vector `(X_n, Y_n, X_{n+1}, Y_{n+1})`. `PMMParams` is a `NamedTuple` (iterable, hashable, immutable) with `as_hmm()` and `is_hmm()` helpers. `validate_pmm` enforces `|b|<1`, `|c|<1` and strict positive-definiteness on both Γ and the noise covariance `BBt`. `pmm_to_linear_params` returns `LinearAmQ`-compatible kwargs (`A`, `B`, `mQ`, `mz0`, `Pz0`) where `B` is the symmetric square root of `BBt` via eigendecomposition. Salvaged from the now-removed companion repository `KalmanApp`. Scope is intentionally limited to the linear scalar case — nonlinear identification for EPKF/UPKF needs a separate procedure (e.g. a neural network).
- **`awesomepkf-fit-pkf` CLI** (`prg/run_learn_pmm.py`) — fits the linear PMM on any CSV/TSV/Parquet/JSON/Excel file with `--x-col` and `--y-col` (positional index or column name). Optionally saves a `numpy.savez(..., allow_pickle=False)`-compatible `.npz` archive with the parameters, the resolved column names, and the two extracted series as a `float64` matrix.
- **Tutorial 08** — [`tutorial_08_real_data_pkf_learning.ipynb`](notebooks/tutorial_08_real_data_pkf_learning.ipynb): end-to-end on the WindFarms sample (load → fit → PMM vs HMM gap → convert to `LinearAmQ` kwargs).
- **WindFarms sample under [`data/samples/windfarms/`](data/samples/windfarms/)** — 586 hourly active-power and wind-speed readings (October 2022, one onshore site), shipped in both raw and standardised form. The complete dataset (BuildingTemp, SeattleTemp, multiple sites/granularities) stays outside the repository; the CLI's `--data-filename` accepts a path to any local copy.
- **`prg/tests/test_learning.py`** — 21 tests covering `PMMParams` unpacking, HMM projection, validation edge cases (`b = ±1`, `c = 1`), method-of-moments recovery on synthetic trajectories (`tol = 0.03` over 50 000 steps), DataFrame and ndarray inputs, symmetric square-root factorisation, real-CSV smoke test, and three CLI tests (happy path with `.npz` roundtrip, missing-file exit code, console-script registration).

---

## [2.9.0] - 2026-05-29

MINOR release: filter/smoother robustness hardening, a new nonlinear
model, and repository housekeeping on top of the v2.8.0 GUI milestone.
The gitignored PyQt6 GUI from v2.8.0 was re-verified against the hardened
filter layer and continues to work unchanged.

### Added
- **`model_x1_y1_SinCoupling_pairwise`** — a strongly-curved sinusoidal
  pairwise coupling model (bounded → stable, but with enough curvature
  over the filter spread that the unscented transform departs markedly
  from the first-order linearisation). Useful as an EPKF/UPKF (and
  EPKS/UPKS) divergence demonstration.
- **Sigma-set comparative guards** for the nonlinear unscented filters
  (UKF / UPKF), with dedicated tests.
- **`SCIENTIFIC_PRE_PUSH_CHECKLIST.md`** — pre-push checklist for
  numerically sensitive refactors.

### Changed / Hardened
- Hardened the sigma-point filter layer and harmonised runtime guards
  across the nonlinear and particle filters; refined the
  `matrix_diagnostics` package; completed Ruff hygiene.
- Added API-contract and finiteness / edge-case tests across the filter
  and smoother families. **Test count: 247 → 290, all passing.**
- Aligned `prg.__version__` with the packaged version (was stuck at
  `0.1.0`).

### Removed
- The auto-generated "Folders structure" tree in `README.md`, its
  `<!-- PROJECT_STRUCTURE_* -->` markers, the `update_readme_structure.sh`
  generator script, and the `pre-push` hook that regenerated it.

### Housekeeping
- Excluded `docs/` (local reference PDFs) from version control.
- The public filter / smoother API is unchanged; the gitignored GUI and
  its 5 smoother presets continue to work against the hardened base
  (verified headless).

---

## [2.8.0] - 2026-05-15

MINOR release: end-to-end GUI integration of the five smoothers shipped
in v2.2.0–v2.6.0. The PyQt6 GUI (kept under ``prg/gui/`` which is
gitignored — out-of-tree GUI surface for the project) now exposes every
smoother through a dedicated tab plus inline checkboxes in the existing
tabs. This release intentionally has **no committed source diff** in
``prg/`` proper: the smoother classes themselves were finalised in
2.2.0–2.6.3, the GUI wiring lives in the gitignored area, and the
tutorial / report were closed out in 2.6.3 / 2.7.0. The tag marks the
GUI completeness milestone.

### Added (gitignored ``prg/gui/`` tree — design summary only)

- **``prg/gui/smoother_mapping.py``** — pure-Python filter → smoother
  registry, with ``SMOOTHER_MAP``, ``SmootherSpec``, ``make_smoother``
  helper (dispatches per filter to the right constructor signature),
  ``supports_joseph``, and ``FILTER_SUPPORTS_SMOOTHER``. Imports the
  five smoother classes directly (``Linear_PKS``, ``NonLinear_EPKS``,
  ``NonLinear_UPKS``, ``NonLinear_UKS``, ``NonLinear_PPS``); ``pf`` is
  explicitly mapped to ``None`` and reports ``False`` in
  ``FILTER_SUPPORTS_SMOOTHER``.

- **``prg/gui/tabs/smoother.py``** — new dedicated **Smoother** tab.
  Filter dropdown is restricted to the five with smoothers
  (PKF → PKS, EPKF → EPKS, UPKF → UPKS, UKF → UKS, PPF → PPS). Joseph
  checkbox auto-disables for PPF. Always runs both passes and shows:
  - state-estimate plot overlaying filter and smoother traces with
    ±2σ envelopes,
  - posterior-covariance trace plot (``trace(P)``) side-by-side for
    both passes — the envelope shrinkage is visible at a glance,
  - metrics table comparing MSE / MAE / NEES / NIS for both passes,
  - **Save Session…** export — config, metrics CSV + LaTeX, history
    pickle, state and covariance plots.

- **``prg/gui/workers/filter_worker.py``** — extended ``FilterJob`` with
  ``smoother: bool`` + ``joseph: bool``. Worker now:
  - runs the forward filter via ``FilterRunner`` (unchanged),
  - if requested, instantiates the matching smoother through
    ``make_smoother(...)``, replays the *exact* forward trajectory
    via ``process_smoother(N=None, data_generator=replay())``,
  - merges ``Xkp1_smooth`` / ``PXXkp1_smooth`` / ``Gk_smooth`` /
    ``w_smooth`` into the forward history dicts so a single record
    stream feeds the plots,
  - emits a payload that includes ``smoother_used: bool`` plus
    ``smoother_metrics: {mse_total, mae_total, nees_mean, nis_mean}``.

- **``prg/gui/workers/sweep_worker.py``** — ``SweepJob`` extended with
  ``smoother`` / ``joseph`` flags. When set, each sweep run also fires
  the matching backward pass and the result dict gains
  ``mse_smooth_mean / mse_smooth_std / nees_smooth_mean /
  nees_smooth_std``. Added support for sweeping ``n_particles``
  directly (special-cased in ``_one_run``: passed to the runner
  constructor rather than as a model kwarg).

- **``prg/gui/widgets/filter_picker.py``** — two new checkboxes:
  *Also run smoother (backward pass)* (greyed out for PF, which has
  no smoother in this release) and *Joseph form (Kalman family only)*
  (greyed out for PF / PPF). ``also_smoother()`` and ``joseph_form()``
  accessors honour the enabled-state so callers never see a stale
  combination.

- **``prg/gui/session.py``** — ``SessionConfig`` extended with
  optional ``smoother: bool = False`` and ``joseph: bool = False``
  fields. Persisted to/from TOML under ``[filter].smoother`` /
  ``[filter].joseph`` so demo presets and saved sessions can pre-tick
  the smoother flow.

- **``prg/gui/tabs/single_run.py``** — propagates the new checkbox
  values into ``FilterJob``. When a smoother pass is included in the
  payload, the **State estimate** plot adds a green smoother trace +
  envelope alongside the orange filter trace, and the metrics table
  shows two columns (e.g. *PKF* + *PKS*).

- **``prg/gui/tabs/comparison.py``** — shared *Also run smoothers* +
  *Joseph form* toggles. PF jobs ignore the smoother flag gracefully
  (``FILTER_SUPPORTS_SMOOTHER`` gating). The metrics table reorders
  columns so each smoother sits next to its filter (PKF | PKS |
  EPKF | EPKS | …); the comparison plot uses dashed lines in the
  matching colour for smoother traces.

- **``prg/gui/tabs/sensitivity.py``** — same two checkboxes; sweepable
  parameters now include ``n_particles`` (integer) in addition to the
  three universal tuning knobs (alpha, beta, kappa); the MSE / NEES
  plots overlay dashed smoother curves alongside the solid filter ones.

- **``prg/gui/main_window.py``** — registers the new **Smoother** tab
  between *Single Run* and *Comparison*; ``Load config…`` now applies
  to all four downstream tabs; loading a preset with ``smoother=True``
  jumps to the Smoother tab instead of Single Run.

- **5 new demo presets** in ``prg/gui/presets/`` (auto-loaded by the
  Demo Gallery):
  - ``11_pks_linear_pairwise.toml`` — PKF + PKS reference run
    (linear pairwise model, paper §2.5).
  - ``12_epks_retroactions.toml`` — EPKF + EPKS on the retroactions
    pairwise model.
  - ``13_upks_retroactions.toml`` — UPKF + UPKS (Wan2000 sigma points)
    on the same model.
  - ``14_uks_multiplicative_augmented.toml`` — UKF + UKS on the
    augmented multiplicative model.
  - ``15_pps_retroactions.toml`` — PPF + PPS (FFBSm) on the
    retroactions model, 300 particles.

### Verification (headless, ``QT_QPA_PLATFORM=offscreen``)

- All six filter cases (PKF, EPKF, UPKF, UKF, PPF, PF) drive the worker
  cleanly with ``smoother=True``; PF gracefully reports
  ``smoother_used=False`` since no PF smoother is registered.
- PKF + PKS on a 50-step linear pairwise trajectory: filter MSE 0.0073,
  smoother MSE 0.0063 (≈ 13 % reduction — within the expected RTS
  envelope shrinkage).
- All 15 demo presets (10 legacy + 5 new) load through ``load_config``,
  apply to the SmootherTab, and route correctly through the
  ``MainWindow._apply_preset`` dispatch.
- Full ``MainWindow`` constructs with 5 tabs in the expected order.

### Tests

- No new tests (per user spec: "Sans tests" for the GUI surface).
- 247 ``prg/`` tests still pass; no code changes under ``prg/``
  proper (only under the gitignored ``prg/gui/`` subtree).

---

## [2.7.0] - 2026-05-15

MINOR release: adds the smoother tutorial and links it from the four
related forward-filter tutorials. Closes the documentation gap left
open by the v2.2.0–v2.6.3 sequence (5 smoothers shipped, 0 tutorials
until now).

### Added
- **`notebooks/tutorial_07_smoothers.ipynb`** — comprehensive
  walkthrough of the five smoothers, 28 cells (13 code + 15 markdown),
  ~5 MB executed with embedded plots. Sections:
  - The smoother family (summary table, shared API)
  - §1 Linear PKS on `model_x1_y1_AQ_pairwise` — exact RTS with ±2σ
    envelope shrinkage
  - §2 EPKS and UPKS on `model_x2_y1_pairwise` — mild non-linearity,
    overlay shows the two coincide
  - §3 UKS on `model_x1_y1_Sinus_classic` — strong non-linearity,
    structural `(p, p)` gain shape callout
  - §4 PPS (FFBSm) on `model_x2_y1_pairwise` — smoothed-particle
    cloud, effective sample size diagnostic
  - §5 Monte-Carlo convergence PPS → PKS on linear-Gaussian model
    (14-orders-of-magnitude separation log-log plot)
  - §6 Joseph form demonstration (standard vs Joseph agree to ~1e-15)
  - §7 Decision rule (which smoother for which scenario)
  - §8 Going Further (pointers to report Sections 2–7 and the
    `generate_comparison.py` reproducibility script)
- "Going Further" tables of tutorials 01, 02, 03, 04 now each carry a
  pointer to tutorial 07 with a one-line API hint
  (`from prg.classes.linear_pks import Linear_PKS`, etc.).
- README "Tutorials" table extended with row 07.

### Tests
- 247 tests still pass; no `prg/` code changes.

---

## [2.6.3] - 2026-05-15

PATCH release — documentation: adds a comparative analysis section
(Section 7) to the companion LaTeX report, with a single auto-running
reproducibility script.

### Added (report tree only — separate doc dir, not under git here)

- **Section 7 "Comparaison des cinq lisseurs"** in
  ``Report/NonLinearSmoothingReport/Sections/Section7_Comparison.tex``,
  ~5 PDF pages, covering:
  - §7.1 Applicability matrix per model class (linear-pairwise,
    NL-pairwise, NL-classical) explaining why no single experiment
    can run all 5 smoothers head-to-head.
  - §7.2 Cross-model × smoother MSE-ratio table averaged over 20 seeds.
  - §7.3 Single-trajectory overlay of the 4 smoothers applicable to
    a linear-pairwise model (PKS = EPKS = UPKS to 3 decimals;
    PPS within MC noise).
  - §7.4 Monte-Carlo convergence of PPS → PKS in log-log axes; EPKS
    and UPKS shown as horizontal reference lines at machine precision
    (~1e-16 to 1e-18). The 14-orders-of-magnitude separation between
    PPS and the Kalman family on linear-Gaussian models is the
    pedagogical highlight.
  - §7.5 Wall-clock timing table (forward / backward / total).
  - §7.6 Decision rule for choosing among the 5 smoothers.
  - §7.7 Reproducibility instructions.
- **`Report/.../Figures/generate_comparison.py`** — single 350-line
  script orchestrating the four experiments. Emits:
  - ``comparison_overlay.png``, ``comparison_mc_convergence.png``
  - ``comparison_cross_model.tex``, ``comparison_timing.tex`` —
    bare ``\begin{tabular}...\end{tabular}`` snippets to be
    ``\input``-ed in the section (the (β) strategy from the design
    proposal: numbers stay in sync with experiments automatically).
  - ``comparison_results.json`` for traceability across machines.
- All script outputs are parametrised via CLI arguments
  (``--N-overlay``, ``--N-mc``, ``--N-table``, ``--seed``,
  ``--n-seeds-table``, ``--n-particles``, ``--n-particles-list``,
  ``--n-reps-timing``). Defaults reproduce the numbers in the report.

### Validation
- 247 tests still pass; no code changes in ``prg/``.

---

## [2.6.2] - 2026-05-15

PATCH release: cross-cutting parity pass over the **5 smoothers** (the
v2.5.1 parity pass only covered the 4 Kalman-family classes; the PPS
added in v2.6.0 was outside its scope). Applies an audit punch list of
18 items — 4 HIGH, 8 MED, 6 LOW — against ``NonLinear_PPS`` and its
tests, plus one substantive robustness fix.

### Fixed (HIGH)
- **PSD safety check on the smoothed sample covariance** (`nonlinear_pps.py`
  ``_record_smoothed``). The PPS now calls ``self._check_covariance(cov,
  k, name="PXXkp1_smooth")`` before writing the result to the history,
  matching the four Kalman smoothers. A degenerate particle cloud
  (catastrophic ESS) producing a non-PSD weighted sample covariance is
  now caught (and Tikhonov-regularised if the parent's diagnostic
  decides to) rather than silently written.
- **``# NOTE:`` import block** on un-imported but re-raised exceptions,
  paritising the comment block present in the four Kalman smoothers.
- **``Raises`` section reformatted per-exception** in
  ``process_smoother`` (was a bundled one-line item). Now each of
  ``ParamError``, ``InvertibilityError``, ``CovarianceError``,
  ``StepValidationError``, ``NumericalError``, ``FilterError`` is
  listed with its trigger condition.
- **Terminal-step Rao-Blackwell caveat** explicitly documented in the
  class docstring (was only in the CHANGELOG): ``Xkp1_smooth[N]`` uses
  the raw cloud weighted by ``weights``, while PPF's ``Xkp1_update``
  uses ``Σ w_i μ'_x,i``. Both target the same posterior but differ by
  Monte-Carlo variance.

### Documentation (MED)
- ``Complexity`` heading renamed ``Cost`` for parity with UPKS / UKS
  docstrings.
- ``Numerical safeguards`` section added (LSE + degenerate-uniform
  fallback + PSD diagnostic now form a structured triple).
- ``verbose > 1`` ``rich_show_fields`` per-step display added, parallel
  to the four Kalman smoothers.
- ``process_N_data_smoother`` docstring expanded with the
  Exception-handling policy paragraph + ``Raises`` block (was a
  one-liner).
- ``History schema additions`` section now documents the ``particles``
  / ``weights`` keys added by the forward (PPF, with
  ``store_particles=True``), in addition to ``Xkp1_smooth``,
  ``PXXkp1_smooth``, ``w_smooth`` added by the backward.

### Tests
- **+3 tests in ``test_nonlinear_pps.py``**:
  - ``test_pkferror_root_catches_smoother_errors`` — name-paritised
    with the four Kalman smoothers.
  - ``test_singular_mQ_xx_raises_covariance_error`` — verifies the
    construction-time ``CovarianceError`` path with structured
    ``step=-1`` and ``matrix_name="mQ[:p,:p]"``.
  - ``test_degenerate_weight_fallback_logs_warning`` — monkeypatches
    ``logsumexp`` to force the weight-degeneracy fallback path and
    verifies the WARNING log + uniform-weights recovery.

### Code cleanup (LOW)
- Pairwise quadratic form in the backward kernel fused into a single
  ``np.einsum`` with ``optimize=True`` (was two sequential einsums).
- Per-iteration ``np.tile + np.concatenate`` hoisted into a
  pre-allocated ``z_buffer`` written by slice assignment.
- New ``_propagate_particles_at`` helper, paritising the
  ``_propagate_sigma_at`` / ``_propagate_sigma_f_at`` factoring of
  UPKS and UKS.
- Class docstring now cross-references
  ``Report/NonLinearSmoothingReport/Sections/Section6_PPS.tex``.
- ``ParamLinear`` / ``ParamNonLinear`` imports added for type hints
  (parity with the four Kalman smoothers; harmless previously, just
  inconsistent).
- ``test_nonlinear_pps.py`` gains a rationale comment explaining why
  the ``test_terminal_gain_is_zero_placeholder`` analog is absent
  (``Gk_smooth`` doesn't exist in particle smoothers — the
  ``w_smooth[N] == weights[N]`` invariant covers the equivalent
  boundary condition).

### Tests
- 247 tests pass (up from 244): +3 PPS exception-policy/logging tests,
  same parity invariants now enforced across all 5 smoothers.

---

## [2.6.1] - 2026-05-15

PATCH release: documentation-only. Audit pass over the companion
LaTeX report ``Report/NonLinearSmoothingReport/`` (separate doc tree,
not under git) which gained Section 6 (PPS / FFBSm) along with v2.6.0.

The report fixes do not touch any code or test; the audit caught a
LaTeX-vs-Markdown bug in Section 6.4 (the complexity table was written
in Markdown syntax inside a ``.tex`` file, rendering as literal ``|``
characters), a missing float wrapper on the Monte-Carlo convergence
table in Section 6.7, and several notation drift items across
Sections 2 / 4 / 5 / 6 (raw ``X, Y, Z, Q_x, W^x`` instead of the
project's ``\mX, \mY, \mZ, \QQ, \Vx{n}`` macros; raw
``\widehat{\mathbf{C}}_n`` instead of the ``\hmC_n`` macro; the
``Lisseur PPF particulaire`` title was renamed ``Lisseur PPS
particulaire`` to match the class name and all internal references).
Sections 6.2 in particular saw a math-notation overhaul: missing
``_{n+1}`` time-subscripts on ``\mB, \calQ`` were restored,
``\Sigma_{xx}`` is now explicitly defined as
``[\calQ_{n+1}]_{xx}`` rather than left implicit, and the bracket
projection convention is unified on lowercase ``_{xx}``.

This release exists solely to make the audit traceable from
``git log`` and to bump the version stamp.

### Tests
- Same as v2.6.0: 244 tests pass. No test changes.

---

## [2.6.0] - 2026-05-15

### Added

- **`NonLinear_PPS` — Pairwise Particle Smoother (FFBSm)** ([`prg/classes/nonlinear_pps.py`](prg/classes/nonlinear_pps.py)). Implements *Forward Filtering, Backward Smoothing* (Doucet et al. 2000) on top of the `NonLinear_PPF`. The forward pass runs the standard PPF with `store_particles=True` (forced internally); the backward pass reweights the forward particle cloud via a pairwise transition-density-based recursion. Complexity: O(N·n_p²) for the backward pass. The smoothed mean and covariance are weighted statistics of the forward cloud with the smoothed weights. Log-sum-exp normalisation guards against underflow when the forward and next-step clouds are far apart in state space; degenerate weight fallback follows the same `WARNING` pattern as the parent's `_safe_normalize_log_weights`.
- **`store_particles` flag on `_BaseParticleFilter`** (and propagated through `NonLinear_PPF.process_filter`). When `True`, the per-step particle cloud and weights are appended to each forward history record via the public `HistoryTracker.update_record` API. Default `False` (no history bloat for regular PF/PPF users); forced `True` by the PPS. Non-breaking additive change.
- **PPS report section** (`Report/NonLinearSmoothingReport/Sections/Section6_PPS.tex`) with FFBSm derivation for pairwise models, log-sum-exp stability discussion, complexity analysis, and the Monte-Carlo convergence table (PPS → PKS) on a linear-Gaussian pairwise model.
- **PPS figure generator** (`Report/.../Figures/generate_pps_figure.py`) parametrised by `--n-particles`.

### Caveats
- **No Joseph form** for the particle smoother — that is a Kalman-family numerical safeguard. The corresponding particle-smoother safeguard is the log-sum-exp normalisation, applied in the backward kernel.
- **No `Gk_smooth`** field in the history — particle smoothers don't have a gain matrix. Instead, a `w_smooth` field carries the smoothed weights (shape `(n_particles,)`).
- **Different terminal-step estimator from the PPF.** The PPS smoothed mean at step N uses the raw particle cloud weighted by `weights`, while the PPF `Xkp1_update` uses the Rao-Blackwellised estimator `Σ w_i μ'_x,i`. Both target the same posterior but differ by Monte-Carlo variance.

### Validation
- **Monte-Carlo convergence to PKS** measured on `model_x1_y1_AQ_pairwise` (linear-Gaussian pairwise model) at fixed seed: RMS deviation from the exact `Linear_PKS` shrinks from ~1.4e-2 (n_p=100) to ~6.7e-3 (n_p=2000), compatible with the standard `O(1/√n_p)` MC rate. Two pytest tests enforce the strict ordering and the absolute bound.

### Tests
- **+19 tests in `prg/tests/test_nonlinear_pps.py`**: shapes, weight normalisation, terminal `w_smooth = weights` boundary condition, Monte-Carlo convergence to `Linear_PKS` (two tests), edge cases (`N=1`, generator semantics, double-call, external `data_generator`, missing ground truth), exception policy, `caplog`-based logging, regression test on linear pairwise model.
- Total: 244 tests pass (up from 225).

---

## [2.5.1] - 2026-05-15

PATCH release: cross-cutting parity pass over the four smoother classes
(`Linear_PKS`, `NonLinear_EPKS`, `NonLinear_UPKS`, `NonLinear_UKS`)
following an independent code audit. No API changes, no math changes,
no behaviour changes — pure consistency cleanup.

### Documentation parity
- All four smoothers now have a uniform docstring structure: prose intro,
  ``Parameters`` section, ``Cost`` section (for UPKS/UKS, mentioning the
  per-step sigma-point regeneration overhead), and ``History schema
  additions`` section. Linear_PKS now exposes its ``Parameters``;
  UPKS and UKS now document the ``Xkp1_smooth / PXXkp1_smooth /
  Gk_smooth`` schema with their respective shapes — including the
  UKS-specific ``(dim_x, dim_x)`` gain shape (vs ``(dim_x, dim_xy)``
  for the pairwise variants) which is the defining structural difference.
- Joseph form precision claims aligned across all four docstrings to
  match the actual test tolerance (``~1e-10``) — previously claimed
  ``~1e-13`` / ``~1e-15`` machine precision, which was never enforced.
- UKS ``Raises`` section reformatted per-exception (one item per
  exception with a clear description), matching the PKS/EPKS/UPKS style.

### Test parity
- **+5 new tests across the three nonlinear smoothers** restoring
  feature parity with ``test_linear_pks.py``:
  - ``test_joseph_psd_shrinkage`` added to EPKS, UPKS, UKS (verifies that
    the Joseph form preserves the same PSD invariant as the standard form).
  - ``test_singular_predicted_covariance_raises`` added to UPKS and UKS
    via ``monkeypatch`` on ``cho_factor``: forces a backward Cholesky
    failure and verifies the structured ``CovarianceError`` (``matrix_name``,
    ``step``, chained ``__cause__``).
  - ``test_terminal_gain_is_zero_placeholder`` added to UKS (the pairwise
    smoothers already had it).
- Test name uniformisation: ``test_pkferror_base_class_catches_smoother_errors``
  renamed to ``test_pkferror_root_catches_smoother_errors`` in Linear_PKS
  to match the three nonlinear smoothers.

### Code cleanup
- Removed redundant ``Gn.copy()`` / ``Cn.copy()`` defensive calls in the
  four ``update_record(..., Gk_smooth=Gn)`` invocations — ``Gn`` is
  freshly produced by ``cho_solve`` at each iteration and never aliased
  back into the loop, so the copy was a no-op.
- UPKS ``_propagate_sigma_at`` no longer returns the unused ``Pa``
  augmented covariance (3-tuple → 2-tuple). Caller unpacks
  ``sigma_X, sigma_propag`` directly.
- UKS receives the missing ``# NOTE:`` comment block (parallel to the
  three other smoothers) documenting the exception types that propagate
  unwrapped from the forward but are not constructed locally.

### Tests
- 225 tests pass (up from 216): +9 tests for parity, no removed tests.

---

## [2.5.0] - 2026-05-15

### Added

- **`NonLinear_UKS` — Unscented Kalman Smoother (classical)** ([`prg/classes/nonlinear_uks.py`](prg/classes/nonlinear_uks.py)). Two-pass smoother extending `NonLinear_UKF`. Unlike the pairwise smoothers, the classical UKS operates on the X-only Markov chain (FxHx model): the smoothing gain is `(dim_x, dim_x)` (not `(dim_x, dim_xy)`) and sigma-points are generated in dimension `dim_x` only. Cross-covariance is estimated from sigma points regenerated at `(X_{n|n}, P^{xx}_{n|n})` and propagated through `f`; the predicted covariance is read from the forward record (already includes the additive `Q_x` term). Joseph form available via `joseph=True`. Compatible with all registered sigma-point sets.
- **UKS report section** (`Report/NonLinearSmoothingReport/Sections/Section5_UKS.tex`) with the classical RTS derivation in sigma-point form, comparison with the pairwise variants, and discussion of MSE sensitivity to the model's degree of nonlinearity (Sinus_classic vs x2y1_classic).
- **UKS figure generator** (`Report/.../Figures/generate_uks_figure.py`).

### Caveats
- **Pairwise models rejected.** The constructor inherited from `NonLinear_UKF` raises `FilterError` if `param.pairwiseModel=True`. Use the UPKS for pairwise models.
- **MSE improvement depends strongly on the curvature of `f` and `h`.** On `model_x1_y1_Sinus_classic` (strongly nonlinear), MSE ratio ≈ 0.91. On the milder `model_x2_y1_classic`, the ratio is ≈ 1.00 (trace ratio still ≈ 0.92, but the linearisation-bias on the empirical MSE absorbs the gain).

### Tests
- **+29 tests in `prg/tests/test_nonlinear_uks.py`**: shapes (including the dedicated test verifying `Gk_smooth` is `(dim_x, dim_x)`), terminal equality, PSD shrinkage, Joseph form equivalence, sigma-set parametric coverage, **pairwise-model guard**, edge cases, exception policy, `caplog`-based INFO/DEBUG emission, regression test on Sinus_classic (ratio < 0.97).
- Total: 216 tests pass (up from 187).

---

## [2.4.0] - 2026-05-15

### Added

- **`NonLinear_UPKS` — Unscented Pairwise Kalman Smoother** ([`prg/classes/nonlinear_upks.py`](prg/classes/nonlinear_upks.py)). Two-pass smoother extending `NonLinear_UPKF`. The backward pass regenerates sigma points at each step's filtered linearisation point `(X_{n|n}, y_n)` with augmented covariance `diag(P^{xx}_{n|n}, mQ)`, propagates them through `g` (with `y_n` inserted between state and noise blocks, mirroring the forward), and estimates the cross-covariance `Cov(X_n, Z_{n+1} | y_{1:n})` and the predicted joint covariance `P^{ZZ}_{n+1|n}` from the sample weighted moments (using `Wc` weights). Joseph form available via `joseph=True`. Compatible with all four registered sigma-point sets (`wan2000`, `cpkf`, `lerner2002`, `ito2000`).
- **UPKS report section** (`Report/NonLinearSmoothingReport/Sections/Section4_UPKS.tex`) with the sigma-point cross-covariance derivation, Joseph form, and §4.4 documenting the auto-consistency of the backward-recomputed `Zhat_{n+1|n}` and its alignment with the forward.
- **UPKS figure generator** (`Report/.../Figures/generate_upks_figure.py`) with the same auto-detected repo-root mechanism as the other smoothers; parametrised by `--sigma-set` to compare sigma-point variants.

### Caveats
- **Not suitable for augmented models.** Same rank-deficient predicted covariance issue as the EPKS.
- **Sigma-set invariance** on mildly nonlinear models: the three commonly-used sigma sets (`wan2000`, `cpkf`, `lerner2002`) produce numerically identical MSE statistics (3 significant digits) on `model_x2_y1_pairwise` and `model_x1_y1_pairwise`. Differences become observable on more strongly nonlinear transitions.

### Tests
- **+29 tests in `prg/tests/test_nonlinear_upks.py`**: shapes, terminal equality, PSD shrinkage on the sigma-point covariance, Joseph form equivalence, **sigma-set parametric coverage (3 sets)** including a `ParamError` on unknown set name, edge cases (`N=1`, generator semantics, double-call, external `data_generator`, missing ground truth), exception policy (`ParamError`, `PKFError` root), `caplog`-based INFO/DEBUG emission, and a regression test (UPKS MSE not significantly degrading vs UPKF, ratio < 1.02 on 20 seeds × N=300).
- Total: 187 tests pass (up from 158).

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

[Unreleased]: https://github.com/sderrode/awesomepkf/compare/v2.10.0...HEAD
[2.10.0]: https://github.com/sderrode/awesomepkf/compare/v2.9.0...v2.10.0
[0.4.0]: https://github.com/sderrode/awesomepkf/compare/v0.3.0...v0.4.0
[0.3.0]: https://github.com/sderrode/awesomepkf/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/sderrode/awesomepkf/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/sderrode/awesomepkf/releases/tag/v0.1.0
