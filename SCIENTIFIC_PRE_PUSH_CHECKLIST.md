# Scientific Pre-Push Checklist

Use this checklist before pushing refactors in numerically sensitive modules.

## Scope and Risk

- [ ] Confirm touched modules (`prg/classes`, `prg/models`, `prg/utils`) and impacted tests.
- [ ] Mark numerical-risk changes explicitly (covariance, inversion, Cholesky, sigma-points, particle weights).
- [ ] Keep changes atomic and API-compatible unless a behavior change is intended.

## Validation Sequence

- [ ] Run targeted tests for touched files.
- [ ] Run full suite: `pytest`.
- [ ] Run lint: `ruff check .`.

## Numerical Safety Gates

- [ ] Check smoothed covariances are finite and PSD in relevant tests.
- [ ] Check smoothing gains (`Gk_smooth`) are finite where applicable.
- [ ] For particle smoothers, check `w_smooth` validity (finite, non-negative, sum to 1).
- [ ] Check filter state estimates are finite (`Xkp1_predict`, `Xkp1_update`) on touched filters.
- [ ] Add or update tests for any new edge-case path.
- [ ] For runtime wrappers, assert step+phase error messages in targeted tests.

## Change Delivery

- [ ] Summarize what changed and why (short and review-friendly).
- [ ] List residual risks and assumptions.
- [ ] Commit only intended files (exclude `.claude/`, local docs/scratch by default).
