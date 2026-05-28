"""Targeted tests for matrix diagnostic helpers."""

from __future__ import annotations

import numpy as np
import pytest

from prg.classes.matrix_diagnostics import (
    CovarianceMatrix,
    InvertibleMatrix,
    StabilityMatrix,
    Status,
)


def test_covariance_matrix_regularize_fixes_negative_eigenvalue() -> None:
    # Symmetric but not PSD (one negative eigenvalue).
    M = np.array([[1.0, 2.0], [2.0, 1.0]])

    cov = CovarianceMatrix(M)
    report_before = cov.check()
    assert report_before.overall_status == Status.FAIL

    result = cov.regularize()
    assert result.is_success
    assert result.min_eigenvalue_after > 0.0


def test_invertible_matrix_inverse_matches_numpy() -> None:
    M = np.array([[2.0, 0.5], [0.5, 1.5]])
    inv_diag = InvertibleMatrix(M).inverse()
    inv_np = np.linalg.inv(M)
    np.testing.assert_allclose(inv_diag, inv_np, rtol=1e-12, atol=1e-12)


def test_invertible_matrix_inverse_uses_cache_on_second_call() -> None:
    M = np.array([[2.0, 0.5], [0.5, 1.5]])
    diag = InvertibleMatrix(M)
    inv_first = diag.inverse()
    inv_second = diag.inverse()
    # Same object identity proves we reused the internal cache.
    assert inv_first is inv_second


def test_invertible_matrix_inverse_raises_for_singular_matrix() -> None:
    singular = np.array([[1.0, 2.0], [2.0, 4.0]])
    with pytest.raises(RuntimeError, match="diagnostic FAILED"):
        InvertibleMatrix(singular).inverse()


def test_stability_matrix_fails_for_unstable_eigenvalue() -> None:
    unstable = np.array([[1.2, 0.0], [0.0, 0.8]])
    report = StabilityMatrix(unstable).check()
    assert report.overall_status == Status.FAIL


def test_stability_matrix_accepts_strictly_stable_matrix() -> None:
    stable = np.array([[0.6, 0.1], [0.0, 0.7]])
    report = StabilityMatrix(stable).check()
    assert report.is_valid
    assert StabilityMatrix(stable).spectral_radius() < 1.0


def test_covariance_matrix_warns_for_near_singular_psd() -> None:
    near_singular = np.diag([1.0, 1e-11])
    report = CovarianceMatrix(near_singular).check()
    assert report.overall_status == Status.WARNING
    assert report.is_valid


def test_invertible_matrix_warns_for_high_condition_number() -> None:
    poorly_conditioned = np.array([[1.0, 0.0], [0.0, 1e-8]])
    report = InvertibleMatrix(poorly_conditioned).check()
    assert report.overall_status == Status.WARNING
    assert report.is_valid


def test_invertible_matrix_inverse_emits_warning_on_warning_status() -> None:
    poorly_conditioned = np.array([[1.0, 0.0], [0.0, 1e-8]])
    diag = InvertibleMatrix(poorly_conditioned)
    with pytest.warns(UserWarning, match="WARNING status"):
        inv = diag.inverse()
    np.testing.assert_allclose(inv, np.linalg.inv(poorly_conditioned), rtol=1e-12, atol=1e-12)


def test_stability_matrix_warns_for_near_unit_spectral_radius() -> None:
    marginal = np.diag([0.995, 0.3])
    report = StabilityMatrix(marginal).check()
    assert report.overall_status == Status.WARNING
    assert report.is_valid


def test_covariance_matrix_regularize_uses_explicit_eps() -> None:
    M = np.array([[1.0, 2.0], [2.0, 1.0]])
    eps = 5.0
    result = CovarianceMatrix(M).regularize(eps=eps)
    expected = (M + M.T) / 2 + eps * np.eye(2)
    np.testing.assert_allclose(result.matrix_regularized, expected, rtol=0.0, atol=1e-12)
    assert result.eps_applied == eps


def test_covariance_regularized_matches_regularize_output() -> None:
    M = np.array([[1.0, 2.0], [2.0, 1.0]])
    cov = CovarianceMatrix(M)
    direct = cov.regularized()
    detailed = cov.regularize().matrix_regularized
    np.testing.assert_allclose(direct, detailed, rtol=0.0, atol=1e-12)
