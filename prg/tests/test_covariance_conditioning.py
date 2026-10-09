"""Regression tests: a large condition number never invalidates a covariance.

Up to 2.16.1, ``CovarianceMatrix`` declared any covariance with condition number
>= 1e12 invalid, although it was positive definite. Consequences: a model with
such a noise covariance or prior was refused at construction, and the filter
"regularised" such covariances in place by adding eps * I with
eps = 10 * (lambda_max / 1e12 - lambda_min), which under a broad prior swamps
the well-determined directions (smoothed covariances wrong by O(1)).
"""

import numpy as np
import pytest

from prg.classes.linear_pks import Linear_PKS_RTS, Linear_PKS_VAR
from prg.classes.matrix_diagnostics import CovarianceMatrix, Status, cholesky_eps
from prg.classes.param_linear import ParamLinear
from prg.models.linear._amq import LinearAmQ

# Textbook model x_{n+1} = F x_n + w, y_n = H x_n + v (non-normal F, rho(F) < 1),
# written as a pairwise chain: A = [[F, 0], [H F, 0]],
# R = [[Q, Q H^T], [H Q, H Q H^T + Rv]].
F = np.array([[-2.381, -3.475], [1.445, 2.4]])
H = np.array([[0.343, 0.121], [0.655, -0.052]])
Q = np.array([[1.0, 0.253], [0.253, 1.0]])
RV = np.array([[1.0, -0.618], [-0.618, 1.0]])


def _pairwise(rv_scale=1.0):
    A = np.block([[F, np.zeros((2, 2))], [H @ F, np.zeros((2, 2))]])
    R = np.block([[Q, Q @ H.T], [H @ Q, H @ Q @ H.T + rv_scale * RV]])
    return A, 0.5 * (R + R.T)


def _param(A, R, P0):
    m = LinearAmQ(2, 2, A=A, mQ=R, mz0=np.zeros((4, 1)), Pz0=P0, pairwiseModel=True)
    p = m.get_params().copy()
    p.pop("dim_x")
    p.pop("dim_y")
    return ParamLinear(0, 2, 2, **p)


def _smooth(cls, param, N=200):
    s = cls(param, sKey=3)
    s.process_N_data_smoother(N=N)
    P = np.array([np.asarray(r["PXXkp1_smooth"], float) for r in s.history])
    return s, P


def test_ill_conditioned_pd_covariance_is_valid_and_not_regularised():
    M = np.diag([1.0, 1e-13])
    report = CovarianceMatrix(M).check()
    assert report.is_valid
    assert report.overall_status == Status.WARNING
    result = CovarianceMatrix(M).regularize()
    assert result.eps_applied == 0.0
    np.testing.assert_array_equal(result.matrix_regularized, M)


def test_negative_eigenvalue_is_still_invalid_and_regularised():
    M = np.diag([1.0, -1e-3])
    report = CovarianceMatrix(M).check()
    assert not report.is_valid
    assert "eigenvalue" in report.failure_messages.lower()
    assert CovarianceMatrix(M).regularize().min_eigenvalue_after > 0.0


def test_cholesky_eps_makes_matrix_factorisable():
    w = np.array([1.0, 1e-17, -1e-18])
    V = np.linalg.qr(np.random.default_rng(0).standard_normal((3, 3)))[0]
    M = (V * w) @ V.T
    eps = cholesky_eps(M)
    assert 0.0 < eps < 1e-13
    np.linalg.cholesky(0.5 * (M + M.T) + eps * np.eye(3))


def test_model_with_ill_conditioned_noise_is_accepted():
    """Nearly noiseless measurements: cond(R) > 1e12, R positive definite."""
    A, R = _pairwise(rv_scale=1e-12)
    assert np.linalg.cond(R) > 1e12
    assert np.linalg.eigvalsh(R).min() > 0.0
    _param(A, R, np.eye(4))  # must not raise


@pytest.mark.parametrize("cls", [Linear_PKS_RTS])
def test_broad_prior_is_not_regularised_and_rts_matches_var(cls):
    """Broad prior P0 = 1e10 I: the filter covariances are positive definite with
    condition number > 1e12. 2.16.1 regularised three of them in place and the RTS
    smoothed covariance was wrong by O(1) at the first steps (relative error 3.3
    against a 60-digit reference)."""
    A, R = _pairwise()
    param = _param(A, R, 1e10 * np.eye(4))
    s, P = _smooth(cls, param)
    assert s.covariance_regularisations == []
    _, Pv = _smooth(Linear_PKS_VAR, param)
    err = max(np.linalg.norm(P[n] - Pv[n], 2) / np.linalg.norm(Pv[n], 2) for n in range(len(P)))
    assert err < 1e-3
