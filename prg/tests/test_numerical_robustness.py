"""Regression tests for three numerical-robustness fixes (release 2.16.0).

1. The shared 2F/DWY backward filter re-symmetrises its covariances each step;
   without it a skew rounding error can grow geometrically and make P^b diverge.
2. The determinant check of ``InvertibleMatrix`` is invariant to scaling and to
   dimension, so a well-conditioned matrix is never declared singular because
   it is small or large.
3. In-place covariance regularisation is recorded per run and can be disabled
   (``strict_covariance``).
"""

import numpy as np
import pytest
from scipy.linalg import solve_discrete_lyapunov, sqrtm

from prg.classes.linear_pks import (
    Linear_PKS_DWY,
    Linear_PKS_MBF,
    Linear_PKS_MF,
    Linear_PKS_RTS,
)
from prg.classes.matrix_diagnostics.invertible import InvertibleMatrix
from prg.classes.matrix_diagnostics.status import Status
from prg.classes.param_linear import ParamLinear
from prg.models.linear._amq import LinearAmQ
from prg.utils.exceptions import CovarianceError

A_21 = np.array([[0.50, 0.60, 0.30], [-0.05, 0.40, 0.25], [0.30, 0.20, 0.20]])
CXX = np.array([[1.0, 0.3], [0.3, 1.0]])
K_21 = np.array([[1.0], [1.0]]) / np.sqrt(2.0)


def _noise(c):
    """Pair noise with canonical correlation c between the X and Y channels."""
    Cxy = c * np.real(sqrtm(CXX)) @ K_21
    return np.block([[CXX, Cxy], [Cxy.T, np.eye(1)]])


def _param(A, R, P0, dx, dy):
    m = LinearAmQ(dx, dy, A=A, mQ=R, mz0=np.zeros((dx + dy, 1)), Pz0=P0,
                  pairwiseModel=True)
    p = m.get_params().copy()
    p.pop("dim_x")
    p.pop("dim_y")
    return ParamLinear(0, dx, dy, **p)


def _smooth(cls, param, seed=1, N=400):
    s = cls(param, sKey=seed)
    s.process_N_data_smoother(N=N)
    X = np.array([np.asarray(r["Xkp1_smooth"], float).ravel() for r in s.history])
    P = np.array([np.asarray(r["PXXkp1_smooth"], float) for r in s.history])
    return s, X, P


class TestBackwardFilterSymmetrisation:
    """Strongly correlated noise, p=2, q=1: the unsymmetrised backward filter
    diverged here (P^b relative error ~1e40) and 2F/DWY aborted."""

    @pytest.fixture
    def param_corr(self):
        R = _noise(0.99)
        return _param(A_21, R, solve_discrete_lyapunov(A_21, R), 2, 1)

    @pytest.mark.parametrize("cls", [Linear_PKS_MF, Linear_PKS_DWY])
    def test_two_filter_and_dwy_match_rts(self, cls, param_corr):
        _, Xr, Pr = _smooth(Linear_PKS_RTS, param_corr)
        _, X, P = _smooth(cls, param_corr)
        assert np.max(np.abs(X - Xr)) / np.max(np.abs(Xr)) < 1e-10
        assert np.max(np.abs(P - Pr)) / np.max(np.abs(Pr)) < 1e-10


def _det_status(M):
    report = InvertibleMatrix(M).check()
    det = next(c for c in report.checks if "determinant" in c.name.lower())
    return report.overall_status, det.status


def _equicorrelated(n, rho):
    return (1.0 - rho) * np.eye(n) + rho * np.ones((n, n))


class TestDeterminantCheck:

    M3 = np.array([[2.0, 0.3, 0.1], [0.3, 1.5, 0.2], [0.1, 0.2, 1.0]])

    @pytest.mark.parametrize("scale", [1.0, 1e-6, 1e-12, 1e6, 1e-170, 1e160])
    def test_scale_invariant(self, scale):
        overall, det = _det_status(scale * self.M3)
        assert det == Status.OK
        assert overall != Status.FAIL

    @pytest.mark.parametrize("n, rho", [(8, 0.999), (15, 0.8), (30, 0.5)])
    def test_dimension_invariant(self, n, rho):
        # well-conditioned (cond <= ~8e3) matrices of growing size
        overall, det = _det_status(_equicorrelated(n, rho))
        assert det != Status.FAIL
        assert overall != Status.FAIL

    def test_redundant_sensors_warn_not_fail(self):
        overall, det = _det_status(100.0 * np.ones((3, 3)) + 1e-6 * np.eye(3))
        assert det != Status.FAIL
        assert overall == Status.WARNING     # flagged by the condition number

    def test_singular_matrix_fails(self):
        overall, det = _det_status(np.array([[1.0, 2.0], [2.0, 4.0]]))
        assert det == Status.FAIL and overall == Status.FAIL

    def test_zero_row_fails(self):
        overall, det = _det_status(np.array([[1.0, 0.0], [0.0, 0.0]]))
        assert det == Status.FAIL and overall == Status.FAIL

    def test_filter_runs_with_six_correlated_sensors(self):
        # p=2, q=6, observation noises sharing one factor: cond(S_n) ~ 1e4
        rng = np.random.default_rng(3)
        A = rng.normal(size=(8, 8))
        A *= 0.6 / max(abs(np.linalg.eigvals(A)))
        R = np.eye(8)
        R[2:, 2:] = np.ones((6, 6)) + 1e-3 * np.eye(6)
        param = _param(A, R, solve_discrete_lyapunov(A, R), 2, 6)
        s = Linear_PKS_RTS(param, sKey=0)
        s.process_N_data_smoother(N=50)
        assert len(s.history) == 51


class TestCovarianceRegularisationTrace:

    @pytest.fixture
    def param_broad_prior(self):
        # Prior 1e10 I against unit noise: MBF's covariance P - P Lambda P is a
        # difference of nearly equal matrices and comes out clearly indefinite
        # (min eigenvalue O(-10) or below) in the first steps.
        return _param(A_21, _noise(0.5), 1e10 * np.eye(3), 2, 1)

    @pytest.fixture
    def rts(self, param_x1y1):
        return Linear_PKS_RTS(param_x1y1, sKey=42)

    def test_check_covariance_records_regularisation(self, rts):
        mat = np.diag([1.0, -1e-3])
        rts._check_covariance(mat, 7, name="X")
        (event,) = rts.covariance_regularisations
        assert event["step"] == 7 and event["name"] == "X"
        assert event["min_eig_before"] < 0.0 <= event["min_eig_after"]
        assert set(event) == {"step", "name", "eps", "min_eig_before", "min_eig_after"}

    def test_check_covariance_strict_raises(self, rts):
        rts.strict_covariance = True
        with pytest.raises(CovarianceError):
            rts._check_covariance(np.diag([1.0, -1e-3]), 7, name="X")
        assert rts.covariance_regularisations == []

    def test_mbf_broad_prior_is_recorded(self, param_broad_prior):
        s, _, _ = _smooth(Linear_PKS_MBF, param_broad_prior)
        events = [e for e in s.covariance_regularisations if e["name"] == "PXXkp1_smooth"]
        assert events
        assert min(e["min_eig_before"] for e in events) < -1.0

    def test_mbf_broad_prior_strict_raises(self, param_broad_prior):
        s = Linear_PKS_MBF(param_broad_prior, sKey=1)
        s.strict_covariance = True
        with pytest.raises(CovarianceError):
            s.process_N_data_smoother(N=400)

    def test_records_are_reset_between_runs(self, param_broad_prior):
        s = Linear_PKS_MBF(param_broad_prior, sKey=1)
        s.process_N_data_smoother(N=100)
        first = len(s.covariance_regularisations)
        s.process_N_data_smoother(N=100)
        assert first > 0
        assert len(s.covariance_regularisations) == first

    def test_default_is_not_strict(self, param_broad_prior):
        assert Linear_PKS_MBF(param_broad_prior, sKey=1).strict_covariance is False

    def test_well_posed_run_records_nothing(self, rts):
        rts.process_N_data_smoother(N=100)
        assert rts.covariance_regularisations == []
