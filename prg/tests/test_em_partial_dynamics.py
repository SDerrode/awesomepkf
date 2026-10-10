"""Tests for the partial-EM dynamics estimator (:mod:`prg.learning.em_partial_dynamics`).

Learns the couple-defining blocks ``A_xy`` (back-action) and ``A_yy`` (observation
memory) with ``A_xx``, ``A_yx``, ``Q`` fixed, and tests for back-action by a
likelihood-ratio test.

Data are generated through :class:`LinearAmQ` so the simulator's transition
callables are consistent with the ``A`` *matrix*: overwriting the ``A`` attribute of
a factory model would leave the forward callables (used by the simulator and the
forward filter) on the model's default transition, silently generating data from the
wrong ``A``.
"""

from __future__ import annotations

import itertools

import numpy as np
import pytest
from scipy.optimize import minimize

from prg.classes.linear_pkf import Linear_PKF
from prg.classes.linear_pks import Linear_PKS
from prg.classes.param_linear import ParamLinear
from prg.learning.em_partial_dynamics import (
    BackActionLRT,
    EMDynamicsResult,
    _y_loglik,
    back_action_lrt,
    estimate_dynamics_em,
)
from prg.learning.em_partial_noise import _gaussian_loglik
from prg.models.linear._amq import LinearAmQ
from prg.utils.exceptions import ParamError

SEED = 7
Q_TRUE = np.array([[0.10, 0.05], [0.05, 0.10]])
CLASSICAL_INIT = [[0.6, 0.0], [0.3, 0.0]]           # A_xy = A_yy = 0
TRUE_COUPLE = [[0.6, 0.4], [0.3, 0.4]]              # A_xy = A_yy = 0.4
TRUE_NO_BACKACTION = [[0.6, 0.0], [0.3, 0.4]]       # A_xy = 0, A_yy = 0.4


def _lin_param(A, Q=Q_TRUE, dim_x=1, dim_y=1) -> ParamLinear:
    dz = dim_x + dim_y
    model = LinearAmQ(
        dim_x, dim_y, A=np.asarray(A, float),
        mQ=0.5 * (Q + Q.T) + 1e-9 * np.eye(dz),
        mz0=np.zeros((dz, 1)), Pz0=np.eye(dz), pairwiseModel=True,
    )
    kw = model.get_params().copy()
    kw.pop("dim_x")
    kw.pop("dim_y")
    return ParamLinear(0, dim_x, dim_y, **kw)


def _is_monotone(ll: list[float]) -> bool:
    return all(b >= a - 1e-6 for a, b in itertools.pairwise(ll))


def test_recovers_couple_coefficients() -> None:
    """From the classical init (0, 0), EM recovers A_xy and A_yy near the truth."""
    sim = Linear_PKF(_lin_param(TRUE_COUPLE), sKey=SEED).simulate_N_data(1500)
    res = estimate_dynamics_em(_lin_param(CLASSICAL_INIT), sim, max_iter=200, tol=1e-4)

    assert isinstance(res, EMDynamicsResult)
    assert _is_monotone(res.loglik), res.loglik
    assert res.converged
    assert abs(res.A_xy.item() - 0.4) < 0.08, res.A_xy
    assert abs(res.A_yy.item() - 0.4) < 0.06, res.A_yy
    # clearly moved off the classical initialisation
    assert abs(res.A_xy.item()) > 0.2 and abs(res.A_yy.item()) > 0.2
    # the fixed blocks are untouched
    assert np.allclose(res.A[:1, :1], [[0.6]])
    assert np.allclose(res.A[1:, :1], [[0.3]])


def test_restricted_fit_holds_back_action_at_zero() -> None:
    """learn_back_action=False keeps A_xy at its initial 0 while learning A_yy."""
    sim = Linear_PKF(_lin_param(TRUE_COUPLE), sKey=SEED).simulate_N_data(1200)
    res = estimate_dynamics_em(
        _lin_param(CLASSICAL_INIT), sim,
        learn_back_action=False, learn_obs_memory=True, max_iter=200, tol=1e-4,
    )
    assert res.A_xy.item() == 0.0
    assert abs(res.A_yy.item()) > 0.2


def test_lrt_detects_back_action() -> None:
    """LRT rejects H0 when back-action is present, and yields a much smaller
    statistic when it is absent (dof = dim_x * dim_y = 1)."""
    sim_ba = Linear_PKF(_lin_param(TRUE_COUPLE), sKey=SEED).simulate_N_data(1500)
    sim_no = Linear_PKF(_lin_param(TRUE_NO_BACKACTION), sKey=SEED).simulate_N_data(1500)

    lrt_ba = back_action_lrt(_lin_param(CLASSICAL_INIT), sim_ba, max_iter=200, tol=1e-4)
    lrt_no = back_action_lrt(_lin_param(CLASSICAL_INIT), sim_no, max_iter=200, tol=1e-4)

    assert isinstance(lrt_ba, BackActionLRT)
    assert lrt_ba.dof == 1
    assert lrt_ba.stat >= 0.0 and lrt_no.stat >= 0.0
    # back-action present -> reject H0 decisively
    assert lrt_ba.pvalue < 0.01, lrt_ba
    # and much more significant than the no-back-action record
    assert lrt_ba.stat > lrt_no.stat
    assert lrt_no.pvalue > lrt_ba.pvalue


def test_rejects_short_data() -> None:
    with pytest.raises(ParamError):
        estimate_dynamics_em(_lin_param(CLASSICAL_INIT), [(0, None, np.zeros((1, 1)))])


# ---------------------------------------------------------------------------
# Restricted M-step and LRT against direct maximisation of the y-likelihood
# ---------------------------------------------------------------------------

def _ref_y_loglik(Y, A, Q, p, q) -> float:
    """Independent pairwise Kalman filter: log-likelihood of y_{1:N} given y_0,
    z_0 ~ N(0, I) (the library's convention: the y_0 prior term is left out)."""
    m, P, ll = np.zeros(p + q), np.eye(p + q), 0.0
    for n, y in enumerate(Y):
        if n > 0:
            m, P = A @ m, A @ P @ A.T + Q
            S = P[p:, p:]
            r = y - m[p:]
            ll -= 0.5 * (q * np.log(2 * np.pi) + np.linalg.slogdet(S)[1]
                         + r @ np.linalg.solve(S, r))
        K = P[:, p:] @ np.linalg.inv(P[p:, p:])
        m, P = m + K @ (y - m[p:]), P - K @ P[p:, :]
        m[p:], P[p:, :], P[:, p:] = y, 0.0, 0.0
    return ll


def _ref_max(Y, A, Q, p, q, rows, starts):
    """Maximise _ref_y_loglik by BFGS over A[rows, p:] from each start."""
    shape = A[rows, p:].shape

    def neg(t):
        At = A.copy()
        At[rows, p:] = t.reshape(shape)
        with np.errstate(all="ignore"):
            ll = _ref_y_loglik(Y, At, Q, p, q)
        return -ll if np.isfinite(ll) else 1e300

    res = [minimize(neg, np.ravel(s), method="BFGS", options={"gtol": 1e-8}) for s in starts]
    best = min(res, key=lambda r: r.fun)
    return best.x, -best.fun


def _records_to_Y(records, dim_y):
    return np.array([np.asarray(r[2], float).reshape(dim_y) for r in records])


def test_y_loglik_matches_filter_history() -> None:
    """The fast likelihood used by the LRT equals the library filter's."""
    param = _lin_param(TRUE_COUPLE)
    sim = list(Linear_PKF(param, sKey=SEED).simulate_N_data(200))
    smoother = Linear_PKS(param, method="VAR")
    smoother.process_N_data_smoother(N=None, data_generator=iter(sim))
    ll_lib = _gaussian_loglik(smoother.history)
    ll_fast = _y_loglik(_records_to_Y(sim, 1), param.A, param.mQ, param.mz0, param.Pz0, 1)
    assert abs(ll_fast - ll_lib) < 1e-8
    assert abs(_ref_y_loglik(_records_to_Y(sim, 1), param.A, param.mQ, 1, 1) - ll_lib) < 1e-8


@pytest.mark.parametrize("seed", [0, 3])
def test_restricted_em_fixed_point_is_restricted_mle(seed) -> None:
    """A_xy held at 0 with Q_xy != 0: the EM fixed point for A_yy is the MLE of
    the restricted model (the OLS M-step is not, since the noise is correlated)."""
    sim = list(Linear_PKF(_lin_param(TRUE_NO_BACKACTION), sKey=seed).simulate_N_data(500))
    init = _lin_param(CLASSICAL_INIT)
    res = estimate_dynamics_em(init, sim, learn_back_action=False, max_iter=300, tol=1e-13)
    Y = _records_to_Y(sim, 1)
    Qe = np.asarray(init.mQ, float)
    A_ref = np.asarray(init.A, float)
    theta, ll_ref = _ref_max(Y, A_ref, Qe, 1, 1, slice(1, None), [[0.0]])
    A_em = A_ref.copy()
    A_em[1, 1] = res.A_yy.item()
    assert res.A_xy.item() == 0.0
    assert abs(res.A_yy.item() - theta.item()) < 1e-4, (res.A_yy, theta)
    assert 2.0 * (ll_ref - _ref_y_loglik(Y, A_em, Qe, 1, 1)) < 1e-6


A_X2Y1 = np.array([[0.50, 0.10, 0.00],
                   [0.00, 0.40, 0.00],
                   [0.30, 0.20, 0.40]])
Q_X2Y1 = np.array([[0.10, 0.02, 0.01],
                   [0.02, 0.10, 0.01],
                   [0.01, 0.01, 0.10]])


@pytest.mark.parametrize("seed", [0, 3])
def test_lrt_matches_direct_maximisation(seed) -> None:
    """On a p=2, q=1 null model with a weakly observable A_xy, the LRT equals the
    statistic of the directly maximised likelihoods (EM alone stops short)."""
    param = _lin_param(A_X2Y1, Q=Q_X2Y1, dim_x=2, dim_y=1)
    sim = list(Linear_PKF(param, sKey=seed).simulate_N_data(400))
    Y = _records_to_Y(sim, 1)
    A0, Qe = np.asarray(param.A, float), np.asarray(param.mQ, float)
    ayy0, ll0 = _ref_max(Y, A0, Qe, 2, 1, slice(2, None), [A0[2:, 2:]])
    _, ll1 = _ref_max(Y, A0, Qe, 2, 1, slice(None), [np.r_[0.0, 0.0, ayy0]])
    ref = 2.0 * (ll1 - ll0)

    lrt = back_action_lrt(param, sim)
    assert lrt.dof == 2
    assert abs(lrt.loglik_restricted - ll0) < 1e-6
    assert lrt.stat == pytest.approx(ref, abs=1e-5, rel=1e-5)
