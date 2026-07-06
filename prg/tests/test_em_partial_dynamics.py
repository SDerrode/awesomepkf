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

from prg.classes.linear_pkf import Linear_PKF
from prg.classes.param_linear import ParamLinear
from prg.learning.em_partial_dynamics import (
    BackActionLRT,
    EMDynamicsResult,
    back_action_lrt,
    estimate_dynamics_em,
)
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
