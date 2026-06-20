"""Tests for the partial EM noise estimator (prg.learning.em_partial_noise).

With ``A`` fixed, the *full* joint ``Q`` is only **weakly** identified through the
hidden state — the cross-noise ``Q_xy`` sits on a near-flat likelihood ridge and
is recovered very slowly in ``N`` (see the module docstring). So we test two
regimes separately:

* ``block_diagonal=True`` — the **well-identified** sub-model (``Q_xy = 0``):
  ``Q_xx`` and ``Q_yy`` recover cleanly at modest ``N`` → tight recovery test.
* full joint ``Q`` — correctness is asserted via the **likelihood** (monotone
  increase, and the EM reaches a ``Q`` at least as likely as the truth), *not*
  tight recovery (which is unrealistic at finite ``N``).
"""

from __future__ import annotations

import itertools

import numpy as np
import pytest

from prg.classes.linear_pkf import Linear_PKF
from prg.classes.linear_pks import Linear_PKS
from prg.classes.param_linear import ParamLinear
from prg.learning.em_partial_noise import (
    EMNoiseResult,
    _gaussian_loglik,
    estimate_noise_em,
)
from prg.models.linear import ModelFactoryLinear
from prg.utils.exceptions import ParamError

SEED = 7


def _make_param(Q: np.ndarray) -> ParamLinear:
    """ParamLinear for model_x1_y1_AQ_pairwise with a known mQ=Q and B=I."""
    params = ModelFactoryLinear.create("model_x1_y1_AQ_pairwise").get_params().copy()
    dim_x, dim_y = params.pop("dim_x"), params.pop("dim_y")
    params["B"] = np.eye(dim_x + dim_y)
    params["mQ"] = np.asarray(Q, dtype=float)
    return ParamLinear(0, dim_x, dim_y, **params)


def _is_monotone(ll: list[float]) -> bool:
    return all(b >= a - 1e-6 for a, b in itertools.pairwise(ll))


def test_block_diagonal_recovers_known_Q() -> None:
    """Well-identified regime: block-diagonal truth, block-diagonal estimate."""
    Q_true = np.array([[0.10, 0.0], [0.0, 0.07]])
    param = _make_param(Q_true)
    sim = Linear_PKF(param, sKey=SEED).simulate_N_data(1500)

    res = estimate_noise_em(
        param, sim, Q_init=np.eye(2), block_diagonal=True, max_iter=60, tol=5e-5
    )

    assert isinstance(res, EMNoiseResult)
    assert _is_monotone(res.loglik), res.loglik
    # cross-block held exactly at zero by the constraint
    assert res.Q[0, 1] == 0.0 and res.Q[1, 0] == 0.0
    # diagonal blocks recovered, and clearly moved off the identity start
    assert np.allclose(res.Q, Q_true, atol=0.03), f"got\n{res.Q}\nwant\n{Q_true}"
    assert not np.allclose(res.Q, np.eye(2), atol=0.03)
    # symmetric PSD
    assert np.allclose(res.Q, res.Q.T, atol=1e-12)
    assert np.all(np.linalg.eigvalsh(res.Q) > -1e-10)


def test_full_joint_maximizes_likelihood() -> None:
    """Full joint Q is weakly identified: assert the EM maximizes the observed
    log-likelihood (monotone, and reaches a Q at least as likely as the truth),
    not tight recovery."""
    Q_true = np.array([[0.10, 0.03], [0.03, 0.07]])
    param = _make_param(Q_true)
    sim = Linear_PKF(param, sKey=SEED).simulate_N_data(800)

    # observed-data log-likelihood at the TRUE Q (single smoother pass)
    ks = Linear_PKS(param, method="VAR")
    ks.process_N_data_smoother(N=None, data_generator=iter(sim))
    ll_true = _gaussian_loglik(ks.history)

    res = estimate_noise_em(param, sim, Q_init=np.eye(2), max_iter=55, tol=1e-6)

    assert _is_monotone(res.loglik), res.loglik
    # EM reaches a Q at least as likely as the ground truth
    assert res.loglik[-1] >= ll_true - 1e-6, (res.loglik[-1], ll_true)
    # the cross-block is free here (not forced to zero) -> generally nonzero
    assert abs(res.Q[0, 1]) > 1e-9
    assert np.all(np.linalg.eigvalsh(res.Q) > -1e-10)


def test_rejects_short_data() -> None:
    param = _make_param(np.eye(2) * 0.1)
    with pytest.raises(ParamError):
        estimate_noise_em(param, [(0, None, np.zeros((1, 1)))])


def test_rejects_bad_Q_init() -> None:
    param = _make_param(np.eye(2) * 0.1)
    sim = Linear_PKF(param, sKey=SEED).simulate_N_data(50)
    with pytest.raises(ParamError):
        estimate_noise_em(param, sim, Q_init=np.eye(3))  # wrong shape
