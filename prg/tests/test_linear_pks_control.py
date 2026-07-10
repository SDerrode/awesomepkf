"""Tests for the deterministic control input (``consigne``) ``u_n`` of the
linear pairwise Kalman smoothers.

The model with control is the linear-Gaussian couple ``Z = (X, Y)``::

    Z_{n+1} = A Z_n + G u_n + B W_n,   W_n ~ N(0, mQ)

with ``u_n`` a KNOWN control and ``G`` of shape ``(dim_xy, dim_u)`` (control on
the whole couple). Because a deterministic control shifts only the *means* and
never the covariances, the six smoother variants incorporate it by
mean-trajectory superposition — exact for the linear-Gaussian model.

The four assertions form the correctness gate:

(a) the six variants agree (with the control) to ~1e-9;
(b) the control is *actually used* — ignoring it (``u=None``) on the same
    control-driven data gives a far larger error vs the truth;
(c) the smoothed covariances are identical with and without the control
    (control shifts means, not covariances);
(d) backward-compat — with ``u=None`` the output matches a plain run on
    control-free data.
"""

from __future__ import annotations

import numpy as np

from prg.classes.linear_pkf import Linear_PKF
from prg.classes.linear_pks import Linear_PKS
from prg.classes.param_linear import ParamLinear
from prg.models.linear import ModelFactoryLinear

SEED = 123
N = 1500
METHODS = ["RTS", "BF", "MBF", "MF", "2F", "DWY", "VAR"]  # "2F" is the alias of "MF"

# Control matrix on the whole couple (dim_xy=2, dim_u=1).
G = np.array([[0.6], [0.3]])

EQ_TOL = 1e-9       # cross-variant agreement / covariance equality
PD_MQ = np.array([[0.10, 0.02], [0.02, 0.08]])   # PD joint process noise


def _make_param(with_control: bool) -> ParamLinear:
    """ParamLinear for model_x1_y1_AQ_pairwise (dim_x=dim_y=1) with B=I, a PD
    mQ and, optionally, the control matrix ``G``."""
    params = ModelFactoryLinear.create("model_x1_y1_AQ_pairwise").get_params().copy()
    dim_x, dim_y = params.pop("dim_x"), params.pop("dim_y")
    params["B"] = np.eye(dim_x + dim_y)
    params["mQ"] = PD_MQ.copy()
    if with_control:
        params["G"] = G.copy()
    return ParamLinear(0, dim_x, dim_y, **params)


def _control_sequence(n: int) -> np.ndarray:
    """A non-trivial control: a step followed by a slow sinusoidal ramp,
    shape ``(n, dim_u=1)``."""
    t = np.arange(n)
    u = np.where(t < n // 3, 1.5, 0.0)              # step on for the first third
    u = u + 0.8 * np.sin(2.0 * np.pi * t / 200.0)   # sinusoidal component
    u = u + 0.003 * t                                # slow linear ramp
    return u.reshape(n, 1)


def _x_true(sim: list) -> np.ndarray:
    return np.array([rec[1].reshape(-1) for rec in sim])  # (R, dim_x)


def _x_smooth(res: list) -> np.ndarray:
    return np.array([row[5].reshape(-1) for row in res])  # (R, dim_x)


def _mse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean((a - b) ** 2))


def test_equivalence_with_control() -> None:
    """(a) All six variants give the same X_smooth trajectory (~1e-9) when the
    control is supplied."""
    param = _make_param(with_control=True)
    u = _control_sequence(N)
    sim = Linear_PKF(param, sKey=SEED).simulate_N_data(N, u=u)

    ref = None
    max_diff = 0.0
    for m in METHODS:
        res = Linear_PKS(param, method=m).process_N_data_smoother(
            N=None, data_generator=iter(sim), u=u
        )
        xs = _x_smooth(res)
        if ref is None:
            ref = xs
        else:
            max_diff = max(max_diff, float(np.max(np.abs(xs - ref))))
    assert max_diff < EQ_TOL, f"cross-variant max diff = {max_diff:.2e}"


def test_control_is_actually_used() -> None:
    """(b) Key test: smoothing the control-driven data WITH the control beats
    smoothing it while IGNORING the control by a wide margin."""
    param = _make_param(with_control=True)
    u = _control_sequence(N)
    sim = Linear_PKF(param, sKey=SEED).simulate_N_data(N, u=u)
    x_true = _x_true(sim)

    res_with = Linear_PKS(param, method="RTS").process_N_data_smoother(
        N=None, data_generator=iter(sim), u=u
    )
    res_without = Linear_PKS(param, method="RTS").process_N_data_smoother(
        N=None, data_generator=iter(sim), u=None
    )

    mse_with = _mse(_x_smooth(res_with), x_true)
    mse_without = _mse(_x_smooth(res_without), x_true)

    assert mse_with < 0.25 * mse_without, (
        f"control not exploited: mse_with={mse_with:.4e} "
        f"mse_without={mse_without:.4e} (ratio={mse_with / mse_without:.3f})"
    )


def test_covariances_unchanged_by_control() -> None:
    """(c) PXXkp1_smooth from the with-u run equals the without-u run (~1e-9):
    a deterministic control shifts means, not covariances."""
    param = _make_param(with_control=True)
    u = _control_sequence(N)
    sim = Linear_PKF(param, sKey=SEED).simulate_N_data(N, u=u)

    s_with = Linear_PKS(param, method="RTS")
    s_with.process_N_data_smoother(N=None, data_generator=iter(sim), u=u)
    s_without = Linear_PKS(param, method="RTS")
    s_without.process_N_data_smoother(N=None, data_generator=iter(sim), u=None)

    max_diff = 0.0
    for r1, r2 in zip(s_with.history, s_without.history, strict=True):
        max_diff = max(
            max_diff,
            float(np.max(np.abs(r1["PXXkp1_smooth"] - r2["PXXkp1_smooth"]))),
        )
    assert max_diff < EQ_TOL, f"covariance max diff = {max_diff:.2e}"


def test_backward_compat_u_none() -> None:
    """(d) With u=None the smoother output is identical to a plain run on
    control-free data (the control path is fully opt-in)."""
    # Control-free model and data (G absent entirely).
    param_plain = _make_param(with_control=False)
    sim_plain = Linear_PKF(param_plain, sKey=SEED).simulate_N_data(N)

    res_a = Linear_PKS(param_plain, method="RTS").process_N_data_smoother(
        N=None, data_generator=iter(sim_plain)
    )
    res_b = Linear_PKS(param_plain, method="RTS").process_N_data_smoother(
        N=None, data_generator=iter(sim_plain), u=None
    )

    # Also: a model that HAS G but is called with u=None must be unchanged too.
    param_G = _make_param(with_control=True)
    res_c = Linear_PKS(param_G, method="RTS").process_N_data_smoother(
        N=None, data_generator=iter(sim_plain), u=None
    )

    xa, xb, xc = _x_smooth(res_a), _x_smooth(res_b), _x_smooth(res_c)
    assert np.array_equal(xa, xb)
    assert np.array_equal(xa, xc)
