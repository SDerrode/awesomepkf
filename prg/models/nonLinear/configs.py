"""
Registry of "simple" nonlinear-model configurations.

Holds the dim/form/symbolic-model triplet for the models whose only
per-model boilerplate was the standard ``_init_random_params`` call.

Models that are NOT in this registry stay as classes in their own files,
because they need one of:
- a non-default constructor (markov_naive, pairwise_param, multiplicative)
- a method override (x2_y1_classic._eval_H)
- substitution-based symbolic_model (the augmented variants)

Each entry is a :class:`NonLinearSpec`. Adding a new "simple" model is
one entry — no new file.
"""

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
import sympy as sp

from prg.utils.generate_matrix_cov import generate_block_matrix
from prg.utils.numerics import EPS_REL

__all__ = ["NONLINEAR_CONFIGS", "NonLinearSpec"]


@dataclass
class NonLinearSpec:
    """Configuration consumed by :class:`_FuncModelFxHx` / :class:`_FuncModelGxGy`."""

    dim_x: int
    dim_y: int
    form: str  # "fxhx" or "gxgy"
    symbolic_model: Callable  # (sx, [sy,] st, su) → (sfx, shx) or (sgx, sgy)
    val_max: float = 0.50
    init_hook: Callable | None = None  # custom (model) -> None for non-default mQ/mz0/Pz0
    attrs: dict = field(default_factory=dict)  # instance attrs to set on the model


# ======================================================================
# FxHx — classic models with f(x, t), h(x, u)
# ======================================================================


def _sm_cubique(sx, st, su):
    x, t, u = sx[0], st[0], su[0]
    return sp.Matrix([[0.9 * x - 0.6 * x**3 + t]]), sp.Matrix([[x + u]])


def _sm_sinus(sx, st, su):
    x, t, u = sx[0], st[0], su[0]
    return sp.Matrix([[0.05 * x + 2.0 * sp.sin(x) + t]]), sp.Matrix([[1.5 * sp.sin(x) + u]])


def _sm_expsaturant(sx, st, su):
    x, t, u = sx[0], st[0], su[0]
    sfx = 0.5 * x + 2.0 * (1 - sp.exp(-0.1 * x)) + t
    shx = sp.log(1 + sp.Max(sp.Abs(x), EPS_REL)) + u
    return sp.Matrix([[sfx]]), sp.Matrix([[shx]])


def _sm_gordon_factory(dt: float = 1.0) -> Callable:
    """Returns a symbolic_model closure with the dt parameter baked in."""
    cos_term = 8 * float(np.cos(1.2 * dt))

    def _sm(sx, st, su):
        x, t, u = sx[0], st[0], su[0]
        sfx = 0.5 * x + 25 * x / (1.0 + x**2) + cos_term + t
        shx = 0.05 * x**2 + u
        return sp.Matrix([[sfx]]), sp.Matrix([[shx]])

    return _sm


def _sm_rapport(sx, st, su):
    alpha, beta, gamma, kappa, dt = 0.5, 0.5, 0.5, 0.15, 0.1
    x1, x2 = sx[0], sx[1]
    t1, t2 = st[0], st[1]
    u = su[0]
    sfx1 = (1 - kappa) * x1 + dt * x2 + t1
    sfx2 = x2 - dt * (alpha * sp.sin(x1) + beta * x2) + t2
    shx = x1**2 / (1 + x1**2) + gamma * sp.sin(x2) + u
    return sp.Matrix([sfx1, sfx2]), sp.Matrix([shx])


def _init_rapport(model) -> None:
    """Rapport_classic uses two different val_max values for mQ and Pz0."""
    rng = model._randMatrices.rng
    model.mQ = generate_block_matrix(rng, model.dim_x, model.dim_y, 0.1)
    model.mz0 = rng.standard_normal((model.dim_xy, 1))
    model.Pz0 = generate_block_matrix(rng, model.dim_x, model.dim_y, 0.05)


# ======================================================================
# GxGy — pairwise models with gx(x, y, t), gy(x, y, u)
# ======================================================================


def _sm_x1_y1_pairwise(sx, sy, st, su):
    a, b, c, d = 0.50, 3, 0.40, 2
    x, y, t, u = sx[0], sy[0], st[0], su[0]
    sgx = sp.Matrix([a * x + b * sp.tanh(y) + t])
    sgy = sp.Matrix([c * y + d * sp.sin(x / 20) + u])
    return sgx, sgy


def _sm_x2_y1_pairwise(sx, sy, st, su):
    a, b, c, d, e, f = 0.95, 0.10, 0.05, 0.9, 0.30, 0.6
    x1, x2 = sx[0], sx[1]
    y1 = sy[0]
    t1, t2 = st[0], st[1]
    u = su[0]
    sgx = sp.Matrix([
        a * x1 + b * x2 + c * sp.tanh(y1) + t1,
        d * x2 + e * sp.sin(y1) + t2,
    ])
    sgy = sp.Matrix([x1**2 / (1 + x1**2) + f * y1 + u])
    return sgx, sgy


def _sm_x2_y2_pairwise(sx, sy, st, su):
    KAPPA = 0.15
    x1, x2 = sx[0], sx[1]
    y1, y2 = sy[0], sy[1]
    t1, t2 = st[0], st[1]
    u1, u2 = su[0], su[1]
    sgx = sp.Matrix([
        (1 - KAPPA) * x1 + sp.Rational(1, 10) * x2 * sp.tanh(y1) + t1,
        sp.Rational(9, 10) * x2 + sp.Rational(1, 10) * sp.sin(x1) + t2,
    ])
    sgy = sp.Matrix([
        x1 - sp.Rational(3, 10) * y2 + u1,
        x2 + sp.Rational(3, 10) * y1 + u2,
    ])
    return sgx, sgy


# Lotka-Volterra needs both a custom symbolic_model (with class consts)
# AND a custom init_hook for mQ/mz0 around the equilibrium point.
_LV = {
    "ALPHA": 0.00312, "BETA": 0.00014, "GAMMA": 0.02534, "DELTA": 0.01175,
    "SIGMAX": 0.39668, "SIGMAY": 0.43850, "DT": 1.0,
}


def _sm_lotka_volterra(sx, sy, st, su):
    A, B, G, D, DT = _LV["ALPHA"], _LV["BETA"], _LV["GAMMA"], _LV["DELTA"], _LV["DT"]
    x, y, t, u = sx[0], sy[0], st[0], su[0]
    y_det = y * sp.exp((D * x - G) * DT)
    sgx = sp.Matrix([x * sp.exp((A - B * y_det) * DT + t)])
    sgy = sp.Matrix([y_det * sp.exp(u)])
    return sgx, sgy


def _init_lotka_volterra(model) -> None:
    x_eq = _LV["GAMMA"] / _LV["DELTA"]
    y_eq = _LV["ALPHA"] / _LV["BETA"]
    model.mQ, model.mz0, model.Pz0 = model._init_random_params(
        model.dim_x, model.dim_y, 0.10, seed=None
    )
    model.mz0 = np.array([[x_eq], [y_eq]])
    model.mQ = np.diag([_LV["SIGMAX"], _LV["SIGMAY"]])


# ======================================================================
# Registry
# ======================================================================


NONLINEAR_CONFIGS: dict[str, NonLinearSpec] = {
    # --- FxHx classic ---
    "model_x1_y1_Cubique_classic": NonLinearSpec(
        dim_x=1, dim_y=1, form="fxhx", symbolic_model=_sm_cubique,
    ),
    "model_x1_y1_Sinus_classic": NonLinearSpec(
        dim_x=1, dim_y=1, form="fxhx", symbolic_model=_sm_sinus,
    ),
    "model_x1_y1_ExpSaturant_classic": NonLinearSpec(
        dim_x=1, dim_y=1, form="fxhx", symbolic_model=_sm_expsaturant,
    ),
    "model_x1_y1_Gordon_classic": NonLinearSpec(
        dim_x=1, dim_y=1, form="fxhx", symbolic_model=_sm_gordon_factory(dt=1.0),
    ),
    "model_x2_y1_Rapport_classic": NonLinearSpec(
        dim_x=2, dim_y=1, form="fxhx", symbolic_model=_sm_rapport,
        init_hook=_init_rapport,
    ),

    # --- GxGy pairwise ---
    "model_x1_y1_pairwise": NonLinearSpec(
        dim_x=1, dim_y=1, form="gxgy", symbolic_model=_sm_x1_y1_pairwise,
    ),
    "model_x2_y1_pairwise": NonLinearSpec(
        dim_x=2, dim_y=1, form="gxgy", symbolic_model=_sm_x2_y1_pairwise, val_max=0.15,
    ),
    "model_x2_y2_pairwise": NonLinearSpec(
        dim_x=2, dim_y=2, form="gxgy", symbolic_model=_sm_x2_y2_pairwise, val_max=0.15,
    ),
    "model_x1_y1_LotkaVolterra_pairwise": NonLinearSpec(
        dim_x=1, dim_y=1, form="gxgy",
        symbolic_model=_sm_lotka_volterra,
        init_hook=_init_lotka_volterra,
    ),
}
