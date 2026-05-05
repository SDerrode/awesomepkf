"""
Linear models package.

Models are now declared as entries in :data:`LINEAR_CONFIGS` (see
:mod:`prg.models.linear.configs`); the previous one-class-per-file
layout was retired in favour of the registry.

Usage
-----
::

    from prg.models.linear import ModelFactoryLinear
    model = ModelFactoryLinear.create("model_x1_y1_AQ_classic")
"""

import numpy as np

from prg.models.linear.base_model_linear import BaseModelLinear, LinearAmQ, LinearSigma
from prg.models.linear.configs import LINEAR_CONFIGS

__all__ = ["LINEAR_CONFIGS", "ModelFactoryLinear"]


# ----------------------------------------------------------------------
# Builder functions, one per variant
# ----------------------------------------------------------------------


def _build_AQ_classic(cfg: dict) -> LinearAmQ:
    dim_x, dim_y = cfg["dim_x"], cfg["dim_y"]
    F, C, H, D = cfg["F"], cfg["C"], cfg["H"], cfg["D"]

    mQ, mz0, Pz0 = LinearAmQ._init_random_params(
        dim_x, dim_y, val_max=cfg["val_max"]
    )
    if cfg.get("diag_only", False):
        mQ = np.diag(np.diag(mQ))

    A = np.block([
        [F, np.zeros((dim_x, dim_y))],
        [H @ F, np.zeros((dim_y, dim_y))],
    ])
    B = np.block([
        [C, np.zeros((dim_x, dim_y))],
        [H @ C, D],
    ])

    return LinearAmQ(
        dim_x=dim_x, dim_y=dim_y,
        A=A, mQ=mQ, mz0=mz0, Pz0=Pz0, B=B,
        pairwiseModel=False,
    )


def _build_AQ_pairwise(cfg: dict) -> LinearAmQ:
    dim_x, dim_y = cfg["dim_x"], cfg["dim_y"]
    mQ, mz0, Pz0 = LinearAmQ._init_random_params(
        dim_x, dim_y, val_max=cfg["val_max"]
    )
    return LinearAmQ(
        dim_x=dim_x, dim_y=dim_y,
        A=cfg["A"], mQ=mQ, mz0=mz0, Pz0=Pz0,
        pairwiseModel=True,
    )


def _build_AQ_augmented(cfg: dict) -> LinearAmQ:
    base = ModelFactoryLinear.create(cfg["based_on"])
    # classic2pairwise does not depend on instance state — call via a
    # transient instance so we can reuse its logic without a static refactor
    args = LinearAmQ.__new__(LinearAmQ).classic2pairwise(base)
    return LinearAmQ(*args, augmented=True, pairwiseModel=False)


def _build_Sigma_pairwise(cfg: dict) -> LinearSigma:
    return LinearSigma(
        dim_x=cfg["dim_x"], dim_y=cfg["dim_y"],
        sxx=cfg["sxx"], syy=cfg["syy"],
        a=cfg["a"], b=cfg["b"], c=cfg["c"], d=cfg["d"], e=cfg["e"],
        pairwiseModel=True,
    )


_BUILDERS = {
    "AQ_classic":     _build_AQ_classic,
    "AQ_pairwise":    _build_AQ_pairwise,
    "AQ_augmented":   _build_AQ_augmented,
    "Sigma_pairwise": _build_Sigma_pairwise,
}


# ----------------------------------------------------------------------
# Public factory
# ----------------------------------------------------------------------


class ModelFactoryLinear:
    """Registry-driven factory for linear models."""

    @classmethod
    def create(cls, name: str, **kwargs) -> BaseModelLinear:
        """Build a linear model.

        Linear models are entirely config-driven: there is no constructor
        scalar to override at build time. ``kwargs`` is therefore not
        consumed during construction; the runner applies surviving keys as
        plain attributes on the resulting instance (this is how the
        Sensitivity tab sweeps universal UPKF / UKF tuning knobs such as
        ``alpha`` / ``beta`` / ``kappa`` on linear models).
        """
        key = name.strip()
        if key not in LINEAR_CONFIGS:
            raise ValueError(
                f"Unknown model: '{key}'. "
                f"Available: {cls.list_models()}"
            )
        cfg = LINEAR_CONFIGS[key]
        variant = cfg["variant"]
        if variant not in _BUILDERS:
            raise ValueError(
                f"Unknown variant '{variant}' for model '{key}'. "
                f"Supported: {sorted(_BUILDERS)}"
            )
        instance = _BUILDERS[variant](cfg)
        instance.MODEL_NAME = key
        # Soft post-construction overrides (only attributes that already exist
        # on the instance are touched; unknown names are silently ignored).
        for k, v in kwargs.items():
            if hasattr(instance, k):
                setattr(instance, k, v)
        return instance

    @classmethod
    def list_models(cls) -> list[str]:
        return sorted(LINEAR_CONFIGS.keys())
