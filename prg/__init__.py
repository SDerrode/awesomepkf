"""
AwesomePKF — Pairwise Kalman Filter and variants.

Public API
----------
Filters:
    Linear_PKF      — Linear Pairwise Kalman Filter
    NonLinear_EPKF  — Extended Pairwise Kalman Filter
    NonLinear_UPKF  — Unscented Pairwise Kalman Filter
    NonLinear_UKF   — Unscented Kalman Filter (adapted for pairwise models)
    NonLinear_PPF   — Pairwise Particle Filter
    NonLinear_PF    — Bootstrap Particle Filter

Parameters:
    ParamLinear     — Parameter object for linear models
    ParamNonLinear  — Parameter object for nonlinear models

Model factories:
    ModelFactoryLinear    — Discover and instantiate linear models by name
    ModelFactoryNonLinear — Discover and instantiate nonlinear models by name

Example
-------
>>> from prg import Linear_PKF, ParamLinear
>>> from prg.models.linear import ModelFactoryLinear
>>> model = ModelFactoryLinear.create("model_x1_y1_AQ_pairwise")
>>> params = model.get_params()
>>> dim_x, dim_y = params.pop("dim_x"), params.pop("dim_y")
>>> param = ParamLinear(0, dim_x, dim_y, **params)
>>> pkf = Linear_PKF(param, sKey=42)
>>> results = pkf.process_N_data(N=100)
"""

__version__ = "0.1.0"

from prg.classes.Linear_PKF import Linear_PKF
from prg.classes.NonLinear_EPKF import NonLinear_EPKF
from prg.classes.NonLinear_UPKF import NonLinear_UPKF
from prg.classes.NonLinear_UKF import NonLinear_UKF
from prg.classes.NonLinear_PPF import NonLinear_PPF
from prg.classes.NonLinear_PF import NonLinear_PF
from prg.classes.ParamLinear import ParamLinear
from prg.classes.ParamNonLinear import ParamNonLinear
from prg.models.linear import ModelFactoryLinear
from prg.models.nonLinear import ModelFactoryNonLinear

__all__ = [
    "__version__",
    # Filters
    "Linear_PKF",
    "NonLinear_EPKF",
    "NonLinear_UPKF",
    "NonLinear_UKF",
    "NonLinear_PPF",
    "NonLinear_PF",
    # Parameters
    "ParamLinear",
    "ParamNonLinear",
    # Model factories
    "ModelFactoryLinear",
    "ModelFactoryNonLinear",
]
