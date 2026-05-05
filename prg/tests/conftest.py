"""Shared pytest fixtures and helpers."""

import pytest

from prg.classes.param_linear import ParamLinear
from prg.classes.param_nonlinear import ParamNonLinear
from prg.models.linear import ModelFactoryLinear
from prg.models.nonLinear import ModelFactoryNonLinear

SEED = 42
N_SHORT = 100
N_CALIB = 500


def make_param_linear(model, verbose=0):
    """Build a ParamLinear from a linear model instance."""
    params = model.get_params().copy()
    dim_x = params.pop("dim_x")
    dim_y = params.pop("dim_y")
    return ParamLinear(verbose, dim_x, dim_y, **params)


def make_param_nonlinear(model, verbose=0):
    """Build a ParamNonLinear from a nonlinear model instance."""
    params = model.get_params().copy()
    dim_x = params.pop("dim_x")
    dim_y = params.pop("dim_y")
    return ParamNonLinear(verbose, dim_x, dim_y, **params)


@pytest.fixture(scope="session")
def model_x1y1():
    return ModelFactoryLinear.create("model_x1_y1_AQ_pairwise")


@pytest.fixture(scope="session")
def model_x2y2():
    return ModelFactoryLinear.create("model_x2_y2_AQ_pairwise")


@pytest.fixture(scope="session")
def model_nl_x2y1():
    return ModelFactoryNonLinear.create("model_x2_y1_pairwise")


@pytest.fixture(scope="session")
def model_nl_x1y1():
    return ModelFactoryNonLinear.create("model_x1_y1_pairwise")


@pytest.fixture(scope="session")
def model_nl_classic_x1y1():
    """A classic (non-pairwise) FxHx model — required by the bootstrap PF
    which doesn't accept pairwise models."""
    return ModelFactoryNonLinear.create("model_x1_y1_Sinus_classic")


@pytest.fixture(scope="session")
def model_nl_classic_x2y1():
    """A 2-D classic FxHx model — used to exercise the UKF and PF on
    higher-dim states without the pairwise back-action."""
    return ModelFactoryNonLinear.create("model_x2_y1_classic")


@pytest.fixture(scope="session")
def param_x1y1(model_x1y1):
    return make_param_linear(model_x1y1)


@pytest.fixture(scope="session")
def param_x2y2(model_x2y2):
    return make_param_linear(model_x2y2)


@pytest.fixture(scope="session")
def param_nl_x2y1(model_nl_x2y1):
    return make_param_nonlinear(model_nl_x2y1)


@pytest.fixture(scope="session")
def param_nl_x1y1(model_nl_x1y1):
    return make_param_nonlinear(model_nl_x1y1)


@pytest.fixture(scope="session")
def param_nl_classic_x1y1(model_nl_classic_x1y1):
    return make_param_nonlinear(model_nl_classic_x1y1)


@pytest.fixture(scope="session")
def param_nl_classic_x2y1(model_nl_classic_x2y1):
    return make_param_nonlinear(model_nl_classic_x2y1)
