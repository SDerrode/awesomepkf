"""Targeted tests for sigma-point set implementations."""

from __future__ import annotations

import numpy as np
import pytest

from prg.classes.sigma_points_set import (
    SetCPKF,
    SetIto2000,
    SetLERNER2002,
    SetWAN2000,
    SigmaPointsSet,
)
from prg.models.nonlinear import ModelFactoryNonLinear
from prg.tests.conftest import make_param_nonlinear
from prg.utils.exceptions import CovarianceError


def _make_param():
    model = ModelFactoryNonLinear.create("model_x2_y1_pairwise")
    return make_param_nonlinear(model)


@pytest.mark.parametrize(
    "cls, key",
    [
        (SetWAN2000, "wan2000"),
        (SetCPKF, "cpkf"),
        (SetLERNER2002, "lerner2002"),
        (SetIto2000, "ito2000"),
    ],
)
def test_registry_contains_expected_sigma_sets(cls, key):
    assert SigmaPointsSet.registry[key] is cls


@pytest.mark.parametrize("sigma_cls", [SetWAN2000, SetCPKF, SetLERNER2002, SetIto2000])
def test_sigma_point_shapes_for_dim_2(sigma_cls):
    param = _make_param()
    sigma = sigma_cls(dim=2, param=param)
    x = np.zeros((2, 1))
    P = np.eye(2)
    points = sigma._sigma_point(x, P)
    assert points.shape == (sigma.nbSigmaPoint, 2, 1)


@pytest.mark.parametrize("sigma_cls", [SetWAN2000, SetCPKF, SetLERNER2002, SetIto2000])
def test_weight_sum_is_one(sigma_cls):
    param = _make_param()
    sigma = sigma_cls(dim=2, param=param)
    assert np.isclose(np.sum(sigma.Wm), 1.0, atol=1e-12)


@pytest.mark.parametrize("sigma_cls", [SetWAN2000, SetCPKF, SetLERNER2002, SetIto2000])
def test_cholesky_failure_raises_covariance_error(sigma_cls):
    param = _make_param()
    sigma = sigma_cls(dim=2, param=param)
    x = np.zeros((2, 1))
    # Strongly indefinite matrix: should fail even with EPS regularization.
    bad_P = np.array([[1.0, 0.0], [0.0, -1.0]])
    with pytest.raises(CovarianceError):
        sigma._sigma_point(x, bad_P)
