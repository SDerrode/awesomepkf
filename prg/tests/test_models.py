"""Tests for linear and nonlinear model instantiation and structure."""

import numpy as np
import pytest

from prg.models.linear.model_x1_y1_AQ_pairwise import Model_x1_y1_AQ_pairwise
from prg.models.linear.model_x1_y1_AQ_classic import Model_x1_y1_AQ_classic
from prg.models.linear.model_x2_y2_AQ_pairwise import Model_x2_y2_AQ_pairwise
from prg.models.linear.model_x3_y1_AQ_pairwise import Model_x3_y1_AQ_pairwise
from prg.models.nonLinear.model_x2_y1_pairwise import Model_x2_y1_pairwise
from prg.models.nonLinear.model_x1_y1_pairwise import Model_x1_y1_pairwise as NL_x1y1


class TestLinearModels:

    @pytest.mark.parametrize("ModelCls, dim_x, dim_y", [
        (Model_x1_y1_AQ_pairwise, 1, 1),
        (Model_x1_y1_AQ_classic,  1, 1),
        (Model_x2_y2_AQ_pairwise, 2, 2),
        (Model_x3_y1_AQ_pairwise, 3, 1),
    ])
    def test_dimensions(self, ModelCls, dim_x, dim_y):
        m = ModelCls()
        assert m.dim_x == dim_x
        assert m.dim_y == dim_y
        assert m.dim_xy == dim_x + dim_y

    @pytest.mark.parametrize("ModelCls", [
        Model_x1_y1_AQ_pairwise,
        Model_x2_y2_AQ_pairwise,
        Model_x3_y1_AQ_pairwise,
    ])
    def test_matrix_shapes(self, ModelCls):
        m = ModelCls()
        dxy = m.dim_xy
        assert m.A.shape  == (dxy, dxy)
        assert m.B.shape  == (dxy, dxy)
        assert m.mQ.shape == (dxy, dxy)

    @pytest.mark.parametrize("ModelCls", [
        Model_x1_y1_AQ_pairwise,
        Model_x2_y2_AQ_pairwise,
        Model_x3_y1_AQ_pairwise,
    ])
    def test_mQ_positive_semidefinite(self, ModelCls):
        m = ModelCls()
        eigvals = np.linalg.eigvalsh(m.mQ)
        assert np.all(eigvals >= -1e-10), f"mQ not PSD: min eigenvalue = {eigvals.min()}"

    def test_pairwise_flag(self):
        m = Model_x1_y1_AQ_pairwise()
        assert m.pairwiseModel is True

    def test_classic_flag(self):
        m = Model_x1_y1_AQ_classic()
        assert m.pairwiseModel is False


class TestNonlinearModels:

    def test_x2y1_dimensions(self):
        m = Model_x2_y1_pairwise()
        assert m.dim_x == 2
        assert m.dim_y == 1
        assert m.dim_xy == 3

    def test_x1y1_dimensions(self):
        m = NL_x1y1()
        assert m.dim_x == 1
        assert m.dim_y == 1
        assert m.dim_xy == 2

    def test_transition_callable(self):
        m = Model_x2_y1_pairwise()
        z = np.zeros((m.dim_xy, 1))
        noise = np.zeros((m.dim_xy, 1))
        result = m.g(z, noise, dt=1)
        assert result.shape == (m.dim_xy, 1)

    def test_prior_shapes(self):
        m = Model_x2_y1_pairwise()
        assert m.mz0.shape == (m.dim_xy, 1)
        assert m.Pz0.shape == (m.dim_xy, m.dim_xy)

    def test_get_params_keys(self):
        m = Model_x2_y1_pairwise()
        p = m.get_params()
        for key in ("dim_x", "dim_y", "mQ", "mz0", "Pz0", "augmented", "pairwiseModel"):
            assert key in p, f"Missing key: {key}"
