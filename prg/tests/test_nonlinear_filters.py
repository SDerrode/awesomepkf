"""Tests for nonlinear filters: EPKF and UPKF."""

import numpy as np
import pytest

from prg.classes.NonLinear_EPKF import NonLinear_EPKF
from prg.classes.NonLinear_UPKF import NonLinear_UPKF
from prg.utils.exceptions import ParamError
from prg.tests.conftest import make_param_nonlinear
from prg.models.nonLinear.model_x2_y1_pairwise import Model_x2_y1_pairwise
from prg.models.nonLinear.model_x1_y1_pairwise import Model_x1_y1_pairwise as NL_x1y1

SEED = 42
N_SHORT = 50
N_CALIB = 300


class TestEPKF:

    def test_valid_init(self, param_nl_x2y1):
        epkf = NonLinear_EPKF(param_nl_x2y1, sKey=SEED)
        assert epkf.dim_x == 2
        assert epkf.dim_y == 1

    def test_output_length(self, param_nl_x2y1):
        epkf = NonLinear_EPKF(param_nl_x2y1, sKey=SEED)
        results = epkf.process_N_data(N=N_SHORT)
        assert len(results) == N_SHORT + 1

    def test_output_shapes(self, param_nl_x2y1):
        epkf = NonLinear_EPKF(param_nl_x2y1, sKey=SEED)
        results = epkf.process_N_data(N=10)
        for k, x_true, y_obs, x_pred, x_upd in results:
            assert x_true.shape == (param_nl_x2y1.dim_x, 1)
            assert y_obs.shape  == (param_nl_x2y1.dim_y, 1)
            assert x_pred.shape == (param_nl_x2y1.dim_x, 1)
            assert x_upd.shape  == (param_nl_x2y1.dim_x, 1)

    def test_x1y1_model(self, param_nl_x1y1):
        epkf = NonLinear_EPKF(param_nl_x1y1, sKey=SEED)
        results = epkf.process_N_data(N=N_SHORT)
        assert len(results) == N_SHORT + 1


class TestUPKF:

    @pytest.mark.parametrize("sigma_set", ["wan2000", "cpkf", "lerner2002"])
    def test_valid_init(self, param_nl_x2y1, sigma_set):
        upkf = NonLinear_UPKF(param_nl_x2y1, sigmaSet=sigma_set, sKey=SEED)
        assert upkf.dim_x == 2

    def test_invalid_sigma_set(self, param_nl_x2y1):
        with pytest.raises((ParamError, Exception)):
            NonLinear_UPKF(param_nl_x2y1, sigmaSet="unknown_set", sKey=SEED)

    def test_output_length_wan2000(self, param_nl_x2y1):
        upkf = NonLinear_UPKF(param_nl_x2y1, sigmaSet="wan2000", sKey=SEED)
        results = upkf.process_N_data(N=N_SHORT)
        assert len(results) == N_SHORT + 1

    def test_output_shapes_wan2000(self, param_nl_x2y1):
        upkf = NonLinear_UPKF(param_nl_x2y1, sigmaSet="wan2000", sKey=SEED)
        results = upkf.process_N_data(N=10)
        for k, x_true, y_obs, x_pred, x_upd in results:
            assert x_pred.shape == (param_nl_x2y1.dim_x, 1)
            assert x_upd.shape  == (param_nl_x2y1.dim_x, 1)

    def test_epkf_upkf_close_estimates(self):
        """EPKF and UPKF should give close estimates on a mildly nonlinear model."""
        p_e = make_param_nonlinear(NL_x1y1())
        p_u = make_param_nonlinear(NL_x1y1())
        epkf = NonLinear_EPKF(p_e, sKey=SEED)
        upkf = NonLinear_UPKF(p_u, sigmaSet="wan2000", sKey=SEED)

        res_e = epkf.process_N_data(N=N_CALIB)
        res_u = upkf.process_N_data(N=N_CALIB)

        upd_e = np.array([r[4] for r in res_e[-50:]])
        upd_u = np.array([r[4] for r in res_u[-50:]])
        mean_diff = float(np.mean(np.abs(upd_e - upd_u)))
        assert mean_diff < 5.0, f"EPKF and UPKF diverged too much: {mean_diff:.3f}"
