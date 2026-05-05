"""Tests for the particle filters (PF, PPF) and UKF."""

import numpy as np
import pytest

from prg.classes.nonlinear_pf import NonLinear_PF
from prg.classes.nonlinear_ppf import NonLinear_PPF
from prg.classes.nonlinear_ukf import NonLinear_UKF
from prg.utils.exceptions import ParamError

SEED = 42
N_SHORT = 50
N_PARTICLES = 80


# =====================================================================
# NonLinear_PPF
# =====================================================================


class TestPPFInit:

    def test_valid_init(self, param_nl_x2y1):
        ppf = NonLinear_PPF(param_nl_x2y1, n_particles=N_PARTICLES, sKey=SEED)
        assert ppf.n_particles == N_PARTICLES
        assert ppf.dim_x == 2
        assert ppf.dim_y == 1

    def test_default_resample_method(self, param_nl_x2y1):
        ppf = NonLinear_PPF(param_nl_x2y1, sKey=SEED)
        assert ppf.resample_method == "stratified"
        assert ppf.resample_threshold == 0.5

    def test_custom_resample_method(self, param_nl_x2y1):
        ppf = NonLinear_PPF(
            param_nl_x2y1,
            n_particles=N_PARTICLES,
            resample_method="systematic",
            sKey=SEED,
        )
        assert ppf.resample_method == "systematic"


class TestPPFFilter:

    def test_output_length(self, param_nl_x2y1):
        ppf = NonLinear_PPF(param_nl_x2y1, n_particles=N_PARTICLES, sKey=SEED)
        results = list(ppf.process_filter(N=N_SHORT))
        assert len(results) == N_SHORT + 1

    def test_output_shapes(self, param_nl_x2y1):
        ppf = NonLinear_PPF(param_nl_x2y1, n_particles=N_PARTICLES, sKey=SEED)
        for _k, x_true, y_obs, x_pred, x_upd in ppf.process_filter(N=10):
            assert x_pred.shape == (param_nl_x2y1.dim_x, 1)
            assert x_upd.shape == (param_nl_x2y1.dim_x, 1)
            if x_true is not None:
                assert x_true.shape == (param_nl_x2y1.dim_x, 1)
            assert y_obs.shape == (param_nl_x2y1.dim_y, 1)

    def test_x1y1_model(self, param_nl_x1y1):
        ppf = NonLinear_PPF(param_nl_x1y1, n_particles=N_PARTICLES, sKey=SEED)
        results = list(ppf.process_filter(N=N_SHORT))
        assert len(results) == N_SHORT + 1

    def test_history_recorded(self, param_nl_x2y1):
        ppf = NonLinear_PPF(param_nl_x2y1, n_particles=N_PARTICLES, sKey=SEED)
        list(ppf.process_filter(N=N_SHORT))
        assert len(ppf.history) == N_SHORT + 1


# =====================================================================
# NonLinear_PF — bootstrap particle filter (classic FxHx models only)
# =====================================================================


class TestPFInit:

    def test_valid_init(self, param_nl_classic_x1y1):
        pf = NonLinear_PF(param_nl_classic_x1y1, n_particles=N_PARTICLES, sKey=SEED)
        assert pf.n_particles == N_PARTICLES
        assert pf.dim_x == 1

    def test_rejects_pairwise_model(self, param_nl_x1y1):
        with pytest.raises(ParamError, match="does not support pairwise"):
            NonLinear_PF(param_nl_x1y1, n_particles=N_PARTICLES, sKey=SEED)


class TestPFFilter:

    def test_output_length(self, param_nl_classic_x1y1):
        pf = NonLinear_PF(param_nl_classic_x1y1, n_particles=N_PARTICLES, sKey=SEED)
        results = list(pf.process_filter(N=N_SHORT))
        assert len(results) == N_SHORT + 1

    def test_output_shapes(self, param_nl_classic_x1y1):
        pf = NonLinear_PF(param_nl_classic_x1y1, n_particles=N_PARTICLES, sKey=SEED)
        for _k, _x_true, _y_obs, x_pred, x_upd in pf.process_filter(N=10):
            assert x_pred.shape == (param_nl_classic_x1y1.dim_x, 1)
            assert x_upd.shape == (param_nl_classic_x1y1.dim_x, 1)

    def test_x2y1_classic_model(self, param_nl_classic_x2y1):
        pf = NonLinear_PF(param_nl_classic_x2y1, n_particles=N_PARTICLES, sKey=SEED)
        results = list(pf.process_filter(N=N_SHORT))
        assert len(results) == N_SHORT + 1


# =====================================================================
# Resampling — shared helper from _BaseParticleFilter
# =====================================================================


class TestResampling:

    @pytest.fixture
    def filter_instance(self, param_nl_x2y1):
        return NonLinear_PPF(param_nl_x2y1, n_particles=20, sKey=SEED)

    @pytest.mark.parametrize(
        "method",
        ["multinomial", "systematic", "stratified", "residual"],
    )
    def test_resample_methods_return_correct_count(self, filter_instance, method):
        weights = np.ones(20) / 20
        indexes = filter_instance.resample(weights, method=method)
        assert len(indexes) == 20
        assert indexes.dtype.kind == "i"
        assert (indexes >= 0).all() and (indexes < 20).all()

    def test_resample_unknown_method_raises(self, filter_instance):
        weights = np.ones(20) / 20
        with pytest.raises(ParamError, match="Unknown resampling method"):
            filter_instance.resample(weights, method="not_a_method")

    def test_safe_normalize_uniform_when_all_neg_inf(self, filter_instance):
        log_w = np.full(20, -np.inf)
        w = filter_instance._safe_normalize_log_weights(log_w)
        assert np.allclose(w, 1.0 / 20)

    def test_safe_normalize_handles_nan(self, filter_instance):
        log_w = np.array([-1.0, np.nan, -2.0])
        w = filter_instance._safe_normalize_log_weights(log_w)
        assert np.isclose(w.sum(), 1.0)
        assert w[1] == 0.0  # NaN entry → -inf → 0 weight


# =====================================================================
# NonLinear_UKF
# =====================================================================


class TestUKFInit:

    def test_valid_init(self, param_nl_classic_x2y1):
        ukf = NonLinear_UKF(param_nl_classic_x2y1, sigmaSet="wan2000", sKey=SEED)
        assert ukf.dim_x == 2
        assert ukf.dim_y == 1

    def test_rejects_pairwise_model(self, param_nl_x2y1):
        from prg.utils.exceptions import FilterError

        with pytest.raises(FilterError, match="UKF does not support pairwise"):
            NonLinear_UKF(param_nl_x2y1, sigmaSet="wan2000", sKey=SEED)

    @pytest.mark.parametrize("sigma_set", ["wan2000", "cpkf", "lerner2002"])
    def test_multiple_sigma_sets(self, param_nl_classic_x2y1, sigma_set):
        ukf = NonLinear_UKF(param_nl_classic_x2y1, sigmaSet=sigma_set, sKey=SEED)
        assert ukf.dim_x == 2

    def test_invalid_sigma_set_raises(self, param_nl_classic_x2y1):
        with pytest.raises(ParamError, match="Unknown sigma-point set"):
            NonLinear_UKF(param_nl_classic_x2y1, sigmaSet="bogus", sKey=SEED)


class TestUKFFilter:

    def test_output_length(self, param_nl_classic_x2y1):
        ukf = NonLinear_UKF(param_nl_classic_x2y1, sigmaSet="wan2000", sKey=SEED)
        results = list(ukf.process_filter(N=N_SHORT))
        assert len(results) == N_SHORT + 1

    def test_output_shapes(self, param_nl_classic_x2y1):
        ukf = NonLinear_UKF(param_nl_classic_x2y1, sigmaSet="wan2000", sKey=SEED)
        for _k, _x_true, _y_obs, x_pred, x_upd in ukf.process_filter(N=10):
            assert x_pred.shape == (param_nl_classic_x2y1.dim_x, 1)
            assert x_upd.shape == (param_nl_classic_x2y1.dim_x, 1)

    def test_x1y1_classic(self, param_nl_classic_x1y1):
        ukf = NonLinear_UKF(param_nl_classic_x1y1, sigmaSet="wan2000", sKey=SEED)
        results = list(ukf.process_filter(N=N_SHORT))
        assert len(results) == N_SHORT + 1
