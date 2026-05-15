"""Tests for the Unscented Kalman Smoother (NonLinear_UKS)."""

import logging

import numpy as np
import pytest

from prg.classes.nonlinear_uks import NonLinear_UKS

SEED = 42
N_SHORT = 100
N_REG = 300
N_SEEDS_REG = 20
SIGMA_SET = "wan2000"

PSD_TOL = 1e-6
SHAPE_TOL = 1e-12


class TestNonLinearUKSShapes:

    def test_output_length(self, param_nl_classic_x2y1):
        uks = NonLinear_UKS(param_nl_classic_x2y1, sigmaSet=SIGMA_SET, sKey=SEED)
        res = uks.process_N_data_smoother(N=N_SHORT)
        assert len(res) == N_SHORT + 1

    def test_output_tuple_shapes(self, param_nl_classic_x2y1):
        uks = NonLinear_UKS(param_nl_classic_x2y1, sigmaSet=SIGMA_SET, sKey=SEED)
        res = uks.process_N_data_smoother(N=10)
        for k, x_true, y_obs, x_pred, x_upd, x_smooth in res:
            assert isinstance(k, int)
            assert x_true.shape == (param_nl_classic_x2y1.dim_x, 1)
            assert y_obs.shape == (param_nl_classic_x2y1.dim_y, 1)
            assert x_pred.shape == (param_nl_classic_x2y1.dim_x, 1)
            assert x_upd.shape == (param_nl_classic_x2y1.dim_x, 1)
            assert x_smooth.shape == (param_nl_classic_x2y1.dim_x, 1)

    def test_step_indices_are_sequential(self, param_nl_classic_x2y1):
        uks = NonLinear_UKS(param_nl_classic_x2y1, sigmaSet=SIGMA_SET, sKey=SEED)
        res = uks.process_N_data_smoother(N=N_SHORT)
        assert [r[0] for r in res] == list(range(N_SHORT + 1))


class TestNonLinearUKSGainShape:
    """The classical UKS gain is (dim_x, dim_x) — NOT (dim_x, dim_xy) like
    the pairwise smoothers — because the model is Markov in X alone."""

    @pytest.mark.parametrize(
        "param_fixture",
        ["param_nl_classic_x1y1", "param_nl_classic_x2y1"],
    )
    def test_gain_is_dim_x_by_dim_x(self, param_fixture, request):
        param = request.getfixturevalue(param_fixture)
        uks = NonLinear_UKS(param, sigmaSet=SIGMA_SET, sKey=SEED)
        uks.process_N_data_smoother(N=20)
        for rec in uks.history:
            assert rec["Gk_smooth"].shape == (param.dim_x, param.dim_x)


class TestNonLinearUKSTerminalEquality:

    @pytest.mark.parametrize(
        "param_fixture",
        ["param_nl_classic_x1y1", "param_nl_classic_x2y1"],
    )
    def test_terminal_X_and_PXX(self, param_fixture, request):
        param = request.getfixturevalue(param_fixture)
        uks = NonLinear_UKS(param, sigmaSet=SIGMA_SET, sKey=SEED)
        uks.process_N_data_smoother(N=N_SHORT)
        last = uks.history[-1]
        assert np.allclose(last["Xkp1_smooth"], last["Xkp1_update"], atol=SHAPE_TOL)
        assert np.allclose(last["PXXkp1_smooth"], last["PXXkp1_update"], atol=SHAPE_TOL)


class TestNonLinearUKSShrinkage:

    @pytest.mark.parametrize(
        "param_fixture",
        ["param_nl_classic_x1y1", "param_nl_classic_x2y1"],
    )
    def test_psd_shrinkage(self, param_fixture, request):
        param = request.getfixturevalue(param_fixture)
        uks = NonLinear_UKS(param, sigmaSet=SIGMA_SET, sKey=SEED)
        uks.process_N_data_smoother(N=N_SHORT)
        for rec in uks.history:
            D = rec["PXXkp1_update"] - rec["PXXkp1_smooth"]
            D = 0.5 * (D + D.T)
            min_eig = np.linalg.eigvalsh(D).min()
            assert min_eig > -PSD_TOL


class TestNonLinearUKSJosephForm:

    JOSEPH_EQ_TOL = 1e-10

    @pytest.mark.parametrize(
        "param_fixture",
        ["param_nl_classic_x1y1", "param_nl_classic_x2y1"],
    )
    def test_joseph_equals_standard_means(self, param_fixture, request):
        param = request.getfixturevalue(param_fixture)
        uks_std = NonLinear_UKS(param, sigmaSet=SIGMA_SET, sKey=SEED, joseph=False)
        uks_jos = NonLinear_UKS(param, sigmaSet=SIGMA_SET, sKey=SEED, joseph=True)
        res_std = uks_std.process_N_data_smoother(N=N_SHORT)
        res_jos = uks_jos.process_N_data_smoother(N=N_SHORT)
        for a, b in zip(res_std, res_jos):
            assert np.allclose(a[5], b[5], atol=self.JOSEPH_EQ_TOL)

    @pytest.mark.parametrize(
        "param_fixture",
        ["param_nl_classic_x1y1", "param_nl_classic_x2y1"],
    )
    def test_joseph_equals_standard_covariances(self, param_fixture, request):
        param = request.getfixturevalue(param_fixture)
        uks_std = NonLinear_UKS(param, sigmaSet=SIGMA_SET, sKey=SEED, joseph=False)
        uks_jos = NonLinear_UKS(param, sigmaSet=SIGMA_SET, sKey=SEED, joseph=True)
        uks_std.process_N_data_smoother(N=N_SHORT)
        uks_jos.process_N_data_smoother(N=N_SHORT)
        for r1, r2 in zip(uks_std.history, uks_jos.history):
            diff = np.max(np.abs(r1["PXXkp1_smooth"] - r2["PXXkp1_smooth"]))
            assert diff < self.JOSEPH_EQ_TOL

    def test_joseph_flag_default_false(self, param_nl_classic_x2y1):
        uks = NonLinear_UKS(param_nl_classic_x2y1, sigmaSet=SIGMA_SET, sKey=SEED)
        assert uks.joseph is False


class TestNonLinearUKSPairwiseGuard:
    """The classical UKS must refuse a pairwise model (inherited from
    NonLinear_UKF), since the model assumption is Markov-in-X."""

    def test_pairwise_model_raises(self, param_nl_x2y1):
        from prg.utils.exceptions import FilterError
        with pytest.raises(FilterError):
            NonLinear_UKS(param_nl_x2y1, sigmaSet=SIGMA_SET, sKey=SEED)


class TestNonLinearUKSSigmaPointSets:

    @pytest.mark.parametrize("sigma_set", ["wan2000", "cpkf", "lerner2002"])
    def test_runs_with_each_sigma_set(self, param_nl_classic_x2y1, sigma_set):
        uks = NonLinear_UKS(param_nl_classic_x2y1, sigmaSet=sigma_set, sKey=SEED)
        res = uks.process_N_data_smoother(N=50)
        assert len(res) == 51

    def test_unknown_sigma_set_raises_paramerror(self, param_nl_classic_x2y1):
        from prg.utils.exceptions import ParamError
        with pytest.raises(ParamError):
            NonLinear_UKS(param_nl_classic_x2y1, sigmaSet="bogus", sKey=SEED)


class TestNonLinearUKSEdgeCases:

    def test_N_equals_1(self, param_nl_classic_x2y1):
        uks = NonLinear_UKS(param_nl_classic_x2y1, sigmaSet=SIGMA_SET, sKey=SEED)
        res = uks.process_N_data_smoother(N=1)
        assert len(res) == 2
        assert np.allclose(res[-1][4], res[-1][5], atol=SHAPE_TOL)

    def test_process_smoother_generator_lazy(self, param_nl_classic_x2y1):
        uks = NonLinear_UKS(param_nl_classic_x2y1, sigmaSet=SIGMA_SET, sKey=SEED)
        gen = uks.process_smoother(N=N_SHORT)
        first_tuple = next(gen)
        assert first_tuple[0] == 0
        for rec in uks.history:
            assert "Xkp1_smooth" in rec
        gen.close()

    def test_process_smoother_twice_in_a_row(self, param_nl_classic_x2y1):
        uks = NonLinear_UKS(param_nl_classic_x2y1, sigmaSet=SIGMA_SET, sKey=SEED)
        res1 = uks.process_N_data_smoother(N=N_SHORT)
        res2 = uks.process_N_data_smoother(N=N_SHORT)
        assert len(res1) == len(res2) == N_SHORT + 1

    def test_external_data_generator(self, param_nl_classic_x2y1):
        uks_ref = NonLinear_UKS(param_nl_classic_x2y1, sigmaSet=SIGMA_SET, sKey=SEED)
        ref = uks_ref.process_N_data_smoother(N=30)
        triplets = [(r[0], r[1], r[2]) for r in ref]

        def replay():
            for k, x, y in triplets:
                yield k, x, y

        uks_ext = NonLinear_UKS(param_nl_classic_x2y1, sigmaSet=SIGMA_SET, sKey=SEED)
        ext = uks_ext.process_N_data_smoother(N=30, data_generator=replay())
        for a, b in zip(ref, ext):
            assert np.allclose(a[4], b[4], atol=SHAPE_TOL)
            assert np.allclose(a[5], b[5], atol=SHAPE_TOL)

    def test_missing_ground_truth(self, param_nl_classic_x2y1):
        uks_ref = NonLinear_UKS(param_nl_classic_x2y1, sigmaSet=SIGMA_SET, sKey=SEED)
        ref = uks_ref.process_N_data_smoother(N=15)

        def gen_no_truth():
            for r in ref:
                yield r[0], None, r[2]

        uks = NonLinear_UKS(param_nl_classic_x2y1, sigmaSet=SIGMA_SET, sKey=SEED)
        res = uks.process_N_data_smoother(N=None, data_generator=gen_no_truth())
        assert len(res) == len(ref)
        assert all(r[1] is None for r in res)


class TestNonLinearUKSExceptionPolicy:

    def test_invalid_N_raises_paramerror(self, param_nl_classic_x2y1):
        from prg.utils.exceptions import ParamError
        uks = NonLinear_UKS(param_nl_classic_x2y1, sigmaSet=SIGMA_SET, sKey=SEED)
        with pytest.raises(ParamError):
            uks.process_N_data_smoother(N=0)


class TestNonLinearUKSLogging:

    LOGGER_NAME = "prg.classes.nonlinear_uks"

    def test_info_logs_emitted_at_entry_and_exit(self, param_nl_classic_x2y1, caplog):
        with caplog.at_level(logging.INFO, logger=self.LOGGER_NAME):
            uks = NonLinear_UKS(param_nl_classic_x2y1, sigmaSet=SIGMA_SET, sKey=SEED)
            uks.process_N_data_smoother(N=20)
        info_msgs = [
            r.message for r in caplog.records
            if r.name == self.LOGGER_NAME and r.levelno == logging.INFO
        ]
        assert len(info_msgs) == 2
        assert "backward pass starting" in info_msgs[0]
        assert "backward pass complete" in info_msgs[1]

    def test_debug_logs_one_per_backward_step(self, param_nl_classic_x2y1, caplog):
        N = 10
        with caplog.at_level(logging.DEBUG, logger=self.LOGGER_NAME):
            uks = NonLinear_UKS(param_nl_classic_x2y1, sigmaSet=SIGMA_SET, sKey=SEED)
            uks.process_N_data_smoother(N=N)
        debug_msgs = [
            r for r in caplog.records
            if r.name == self.LOGGER_NAME and r.levelno == logging.DEBUG
        ]
        assert len(debug_msgs) == N


class TestNonLinearUKSRegression:
    """Regression tests on `model_x1_y1_Sinus_classic` — a sufficiently
    nonlinear model for the UKS to show a clear MSE improvement."""

    def test_avg_trace_shrinks(self, param_nl_classic_x1y1):
        uks = NonLinear_UKS(param_nl_classic_x1y1, sigmaSet=SIGMA_SET, sKey=SEED)
        uks.process_N_data_smoother(N=N_REG)
        tr_f = np.array([np.trace(r["PXXkp1_update"]) for r in uks.history])
        tr_s = np.array([np.trace(r["PXXkp1_smooth"]) for r in uks.history])
        assert (tr_s <= tr_f + PSD_TOL).all()
        margin = 0.01 * tr_f.mean()
        assert (tr_s < tr_f - margin).sum() > N_REG // 2

    def test_smoother_beats_filter_on_average(self, param_nl_classic_x1y1):
        mses_f, mses_s = [], []
        for sd in range(N_SEEDS_REG):
            uks = NonLinear_UKS(param_nl_classic_x1y1, sigmaSet=SIGMA_SET, sKey=sd)
            res = uks.process_N_data_smoother(N=N_REG)
            xt = np.array([r[1].flatten() for r in res])
            xf = np.array([r[4].flatten() for r in res])
            xs = np.array([r[5].flatten() for r in res])
            mses_f.append(((xf - xt) ** 2).mean())
            mses_s.append(((xs - xt) ** 2).mean())
        ratio = np.mean(mses_s) / np.mean(mses_f)
        assert ratio < 0.97, (
            f"Expected UKS / UKF MSE ratio < 0.97 on Sinus_classic, got {ratio:.3f}"
        )
