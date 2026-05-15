"""Tests for the Extended Pairwise Kalman Smoother (NonLinear_EPKS)."""

import logging

import numpy as np
import pytest

from prg.classes.nonlinear_epks import NonLinear_EPKS
from prg.utils.exceptions import CovarianceError

SEED = 42
N_SHORT = 100
N_REG = 300
N_SEEDS_REG = 20

PSD_TOL = 1e-6      # eigenvalue tolerance — looser than linear due to linearisation noise
SHAPE_TOL = 1e-12


class TestNonLinearEPKSShapes:

    def test_output_length(self, param_nl_x2y1):
        epks = NonLinear_EPKS(param_nl_x2y1, sKey=SEED)
        res = epks.process_N_data_smoother(N=N_SHORT)
        assert len(res) == N_SHORT + 1

    def test_output_tuple_shapes(self, param_nl_x2y1):
        epks = NonLinear_EPKS(param_nl_x2y1, sKey=SEED)
        res = epks.process_N_data_smoother(N=10)
        for k, x_true, y_obs, x_pred, x_upd, x_smooth in res:
            assert isinstance(k, int)
            assert x_true.shape == (param_nl_x2y1.dim_x, 1)
            assert y_obs.shape == (param_nl_x2y1.dim_y, 1)
            assert x_pred.shape == (param_nl_x2y1.dim_x, 1)
            assert x_upd.shape == (param_nl_x2y1.dim_x, 1)
            assert x_smooth.shape == (param_nl_x2y1.dim_x, 1)

    def test_step_indices_are_sequential(self, param_nl_x2y1):
        epks = NonLinear_EPKS(param_nl_x2y1, sKey=SEED)
        res = epks.process_N_data_smoother(N=N_SHORT)
        assert [r[0] for r in res] == list(range(N_SHORT + 1))


class TestNonLinearEPKSTerminalEquality:

    @pytest.mark.parametrize("param_fixture", ["param_nl_x1y1", "param_nl_x2y1"])
    def test_terminal_X_and_PXX(self, param_fixture, request):
        param = request.getfixturevalue(param_fixture)
        epks = NonLinear_EPKS(param, sKey=SEED)
        epks.process_N_data_smoother(N=N_SHORT)
        last = epks.history[-1]
        assert np.allclose(last["Xkp1_smooth"], last["Xkp1_update"], atol=SHAPE_TOL)
        assert np.allclose(last["PXXkp1_smooth"], last["PXXkp1_update"], atol=SHAPE_TOL)


class TestNonLinearEPKSShrinkage:
    """The linearised covariance still satisfies P^{xx}_{n|n} >= P^{xx}_{n|N}.

    Note that, unlike the strictly linear case, this PSD ordering on the
    *linearised* covariance does **not** transfer to the *empirical* MSE
    on a single trajectory — the linearisation bias breaks that link.
    """

    @pytest.mark.parametrize("param_fixture", ["param_nl_x1y1", "param_nl_x2y1"])
    def test_psd_shrinkage(self, param_fixture, request):
        param = request.getfixturevalue(param_fixture)
        epks = NonLinear_EPKS(param, sKey=SEED)
        epks.process_N_data_smoother(N=N_SHORT)
        for rec in epks.history:
            D = rec["PXXkp1_update"] - rec["PXXkp1_smooth"]
            D = 0.5 * (D + D.T)
            min_eig = np.linalg.eigvalsh(D).min()
            assert min_eig > -PSD_TOL, (
                f"Step {rec['k']}: P_filter - P_smooth not PSD "
                f"(min eig {min_eig})"
            )


class TestNonLinearEPKSJosephForm:
    """Joseph form must match the standard form on the linearised covariance."""

    JOSEPH_EQ_TOL = 1e-10

    @pytest.mark.parametrize("param_fixture", ["param_nl_x1y1", "param_nl_x2y1"])
    def test_joseph_equals_standard_means(self, param_fixture, request):
        param = request.getfixturevalue(param_fixture)
        epks_std = NonLinear_EPKS(param, sKey=SEED, joseph=False)
        epks_jos = NonLinear_EPKS(param, sKey=SEED, joseph=True)
        res_std = epks_std.process_N_data_smoother(N=N_SHORT)
        res_jos = epks_jos.process_N_data_smoother(N=N_SHORT)
        for a, b in zip(res_std, res_jos):
            assert np.allclose(a[5], b[5], atol=self.JOSEPH_EQ_TOL)

    @pytest.mark.parametrize("param_fixture", ["param_nl_x1y1", "param_nl_x2y1"])
    def test_joseph_equals_standard_covariances(self, param_fixture, request):
        param = request.getfixturevalue(param_fixture)
        epks_std = NonLinear_EPKS(param, sKey=SEED, joseph=False)
        epks_jos = NonLinear_EPKS(param, sKey=SEED, joseph=True)
        epks_std.process_N_data_smoother(N=N_SHORT)
        epks_jos.process_N_data_smoother(N=N_SHORT)
        for r1, r2 in zip(epks_std.history, epks_jos.history):
            diff = np.max(np.abs(r1["PXXkp1_smooth"] - r2["PXXkp1_smooth"]))
            assert diff < self.JOSEPH_EQ_TOL, (
                f"Step {r1['k']}: |P_std - P_joseph| = {diff:.2e}"
            )

    def test_joseph_flag_default_false(self, param_nl_x2y1):
        epks = NonLinear_EPKS(param_nl_x2y1, sKey=SEED)
        assert epks.joseph is False

    @pytest.mark.parametrize("param_fixture", ["param_nl_x1y1", "param_nl_x2y1"])
    def test_joseph_psd_shrinkage(self, param_fixture, request):
        """Joseph form preserves the linearised-covariance PSD shrinkage."""
        param = request.getfixturevalue(param_fixture)
        epks = NonLinear_EPKS(param, sKey=SEED, joseph=True)
        epks.process_N_data_smoother(N=N_SHORT)
        for rec in epks.history:
            D = rec["PXXkp1_update"] - rec["PXXkp1_smooth"]
            D = 0.5 * (D + D.T)
            min_eig = np.linalg.eigvalsh(D).min()
            assert min_eig > -PSD_TOL, (
                f"Step {rec['k']}: P_f - P_s not PSD under Joseph (min eig {min_eig})"
            )


class TestNonLinearEPKSEdgeCases:

    def test_N_equals_1(self, param_nl_x2y1):
        epks = NonLinear_EPKS(param_nl_x2y1, sKey=SEED)
        res = epks.process_N_data_smoother(N=1)
        assert len(res) == 2
        # Terminal step: smoothed == filtered
        assert np.allclose(res[-1][4], res[-1][5], atol=SHAPE_TOL)

    def test_process_smoother_generator_lazy(self, param_nl_x2y1):
        epks = NonLinear_EPKS(param_nl_x2y1, sKey=SEED)
        gen = epks.process_smoother(N=N_SHORT)
        first_tuple = next(gen)
        assert first_tuple[0] == 0
        # Forward + backward both completed before the first yield
        for rec in epks.history:
            assert "Xkp1_smooth" in rec
            assert "Gk_smooth" in rec
        gen.close()

    def test_process_smoother_twice_in_a_row(self, param_nl_x2y1):
        epks = NonLinear_EPKS(param_nl_x2y1, sKey=SEED)
        res1 = epks.process_N_data_smoother(N=N_SHORT)
        res2 = epks.process_N_data_smoother(N=N_SHORT)
        assert len(res1) == len(res2) == N_SHORT + 1

    def test_smoother_fields_have_correct_shapes(self, param_nl_x2y1):
        epks = NonLinear_EPKS(param_nl_x2y1, sKey=SEED)
        epks.process_N_data_smoother(N=N_SHORT)
        for rec in epks.history:
            assert rec["Gk_smooth"].shape == (param_nl_x2y1.dim_x, param_nl_x2y1.dim_xy)
            assert rec["Xkp1_smooth"].shape == (param_nl_x2y1.dim_x, 1)
            assert rec["PXXkp1_smooth"].shape == (param_nl_x2y1.dim_x, param_nl_x2y1.dim_x)
        assert np.allclose(epks.history[-1]["Gk_smooth"], 0.0)

    def test_external_data_generator(self, param_nl_x2y1):
        epks_ref = NonLinear_EPKS(param_nl_x2y1, sKey=SEED)
        ref = epks_ref.process_N_data_smoother(N=30)
        triplets = [(r[0], r[1], r[2]) for r in ref]

        def replay():
            for k, x, y in triplets:
                yield k, x, y

        epks_ext = NonLinear_EPKS(param_nl_x2y1, sKey=SEED)
        ext = epks_ext.process_N_data_smoother(N=30, data_generator=replay())
        for a, b in zip(ref, ext):
            assert np.allclose(a[4], b[4], atol=SHAPE_TOL)
            assert np.allclose(a[5], b[5], atol=SHAPE_TOL)

    def test_missing_ground_truth(self, param_nl_x2y1):
        """Replay through a generator yielding x_true=None — the smoother
        must complete and propagate None through the output tuples."""
        epks_ref = NonLinear_EPKS(param_nl_x2y1, sKey=SEED)
        ref = epks_ref.process_N_data_smoother(N=15)

        def gen_no_truth():
            for r in ref:
                # Re-yield same observations but with None as truth
                yield r[0], None, r[2]

        epks = NonLinear_EPKS(param_nl_x2y1, sKey=SEED)
        res = epks.process_N_data_smoother(N=None, data_generator=gen_no_truth())
        assert len(res) == len(ref)
        assert all(r[1] is None for r in res)


class TestNonLinearEPKSExceptionPolicy:

    def test_invalid_N_raises_paramerror(self, param_nl_x2y1):
        from prg.utils.exceptions import ParamError
        epks = NonLinear_EPKS(param_nl_x2y1, sKey=SEED)
        with pytest.raises(ParamError):
            epks.process_N_data_smoother(N=0)
        with pytest.raises(ParamError):
            epks.process_N_data_smoother(N=-1)

    def test_pkferror_root_catches_smoother_errors(self, param_nl_x1y1):
        """All EPKS errors derive from ``PKFError``; the root class
        suffices to intercept any domain failure."""
        from prg.utils.exceptions import PKFError
        # Strongly pathological: zero process noise on x1y1 augmented-like
        # pairwise — forces a Cholesky failure in the backward Cholesky.
        import copy
        param_bad = copy.copy(param_nl_x1y1)
        param_bad._mQ = np.zeros_like(param_bad._mQ)
        epks = NonLinear_EPKS(param_bad, sKey=SEED)
        with pytest.raises(PKFError):
            epks.process_N_data_smoother(N=10)


class TestNonLinearEPKSLogging:
    """The EPKS emits INFO at backward entry/exit and DEBUG per step."""

    LOGGER_NAME = "prg.classes.nonlinear_epks"

    def test_info_logs_emitted_at_entry_and_exit(self, param_nl_x2y1, caplog):
        with caplog.at_level(logging.INFO, logger=self.LOGGER_NAME):
            epks = NonLinear_EPKS(param_nl_x2y1, sKey=SEED)
            epks.process_N_data_smoother(N=20)
        info_msgs = [
            r.message for r in caplog.records
            if r.name == self.LOGGER_NAME and r.levelno == logging.INFO
        ]
        assert len(info_msgs) == 2
        assert "backward pass starting" in info_msgs[0]
        assert "joseph=False" in info_msgs[0]
        assert "backward pass complete" in info_msgs[1]

    def test_info_log_reflects_joseph_mode(self, param_nl_x2y1, caplog):
        with caplog.at_level(logging.INFO, logger=self.LOGGER_NAME):
            epks = NonLinear_EPKS(param_nl_x2y1, sKey=SEED, joseph=True)
            epks.process_N_data_smoother(N=5)
        entry = next(
            r.message for r in caplog.records
            if r.name == self.LOGGER_NAME and r.levelno == logging.INFO
        )
        assert "joseph=True" in entry

    def test_debug_logs_one_per_backward_step(self, param_nl_x2y1, caplog):
        N = 10
        with caplog.at_level(logging.DEBUG, logger=self.LOGGER_NAME):
            epks = NonLinear_EPKS(param_nl_x2y1, sKey=SEED)
            epks.process_N_data_smoother(N=N)
        debug_msgs = [
            r for r in caplog.records
            if r.name == self.LOGGER_NAME and r.levelno == logging.DEBUG
        ]
        assert len(debug_msgs) == N


class TestNonLinearEPKSRegression:
    """On average over many seeds, the linearised smoother covariance
    shrinks compared to the filter covariance. The MSE may also shrink
    in expectation, but with a smaller margin than in the linear case
    because the linearisation introduces a bias the smoother cannot
    correct on its own."""

    def test_avg_trace_shrinks_x2y1(self, param_nl_x2y1):
        epks = NonLinear_EPKS(param_nl_x2y1, sKey=SEED)
        epks.process_N_data_smoother(N=N_REG)
        tr_f = np.array([np.trace(r["PXXkp1_update"]) for r in epks.history])
        tr_s = np.array([np.trace(r["PXXkp1_smooth"]) for r in epks.history])
        assert (tr_s <= tr_f + PSD_TOL).all()
        # Strict shrinkage on a majority of steps
        margin = 0.01 * tr_f.mean()
        assert (tr_s < tr_f - margin).sum() > N_REG // 2

    def test_smoother_beats_filter_on_average_x2y1(self, param_nl_x2y1):
        mses_f, mses_s = [], []
        for sd in range(N_SEEDS_REG):
            epks = NonLinear_EPKS(param_nl_x2y1, sKey=sd)
            res = epks.process_N_data_smoother(N=N_REG)
            xt = np.array([r[1].flatten() for r in res])
            xf = np.array([r[4].flatten() for r in res])
            xs = np.array([r[5].flatten() for r in res])
            mses_f.append(((xf - xt) ** 2).mean())
            mses_s.append(((xs - xt) ** 2).mean())
        ratio = np.mean(mses_s) / np.mean(mses_f)
        # The EPKS is approximate: we only assert it does not *significantly*
        # degrade the filter (ratio < 1.02 covers seed-to-seed variance and
        # mild linearisation bias).
        assert ratio < 1.02, (
            f"Expected EPKS / EPKF MSE ratio < 1.02 on x2y1 pairwise, "
            f"got {ratio:.3f}"
        )
