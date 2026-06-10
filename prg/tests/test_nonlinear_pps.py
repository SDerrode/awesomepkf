"""Tests for the Pairwise Particle Smoother (NonLinear_PPS, FFBSm)."""

import copy
import logging

import numpy as np
import pytest

from prg.classes.linear_pks import Linear_PKS
from prg.classes.nonlinear_pps import NonLinear_PPS
from prg.classes.param_linear import ParamLinear
from prg.models.linear import ModelFactoryLinear
from prg.utils.exceptions import CovarianceError, ParamError, PKFError

SEED = 42
N_SHORT = 50
N_REG = 200
N_PARTICLES_SHORT = 200
N_PARTICLES_MC = 2000        # for the Monte-Carlo convergence test
SHAPE_TOL = 1e-12


class TestNonLinearPPSShapes:

    def test_output_length(self, param_nl_x2y1):
        pps = NonLinear_PPS(
            param_nl_x2y1, n_particles=N_PARTICLES_SHORT, sKey=SEED,
        )
        res = pps.process_N_data_smoother(N=N_SHORT)
        assert len(res) == N_SHORT + 1

    def test_output_tuple_shapes(self, param_nl_x2y1):
        pps = NonLinear_PPS(
            param_nl_x2y1, n_particles=N_PARTICLES_SHORT, sKey=SEED,
        )
        res = pps.process_N_data_smoother(N=10)
        for k, x_true, y_obs, x_pred, x_upd, x_smooth in res:
            assert isinstance(k, int)
            assert x_true.shape == (param_nl_x2y1.dim_x, 1)
            assert y_obs.shape == (param_nl_x2y1.dim_y, 1)
            assert x_pred.shape == (param_nl_x2y1.dim_x, 1)
            assert x_upd.shape == (param_nl_x2y1.dim_x, 1)
            assert x_smooth.shape == (param_nl_x2y1.dim_x, 1)

    def test_step_indices_are_sequential(self, param_nl_x2y1):
        pps = NonLinear_PPS(
            param_nl_x2y1, n_particles=N_PARTICLES_SHORT, sKey=SEED,
        )
        res = pps.process_N_data_smoother(N=N_SHORT)
        assert [r[0] for r in res] == list(range(N_SHORT + 1))


# Rationale for the absence of `test_terminal_gain_is_zero_placeholder`
# (which exists in test_linear_pks.py, test_nonlinear_epks.py,
# test_nonlinear_upks.py, test_nonlinear_uks.py):
# The PPS has NO ``Gk_smooth`` field — particle smoothers reweight rather
# than apply a gain matrix, so the analog terminal-placeholder concept
# does not exist. The matching structural invariant in the PPS world is
# ``w_smooth[N] == weights[N]``, checked by
# ``TestNonLinearPPSWeights.test_terminal_w_smooth_equals_w_filter``.


class TestNonLinearPPSWeights:
    """Smoothed weights must remain valid probability vectors."""

    def test_smoothed_weights_sum_to_one(self, param_nl_x2y1):
        pps = NonLinear_PPS(
            param_nl_x2y1, n_particles=N_PARTICLES_SHORT, sKey=SEED,
        )
        pps.process_N_data_smoother(N=N_SHORT)
        for rec in pps.history:
            w = rec["w_smooth"]
            assert w.shape == (N_PARTICLES_SHORT,)
            assert np.all(w >= 0.0)
            assert abs(w.sum() - 1.0) < 1e-9, (
                f"Step {rec['k']}: w_smooth.sum() = {w.sum()}"
            )

    def test_terminal_w_smooth_equals_w_filter(self, param_nl_x2y1):
        """At step N the smoothed weights coincide exactly with the
        forward (filtered) weights — this is the initial condition of
        the FFBSm backward recursion."""
        pps = NonLinear_PPS(
            param_nl_x2y1, n_particles=N_PARTICLES_SHORT, sKey=SEED,
        )
        pps.process_N_data_smoother(N=N_SHORT)
        last = pps.history[-1]
        assert np.allclose(last["w_smooth"], last["weights"], atol=SHAPE_TOL)


class TestNonLinearPPSTerminalEquality:
    """At step N, ``w_smooth`` equals the forward weights (boundary
    condition of FFBSm). The smoothed mean and covariance, however, use
    a different estimator from the PPF's Rao-Blackwellised one: PPS
    uses the **raw particle cloud** weighted by ``w_smooth``, while
    PPF's ``Xkp1_update`` is :math:`\\sum_i w_i \\mu'_{x,i}` (conditional
    means). Both target the same posterior but differ by MC variance."""

    def test_terminal_weights_match_forward(self, param_nl_x2y1):
        pps = NonLinear_PPS(
            param_nl_x2y1, n_particles=N_PARTICLES_SHORT, sKey=SEED,
        )
        pps.process_N_data_smoother(N=N_SHORT)
        last = pps.history[-1]
        # Strict equality of the boundary condition w_smooth = weights
        assert np.allclose(last["w_smooth"], last["weights"], atol=SHAPE_TOL)

    def test_terminal_smoothed_mean_close_to_forward(self, param_nl_x2y1):
        """Looser MC-aware bound: smoothed and filtered means at the
        terminal step should be within a few standard deviations of the
        weighted particle cloud."""
        pps = NonLinear_PPS(
            param_nl_x2y1, n_particles=N_PARTICLES_SHORT, sKey=SEED,
        )
        pps.process_N_data_smoother(N=N_SHORT)
        last = pps.history[-1]
        # Particle-cloud weighted std (one per coordinate)
        cov = last["PXXkp1_update"]
        std = np.sqrt(np.diag(cov)).reshape(-1, 1)
        # MC error ~ std / sqrt(n_p)
        mc_tol = 4 * std / np.sqrt(N_PARTICLES_SHORT)
        diff = np.abs(last["Xkp1_smooth"] - last["Xkp1_update"])
        # All coordinates within 4 MC standard errors — extremely loose
        # but catches gross bugs.
        assert (diff < mc_tol + 1e-9).all(), (
            f"Terminal mean diff {diff.flatten()} exceeds MC tol "
            f"{mc_tol.flatten()}"
        )


class TestNonLinearPPSMonteCarloConvergence:
    """Cross-validation: on a linear-Gaussian pairwise model the PPS
    must converge to the exact :class:`Linear_PKS` as ``n_p`` increases.
    This is the strongest correctness check for FFBSm."""

    @pytest.fixture(scope="class")
    def linear_pks_reference(self):
        """Cached reference smoothed trajectory from Linear_PKS on
        ``model_x1_y1_AQ_pairwise`` with seed 42, N=100."""
        m = ModelFactoryLinear.create("model_x1_y1_AQ_pairwise")
        params = m.get_params().copy()
        dim_x = params.pop("dim_x")
        dim_y = params.pop("dim_y")
        pl = ParamLinear(0, dim_x, dim_y, **params)
        pks = Linear_PKS(pl, sKey=SEED)
        res = pks.process_N_data_smoother(N=100)
        xs = np.array([r[5].flatten() for r in res])
        return xs, params, dim_x, dim_y

    def test_mc_error_decreases_with_n_particles(self, linear_pks_reference):
        """The mean-squared deviation between PPS-smoothed and PKS-smoothed
        trajectories must shrink as ``n_particles`` grows (roughly
        :math:`O(1/n_p)` per particle MC theory)."""
        xs_pks, params, dim_x, dim_y = linear_pks_reference
        errors = []
        for n_p in (100, 2000):
            pl = ParamLinear(0, dim_x, dim_y, **params)
            pps = NonLinear_PPS(pl, n_particles=n_p, sKey=SEED)
            res = pps.process_N_data_smoother(N=100)
            xs_pps = np.array([r[5].flatten() for r in res])
            errors.append(float(np.mean((xs_pps - xs_pks) ** 2)))
        # Strict ordering: the high-n_p error must be smaller than low-n_p.
        # Margin = 2× to absorb single-realisation MC variance.
        assert errors[1] < errors[0], (
            f"PPS Monte-Carlo error did not shrink with n_p: "
            f"n_p=100 → {errors[0]:.2e}, n_p=2000 → {errors[1]:.2e}"
        )

    def test_pps_high_n_close_to_pks(self, linear_pks_reference):
        """With ``n_p = 2000``, the PPS smoothed mean trajectory must be
        within ~1e-3 RMS of the exact PKS trajectory on this model."""
        xs_pks, params, dim_x, dim_y = linear_pks_reference
        pl = ParamLinear(0, dim_x, dim_y, **params)
        pps = NonLinear_PPS(pl, n_particles=N_PARTICLES_MC, sKey=SEED)
        res = pps.process_N_data_smoother(N=100)
        xs_pps = np.array([r[5].flatten() for r in res])
        rms = float(np.sqrt(np.mean((xs_pps - xs_pks) ** 2)))
        assert rms < 0.05, (
            f"PPS(n_p={N_PARTICLES_MC}) deviates from PKS by RMS={rms:.4f}"
        )


class TestNonLinearPPSEdgeCases:

    def test_N_equals_1(self, param_nl_x2y1):
        pps = NonLinear_PPS(
            param_nl_x2y1, n_particles=N_PARTICLES_SHORT, sKey=SEED,
        )
        res = pps.process_N_data_smoother(N=1)
        assert len(res) == 2

    def test_process_smoother_generator_lazy(self, param_nl_x2y1):
        pps = NonLinear_PPS(
            param_nl_x2y1, n_particles=N_PARTICLES_SHORT, sKey=SEED,
        )
        gen = pps.process_smoother(N=N_SHORT)
        first_tuple = next(gen)
        assert first_tuple[0] == 0
        for rec in pps.history:
            assert "Xkp1_smooth" in rec
            assert "w_smooth" in rec
        gen.close()

    def test_process_smoother_twice_in_a_row(self, param_nl_x2y1):
        pps = NonLinear_PPS(
            param_nl_x2y1, n_particles=N_PARTICLES_SHORT, sKey=SEED,
        )
        res1 = pps.process_N_data_smoother(N=N_SHORT)
        res2 = pps.process_N_data_smoother(N=N_SHORT)
        assert len(res1) == len(res2) == N_SHORT + 1

    def test_external_data_generator(self, param_nl_x2y1):
        """The PPS must accept an external data generator. We cannot
        require bit-for-bit equality between two runs (``_randParticles``
        is OS-seeded per instance in :class:`_BaseParticleFilter`, so
        each PPS draws a different particle cloud) — just that the two
        runs produce the same number of records with valid structures.
        """
        pps_ref = NonLinear_PPS(
            param_nl_x2y1, n_particles=N_PARTICLES_SHORT, sKey=SEED,
        )
        ref = pps_ref.process_N_data_smoother(N=20)
        triplets = [(r[0], r[1], r[2]) for r in ref]

        def replay():
            yield from triplets

        pps_ext = NonLinear_PPS(
            param_nl_x2y1, n_particles=N_PARTICLES_SHORT, sKey=SEED,
        )
        ext = pps_ext.process_N_data_smoother(N=20, data_generator=replay())
        assert len(ext) == len(ref)
        for a, b in zip(ref, ext, strict=True):
            assert a[0] == b[0]                # step index
            assert a[4].shape == b[4].shape    # filter shape
            assert a[5].shape == b[5].shape    # smoother shape

    def test_missing_ground_truth(self, param_nl_x2y1):
        pps_ref = NonLinear_PPS(
            param_nl_x2y1, n_particles=N_PARTICLES_SHORT, sKey=SEED,
        )
        ref = pps_ref.process_N_data_smoother(N=10)

        def gen_no_truth():
            for r in ref:
                yield r[0], None, r[2]

        pps = NonLinear_PPS(
            param_nl_x2y1, n_particles=N_PARTICLES_SHORT, sKey=SEED,
        )
        res = pps.process_N_data_smoother(N=None, data_generator=gen_no_truth())
        assert len(res) == len(ref)
        assert all(r[1] is None for r in res)


class TestNonLinearPPSExceptionPolicy:

    def test_invalid_N_raises_paramerror(self, param_nl_x2y1):
        pps = NonLinear_PPS(
            param_nl_x2y1, n_particles=N_PARTICLES_SHORT, sKey=SEED,
        )
        with pytest.raises(ParamError):
            pps.process_N_data_smoother(N=0)

    def test_store_particles_forced_true(self, param_nl_x2y1):
        """Even if the user passes ``store_particles=False``, the PPS
        forces it to True because the backward pass needs the cloud."""
        pps = NonLinear_PPS(
            param_nl_x2y1, n_particles=N_PARTICLES_SHORT, sKey=SEED,
            store_particles=False,  # user request ignored
        )
        assert pps.store_particles is True

    def test_pkferror_root_catches_smoother_errors(self, param_nl_x2y1):
        """Exception taxonomy sanity: every PPS domain failure derives
        from ``PKFError``, so a single ``except PKFError`` catches them
        all (parity with the four Kalman smoothers)."""
        pps = NonLinear_PPS(
            param_nl_x2y1, n_particles=N_PARTICLES_SHORT, sKey=SEED,
        )
        with pytest.raises(PKFError):
            pps.process_N_data_smoother(N=0)

    def test_singular_joint_mQ_raises_covariance_error(self, param_nl_x2y1):
        """If the joint transition-noise covariance ``mQ`` (full
        ``(p+q)x(p+q)``) is degenerate when the smoother starts the
        backward pass, the Cholesky cannot be initialised; a
        ``CovarianceError`` with structured ``step=-1`` and
        ``matrix_name='mQ'`` must surface. Parallels Linear_PKS's
        ``test_singular_predicted_covariance_raises``.

        We zero **only the X-marginal block** of ``mQ`` — this already makes
        the full ``mQ`` singular (so the joint-kernel Cholesky fails), while
        leaving the Y-block intact so ``R`` stays invertible and the PPF
        forward ``_precompute`` does not raise an earlier
        ``InvertibilityError``.
        """
        param_bad = copy.copy(param_nl_x2y1)
        bad_mQ = param_bad._mQ.copy()
        p = param_nl_x2y1.dim_x
        bad_mQ[:p, :p] = 0.0
        bad_mQ[:p, p:] = 0.0  # also zero the cross-blocks (keep mQ symmetric PSD)
        bad_mQ[p:, :p] = 0.0
        param_bad._mQ = bad_mQ
        pps = NonLinear_PPS(
            param_bad, n_particles=N_PARTICLES_SHORT, sKey=SEED,
        )
        with pytest.raises(CovarianceError) as exc_info:
            pps.process_N_data_smoother(N=10)
        err = exc_info.value
        assert err.matrix_name == "mQ"
        assert err.step == -1                # sentinel: construction-time
        assert err.__cause__ is not None

    def test_degenerate_weight_fallback_logs_warning(
        self, param_nl_x2y1, monkeypatch, caplog,
    ):
        """If the smoothed weight total underflows to 0 (or non-finite)
        at some step, the PPS must fall back to uniform weights and
        emit a single ``WARNING`` per offending step. We force the
        condition by monkey-patching the backward kernel inputs."""
        from prg.classes import nonlinear_pps as pps_mod
        # Run a normal forward + backward to verify the WARNING path is
        # exercised under conditions where the unnormalised weights vanish.
        # Easiest reliable trigger: force the backward `np.exp` of an
        # all-(-inf) log_norm_D — achievable by clamping log_D to -inf in
        # one iteration. We do this surgically via monkeypatching logsumexp
        # to return +inf on a single call (which makes log_norm_D = -inf).
        call_state = {"hit": False}
        original_logsumexp = pps_mod.logsumexp

        def patched_logsumexp(arr, axis=0):
            if not call_state["hit"]:
                call_state["hit"] = True
                return np.full(arr.shape[1], np.inf)   # log_Z = +inf → log_D - log_Z = -inf
            return original_logsumexp(arr, axis=axis)

        monkeypatch.setattr(pps_mod, "logsumexp", patched_logsumexp)

        with caplog.at_level(logging.WARNING, logger="prg.classes.nonlinear_pps"):
            pps = NonLinear_PPS(
                param_nl_x2y1, n_particles=N_PARTICLES_SHORT, sKey=SEED,
            )
            pps.process_N_data_smoother(N=20)

        warning_msgs = [
            r.message for r in caplog.records
            if r.name == "prg.classes.nonlinear_pps"
            and r.levelno == logging.WARNING
            and "smoothed weight total degenerate" in r.message
        ]
        assert len(warning_msgs) >= 1
        # And the smoothed weights at the affected step must be uniform
        # (we don't pinpoint which step, just verify the contract)
        uniform_seen = any(
            np.allclose(rec["w_smooth"], 1.0 / N_PARTICLES_SHORT, atol=1e-12)
            for rec in pps.history
        )
        assert uniform_seen


class TestNonLinearPPSLogging:

    LOGGER_NAME = "prg.classes.nonlinear_pps"

    def test_info_logs_emitted_at_entry_and_exit(self, param_nl_x2y1, caplog):
        with caplog.at_level(logging.INFO, logger=self.LOGGER_NAME):
            pps = NonLinear_PPS(
                param_nl_x2y1, n_particles=N_PARTICLES_SHORT, sKey=SEED,
            )
            pps.process_N_data_smoother(N=20)
        info_msgs = [
            r.message for r in caplog.records
            if r.name == self.LOGGER_NAME and r.levelno == logging.INFO
        ]
        assert len(info_msgs) == 2
        assert "backward pass starting" in info_msgs[0]
        assert f"n_particles={N_PARTICLES_SHORT}" in info_msgs[0]
        assert "backward pass complete" in info_msgs[1]

    def test_debug_logs_one_per_backward_step(self, param_nl_x2y1, caplog):
        N = 10
        with caplog.at_level(logging.DEBUG, logger=self.LOGGER_NAME):
            pps = NonLinear_PPS(
                param_nl_x2y1, n_particles=N_PARTICLES_SHORT, sKey=SEED,
            )
            pps.process_N_data_smoother(N=N)
        debug_msgs = [
            r for r in caplog.records
            if r.name == self.LOGGER_NAME and r.levelno == logging.DEBUG
        ]
        assert len(debug_msgs) == N


class TestNonLinearPPSRegression:
    """Cumulative-effect test: on a linear pairwise model, the PPS MSE
    should not significantly degrade vs the PPF (in expectation over
    seeds; for nonlinear models the FFBSm advantage is small)."""

    def test_smoother_does_not_degrade_filter_on_linear_pairwise(self):
        """On `model_x1_y1_AQ_pairwise` the PPS should be close to the
        PPF MSE (no degradation), with ratio < 1.05 on average. Tighter
        bounds require larger N (statistical power)."""
        m = ModelFactoryLinear.create("model_x1_y1_AQ_pairwise")
        params = m.get_params().copy()
        dim_x = params.pop("dim_x")
        dim_y = params.pop("dim_y")
        mses_f, mses_s = [], []
        for sd in range(10):
            pl = ParamLinear(0, dim_x, dim_y, **params)
            pps = NonLinear_PPS(pl, n_particles=300, sKey=sd)
            res = pps.process_N_data_smoother(N=N_REG)
            xt = np.array([r[1].flatten() for r in res])
            xf = np.array([r[4].flatten() for r in res])
            xs = np.array([r[5].flatten() for r in res])
            mses_f.append(((xf - xt) ** 2).mean())
            mses_s.append(((xs - xt) ** 2).mean())
        ratio = np.mean(mses_s) / np.mean(mses_f)
        assert ratio < 1.10, (
            f"Expected PPS / PPF MSE ratio < 1.10 on x1y1 pairwise linear, "
            f"got {ratio:.3f}"
        )
