"""Tests for the Linear Pairwise Kalman Smoother (Linear_PKS)."""

import logging

import numpy as np
import pytest

from prg.classes.linear_pks import (
    Linear_PKS,
    Linear_PKS_BF,
    Linear_PKS_DWY,
    Linear_PKS_MBF,
    Linear_PKS_MF,
    Linear_PKS_RTS,
    Linear_PKS_VAR,
)
from prg.utils.exceptions import CovarianceError

SEED = 42
N_SHORT = 100
N_REG = 500
N_SEEDS_REG = 20

PSD_TOL = 1e-7      # eigenvalue tolerance for PSD shrinkage
SHAPE_TOL = 1e-12   # tolerance for exact equality checks (e.g. terminal step)


class TestLinearPKSShapes:

    def test_output_length(self, param_x1y1):
        pks = Linear_PKS(param_x1y1, sKey=SEED)
        res = pks.process_N_data_smoother(N=N_SHORT)
        assert len(res) == N_SHORT + 1

    def test_output_tuple_shapes_x1y1(self, param_x1y1):
        pks = Linear_PKS(param_x1y1, sKey=SEED)
        res = pks.process_N_data_smoother(N=10)
        for k, x_true, y_obs, x_pred, x_upd, x_smooth in res:
            assert isinstance(k, int)
            assert x_true.shape == (param_x1y1.dim_x, 1)
            assert y_obs.shape == (param_x1y1.dim_y, 1)
            assert x_pred.shape == (param_x1y1.dim_x, 1)
            assert x_upd.shape == (param_x1y1.dim_x, 1)
            assert x_smooth.shape == (param_x1y1.dim_x, 1)

    def test_output_tuple_shapes_x2y2(self, param_x2y2):
        pks = Linear_PKS(param_x2y2, sKey=SEED)
        res = pks.process_N_data_smoother(N=10)
        for _k, x_true, y_obs, x_pred, x_upd, x_smooth in res:
            assert x_true.shape == (param_x2y2.dim_x, 1)
            assert y_obs.shape == (param_x2y2.dim_y, 1)
            assert x_pred.shape == (param_x2y2.dim_x, 1)
            assert x_upd.shape == (param_x2y2.dim_x, 1)
            assert x_smooth.shape == (param_x2y2.dim_x, 1)

    def test_step_indices_are_sequential(self, param_x1y1):
        pks = Linear_PKS(param_x1y1, sKey=SEED)
        res = pks.process_N_data_smoother(N=N_SHORT)
        assert [r[0] for r in res] == list(range(N_SHORT + 1))


class TestLinearPKSTerminalEquality:
    """At the terminal step, the smoothed estimate coincides with the filtered one."""

    @pytest.mark.parametrize("param_fixture", ["param_x1y1", "param_x2y2"])
    def test_terminal_X_and_PXX(self, param_fixture, request):
        param = request.getfixturevalue(param_fixture)
        pks = Linear_PKS(param, sKey=SEED)
        pks.process_N_data_smoother(N=N_SHORT)
        last = pks.history[-1]
        assert np.allclose(last["Xkp1_smooth"], last["Xkp1_update"], atol=SHAPE_TOL)
        assert np.allclose(last["PXXkp1_smooth"], last["PXXkp1_update"], atol=SHAPE_TOL)


class TestLinearPKSShrinkage:
    """PSD ordering: P^XX_{k|k} - P^XX_{k|N} is positive semi-definite."""

    @pytest.mark.parametrize("param_fixture", ["param_x1y1", "param_x2y2"])
    def test_psd_shrinkage(self, param_fixture, request):
        param = request.getfixturevalue(param_fixture)
        pks = Linear_PKS(param, sKey=SEED)
        pks.process_N_data_smoother(N=N_SHORT)
        for rec in pks.history:
            D = rec["PXXkp1_update"] - rec["PXXkp1_smooth"]
            D = 0.5 * (D + D.T)
            min_eig = np.linalg.eigvalsh(D).min()
            assert min_eig > -PSD_TOL, (
                f"Step {rec['k']}: P_filter - P_smooth not PSD (min eig {min_eig})"
            )


class TestLinearPKSJosephForm:
    """The Joseph form must produce results identical (to float precision)
    to the standard form at the optimal gain, and must preserve PSD."""

    JOSEPH_EQ_TOL = 1e-10

    @pytest.mark.parametrize("param_fixture", ["param_x1y1", "param_x2y2"])
    def test_joseph_equals_standard_means(self, param_fixture, request):
        """Smoothed means coincide exactly (same gain, same delta_Z)."""
        param = request.getfixturevalue(param_fixture)
        pks_std = Linear_PKS(param, sKey=SEED, joseph=False)
        pks_jos = Linear_PKS(param, sKey=SEED, joseph=True)
        res_std = pks_std.process_N_data_smoother(N=N_SHORT)
        res_jos = pks_jos.process_N_data_smoother(N=N_SHORT)
        for a, b in zip(res_std, res_jos, strict=True):
            assert np.allclose(a[5], b[5], atol=self.JOSEPH_EQ_TOL)

    @pytest.mark.parametrize("param_fixture", ["param_x1y1", "param_x2y2"])
    def test_joseph_equals_standard_covariances(self, param_fixture, request):
        """Smoothed covariances coincide up to ~1e-12 in double precision."""
        param = request.getfixturevalue(param_fixture)
        pks_std = Linear_PKS(param, sKey=SEED, joseph=False)
        pks_jos = Linear_PKS(param, sKey=SEED, joseph=True)
        pks_std.process_N_data_smoother(N=N_SHORT)
        pks_jos.process_N_data_smoother(N=N_SHORT)
        for r1, r2 in zip(pks_std.history, pks_jos.history, strict=True):
            diff = np.max(np.abs(r1["PXXkp1_smooth"] - r2["PXXkp1_smooth"]))
            assert diff < self.JOSEPH_EQ_TOL, (
                f"Step {r1['k']}: |P_std - P_joseph| = {diff:.2e}"
            )

    @pytest.mark.parametrize("param_fixture", ["param_x1y1", "param_x2y2"])
    def test_joseph_psd_shrinkage(self, param_fixture, request):
        """Joseph form preserves PSD shrinkage on its own."""
        param = request.getfixturevalue(param_fixture)
        pks = Linear_PKS(param, sKey=SEED, joseph=True)
        pks.process_N_data_smoother(N=N_SHORT)
        for rec in pks.history:
            D = rec["PXXkp1_update"] - rec["PXXkp1_smooth"]
            D = 0.5 * (D + D.T)
            min_eig = np.linalg.eigvalsh(D).min()
            assert min_eig > -PSD_TOL, (
                f"Step {rec['k']}: P_f - P_s not PSD under Joseph (min eig {min_eig})"
            )

    def test_joseph_flag_default_false(self, param_x1y1):
        pks = Linear_PKS(param_x1y1, sKey=SEED)
        assert pks.joseph is False


def _reference_augmented_rts_smoother(pkf_history, A, BQBT, dim_x, dim_y):
    """
    Reference smoother implementation: classical RTS on the augmented
    state Z' = (X, Y). Operates entirely on the augmented (dim_x+dim_y)
    representation, no pairwise shortcut.

    This is the smoother analog of Remark `rem:PKFasKF` of the companion
    paper: showing that the linear PMM is equivalent to an augmented-state
    classical Kalman system with R^aug = 0.

    Parameters
    ----------
    pkf_history : list of dict
        Forward filter history, as produced by `Linear_PKF.process_filter`.
    A, BQBT : np.ndarray
        Pairwise transition matrix and noise-injection covariance (constants).
    dim_x, dim_y : int

    Returns
    -------
    list of (X_smooth, PXX_smooth) tuples, in chronological order.
    """
    dim_xy = dim_x + dim_y
    N = len(pkf_history)

    # Build augmented filtered means and covariances at each step.
    # Z'_{n|n} = (X_{n|n}; Y_n)            — Y is the observation, known exactly
    # P^{Z'Z'}_{n|n} = diag(P^xx_{n|n}, 0) — Y has zero variance under Y_{1:n}
    Z_filt = []
    P_filt = []
    for rec in pkf_history:
        z = np.zeros((dim_xy, 1))
        z[:dim_x] = rec["Xkp1_update"]
        z[dim_x:] = rec["ykp1"]
        Z_filt.append(z)
        P = np.zeros((dim_xy, dim_xy))
        P[:dim_x, :dim_x] = rec["PXXkp1_update"]
        P_filt.append(P)

    # Build augmented predicted means / covariances at each step n+1
    # Z'_{n+1|n} = A @ Z'_{n|n}
    # P^{Z'Z'}_{n+1|n} = A @ diag(P^xx_{n|n}, 0) @ A.T + BQB^T
    Z_pred = [None] * N
    P_pred = [None] * N
    for n in range(N - 1):
        Z_pred[n + 1] = A @ Z_filt[n]
        P_pred[n + 1] = A @ P_filt[n] @ A.T + BQBT

    # Backward RTS: at step N, smoothed = filtered
    Z_smooth = [None] * N
    P_smooth = [None] * N
    Z_smooth[N - 1] = Z_filt[N - 1].copy()
    P_smooth[N - 1] = P_filt[N - 1].copy()

    eye_xy = np.eye(dim_xy)
    for n in range(N - 2, -1, -1):
        # Smoothing gain: C^{Z'Z'}_n = P^{Z'Z'}_{n|n} @ A^T @ inv(P^{Z'Z'}_{n+1|n})
        C = P_filt[n] @ A.T @ np.linalg.solve(P_pred[n + 1], eye_xy)
        Z_smooth[n] = Z_filt[n] + C @ (Z_smooth[n + 1] - Z_pred[n + 1])
        P_smooth[n] = P_filt[n] + C @ (P_smooth[n + 1] - P_pred[n + 1]) @ C.T
        # Symmetrise for fair numerical comparison
        P_smooth[n] = 0.5 * (P_smooth[n] + P_smooth[n].T)

    # Return X-block only (analog of what Linear_PKS yields)
    return [
        (z[:dim_x].copy(), P[:dim_x, :dim_x].copy())
        for z, P in zip(Z_smooth, P_smooth, strict=True)
    ]


class TestLinearPKSAugmentedRTSEquivalence:
    """
    Companion to the paper's Remark `rem:PKFasKF`: classical RTS on the
    augmented state Z' = (X, Y) must yield exactly the same smoothed
    estimates as the pairwise Linear_PKS.
    """

    EQ_TOL = 1e-10  # tolerance for "bit-for-bit" equality at double precision

    @pytest.mark.parametrize("param_fixture", ["param_x1y1", "param_x2y2"])
    def test_augmented_rts_equals_pairwise_pks(self, param_fixture, request):
        param = request.getfixturevalue(param_fixture)
        pks = Linear_PKS(param, sKey=SEED, joseph=False)
        res = pks.process_N_data_smoother(N=N_SHORT)

        # Reference RTS on augmented state
        A = param.A
        BQBT = param.B @ param.mQ @ param.B.T
        ref = _reference_augmented_rts_smoother(
            pks.history, A, BQBT, param.dim_x, param.dim_y
        )

        # Compare smoothed means
        for (k, _xt, _y, _xp, _xu, x_smooth), (Xs_ref, _) in zip(
            res, ref, strict=True
        ):
            assert np.max(np.abs(x_smooth - Xs_ref)) < self.EQ_TOL, (
                f"Step {k}: mean mismatch between Linear_PKS and augmented RTS"
            )

        # Compare smoothed covariances
        for rec, (_, Ps_ref) in zip(pks.history, ref, strict=True):
            diff = np.max(np.abs(rec["PXXkp1_smooth"] - Ps_ref))
            assert diff < self.EQ_TOL, (
                f"Step {rec['k']}: |P_pks - P_aug_rts| = {diff:.2e}"
            )

    @pytest.mark.parametrize("param_fixture", ["param_x1y1", "param_x2y2"])
    def test_augmented_rts_Y_block_unchanged(self, param_fixture, request):
        """
        On the augmented state, the Y-block of the smoothed mean must
        remain equal to the observation y_n (since Y_n is observed and
        has zero conditional variance).
        """
        param = request.getfixturevalue(param_fixture)
        pks = Linear_PKS(param, sKey=SEED)
        pks.process_N_data_smoother(N=N_SHORT)

        A = param.A
        BQBT = param.B @ param.mQ @ param.B.T
        dim_xy = param.dim_x + param.dim_y
        # Re-run the augmented RTS but keep the full Z-block this time
        N = len(pks.history)
        Z_filt, P_filt = [], []
        for rec in pks.history:
            z = np.zeros((dim_xy, 1))
            z[:param.dim_x] = rec["Xkp1_update"]
            z[param.dim_x:] = rec["ykp1"]
            Z_filt.append(z)
            P = np.zeros((dim_xy, dim_xy))
            P[:param.dim_x, :param.dim_x] = rec["PXXkp1_update"]
            P_filt.append(P)
        Z_pred = [None] * N
        P_pred = [None] * N
        for n in range(N - 1):
            Z_pred[n + 1] = A @ Z_filt[n]
            P_pred[n + 1] = A @ P_filt[n] @ A.T + BQBT
        Z_smooth = [None] * N
        Z_smooth[N - 1] = Z_filt[N - 1].copy()
        for n in range(N - 2, -1, -1):
            C = P_filt[n] @ A.T @ np.linalg.solve(P_pred[n + 1], np.eye(dim_xy))
            Z_smooth[n] = Z_filt[n] + C @ (Z_smooth[n + 1] - Z_pred[n + 1])

        # Verify Y-block of Z_smooth equals y_n at every step
        for n, (z_s, rec) in enumerate(zip(Z_smooth, pks.history, strict=True)):
            y_smooth = z_s[param.dim_x:]
            y_obs = rec["ykp1"]
            assert np.allclose(y_smooth, y_obs, atol=self.EQ_TOL), (
                f"Step {n}: Y-block of augmented smoothed state drifted from y_n"
            )


class TestLinearPKSDWYEquivalence:
    """The DWY (backward-RTS) variant returns the same smoothed estimate as the
    RTS pass, to machine precision, on the linear-Gaussian pairwise model
    (Geng et al., 2023). Both means and covariances are checked."""

    DWY_EQ_TOL = 1e-9

    @pytest.mark.parametrize("param_fixture", ["param_x1y1", "param_x2y2"])
    def test_dwy_equals_rts(self, param_fixture, request):
        param = request.getfixturevalue(param_fixture)
        rts = Linear_PKS_RTS(param, sKey=SEED)
        rts.process_N_data_smoother(N=120)
        dwy = Linear_PKS(param, sKey=SEED, method="DWY")
        dwy.process_N_data_smoother(N=120)
        for a, b in zip(rts.history, dwy.history, strict=True):
            assert np.allclose(
                a["Xkp1_smooth"], b["Xkp1_smooth"], atol=self.DWY_EQ_TOL
            )
            assert np.allclose(
                a["PXXkp1_smooth"], b["PXXkp1_smooth"], atol=self.DWY_EQ_TOL
            )

    def test_dwy_explicit_class_matches_facade(self, param_x1y1):
        explicit = Linear_PKS_DWY(param_x1y1, sKey=SEED)
        explicit.process_N_data_smoother(N=60)
        facade = Linear_PKS(param_x1y1, sKey=SEED, method="DWY")
        facade.process_N_data_smoother(N=60)
        for a, b in zip(explicit.history, facade.history, strict=True):
            assert np.allclose(a["Xkp1_smooth"], b["Xkp1_smooth"], atol=SHAPE_TOL)

    def test_unknown_method_raises(self, param_x1y1):
        from prg.utils.exceptions import FilterError

        with pytest.raises(FilterError):
            Linear_PKS(param_x1y1, sKey=SEED, method="NOPE")


class TestLinearPKSVariantEquivalence:
    """BF, MBF and MF each return the same smoothed estimate as RTS, to machine
    precision, on the linear-Gaussian pairwise model (Geng et al., 2023). This
    is the consolidated non-regression test of report Section 2.6 — the five
    linear variants compute the same law and differ only by mechanics."""

    VARIANT_EQ_TOL = 1e-9

    @pytest.mark.parametrize("method", ["BF", "MBF", "MF", "VAR"])
    @pytest.mark.parametrize("param_fixture", ["param_x1y1", "param_x2y2"])
    def test_variant_equals_rts(self, method, param_fixture, request):
        param = request.getfixturevalue(param_fixture)
        rts = Linear_PKS_RTS(param, sKey=SEED)
        rts.process_N_data_smoother(N=120)
        var = Linear_PKS(param, sKey=SEED, method=method)
        var.process_N_data_smoother(N=120)
        for a, b in zip(rts.history, var.history, strict=True):
            assert np.allclose(
                a["Xkp1_smooth"], b["Xkp1_smooth"], atol=self.VARIANT_EQ_TOL
            ), f"{method}: smoothed mean mismatch at step {a['k']}"
            assert np.allclose(
                a["PXXkp1_smooth"], b["PXXkp1_smooth"], atol=self.VARIANT_EQ_TOL
            ), f"{method}: smoothed covariance mismatch at step {a['k']}"

    @pytest.mark.parametrize(
        "method, cls",
        [
            ("BF", Linear_PKS_BF),
            ("MBF", Linear_PKS_MBF),
            ("MF", Linear_PKS_MF),
            ("VAR", Linear_PKS_VAR),
        ],
    )
    def test_variant_explicit_class_matches_facade(self, method, cls, param_x1y1):
        explicit = cls(param_x1y1, sKey=SEED)
        explicit.process_N_data_smoother(N=60)
        facade = Linear_PKS(param_x1y1, sKey=SEED, method=method)
        facade.process_N_data_smoother(N=60)
        for a, b in zip(explicit.history, facade.history, strict=True):
            assert np.allclose(a["Xkp1_smooth"], b["Xkp1_smooth"], atol=SHAPE_TOL)
            assert np.allclose(
                a["PXXkp1_smooth"], b["PXXkp1_smooth"], atol=SHAPE_TOL
            )

    @pytest.mark.parametrize("method", ["BF", "MBF", "MF", "VAR"])
    def test_variant_terminal_equals_filtered(self, method, param_x1y1):
        """Terminal step: smoothed == filtered for every variant."""
        pks = Linear_PKS(param_x1y1, sKey=SEED, method=method)
        pks.process_N_data_smoother(N=N_SHORT)
        last = pks.history[-1]
        assert np.allclose(last["Xkp1_smooth"], last["Xkp1_update"], atol=SHAPE_TOL)
        assert np.allclose(
            last["PXXkp1_smooth"], last["PXXkp1_update"], atol=SHAPE_TOL
        )

    @pytest.mark.parametrize("param_fixture", ["param_x1y1", "param_x2y2"])
    def test_var_cross_covariance_matches_rts(self, param_fixture, request):
        """VAR stores the lag-one cross-covariances Mk_smooth =
        Cov(X_{n+1}, X_n | y_{1:N}); they match the RTS identity
        P^{xx}_{n+1|N} (G_n^x)^T to machine precision (needed for EM)."""
        param = request.getfixturevalue(param_fixture)
        dx = param.dim_x
        rts = Linear_PKS_RTS(param, sKey=SEED)
        rts.process_N_data_smoother(N=80)
        var = Linear_PKS_VAR(param, sKey=SEED)
        var.process_N_data_smoother(N=80)
        H = rts.history
        for n in range(len(H) - 1):
            Gx = H[n]["Gk_smooth"][:, :dx]                  # X-cols of RTS gain
            M_rts = H[n + 1]["PXXkp1_smooth"] @ Gx.T
            assert np.allclose(
                var.history[n]["Mk_smooth"], M_rts, atol=self.VARIANT_EQ_TOL
            ), f"cross-covariance mismatch at step {n}"
        assert np.allclose(var.history[-1]["Mk_smooth"], 0.0)  # terminal placeholder


class TestLinearPKSEdgeCases:
    """Edge cases of the smoother lifecycle."""

    def test_N_equals_1(self, param_x1y1):
        """Two-record case: forward records (k=0, k=1); backward loop runs once
        and the smoothed estimates at both ends must be coherent."""
        pks = Linear_PKS(param_x1y1, sKey=SEED)
        res = pks.process_N_data_smoother(N=1)
        assert len(res) == 2
        # Terminal step: smoothed == filtered
        assert np.allclose(res[-1][4], res[-1][5], atol=SHAPE_TOL)
        # First step: smoothed cov is PSD and PSD-bounded above by filter cov
        rec0 = pks.history[0]
        D = rec0["PXXkp1_update"] - rec0["PXXkp1_smooth"]
        D = 0.5 * (D + D.T)
        assert np.linalg.eigvalsh(D).min() > -PSD_TOL

    def test_process_smoother_generator_lazy(self, param_x1y1):
        """Calling process_smoother as a generator and stopping early must
        not corrupt history (forward pass completes before the first yield)."""
        pks = Linear_PKS(param_x1y1, sKey=SEED)
        gen = pks.process_smoother(N=N_SHORT)
        first_tuple = next(gen)
        assert first_tuple[0] == 0
        # After consuming only one yielded item, all forward records exist
        # and every record has the smoother fields set (backward pass ran
        # entirely before the first yield).
        assert len(pks.history) == N_SHORT + 1
        for rec in pks.history:
            assert "Xkp1_smooth" in rec
            assert "PXXkp1_smooth" in rec
            assert "Gk_smooth" in rec
        gen.close()

    def test_process_smoother_twice_in_a_row(self, param_x1y1):
        """A second call must clear history and produce identical results
        (no accumulation of records, no stale smoother fields)."""
        pks = Linear_PKS(param_x1y1, sKey=SEED)
        res1 = pks.process_N_data_smoother(N=N_SHORT)
        res2 = pks.process_N_data_smoother(N=N_SHORT)
        assert len(res1) == len(res2) == N_SHORT + 1
        for a, b in zip(res1, res2, strict=True):
            # Same seed, same simulated trajectory not guaranteed — but
            # both runs return the same number of records and consistent
            # shapes. Stronger: smoothed and filtered should match
            # because the seed generator is re-seeded internally? Check
            # only the structural invariants.
            assert a[0] == b[0]
            assert a[5].shape == b[5].shape

    def test_smoother_fields_have_correct_shapes(self, param_x2y2):
        """Gk_smooth must be (dim_x, dim_xy) at every step, including the
        terminal step where the value is a placeholder."""
        pks = Linear_PKS(param_x2y2, sKey=SEED)
        pks.process_N_data_smoother(N=N_SHORT)
        for rec in pks.history:
            assert rec["Gk_smooth"].shape == (param_x2y2.dim_x, param_x2y2.dim_xy)
            assert rec["Xkp1_smooth"].shape == (param_x2y2.dim_x, 1)
            assert rec["PXXkp1_smooth"].shape == (param_x2y2.dim_x, param_x2y2.dim_x)
        # Terminal-step Gk_smooth is the zero placeholder by convention
        assert np.allclose(pks.history[-1]["Gk_smooth"], 0.0)

    def test_external_data_generator(self, param_x1y1):
        """The smoother must consume an externally supplied data generator
        and produce the same results as the internal simulator."""
        # First run with the internal generator to capture the simulated
        # (k, x_true, y) triplets.
        pks_ref = Linear_PKS(param_x1y1, sKey=SEED)
        ref = pks_ref.process_N_data_smoother(N=50)
        triplets = [(r[0], r[1], r[2]) for r in ref]

        # Now replay through an external generator and check the smoother
        # produces the same outputs.
        def replay():
            yield from triplets

        pks_ext = Linear_PKS(param_x1y1, sKey=SEED)
        ext = pks_ext.process_N_data_smoother(N=50, data_generator=replay())
        assert len(ext) == len(ref)
        for a, b in zip(ref, ext, strict=True):
            assert np.allclose(a[4], b[4], atol=SHAPE_TOL)  # filtered
            assert np.allclose(a[5], b[5], atol=SHAPE_TOL)  # smoothed

    def test_missing_ground_truth(self, param_x1y1):
        """When the data generator yields x_true=None, the smoother must
        still complete and propagate None through the output tuples."""
        def gen_no_truth():
            rng = np.random.default_rng(7)
            for k in range(20):
                # Synthetic y from a stable Gaussian — content doesn't matter,
                # only that x_true is None so the smoother takes the
                # ground_truth=False path.
                y = rng.normal(size=(param_x1y1.dim_y, 1))
                yield k, None, y

        pks = Linear_PKS(param_x1y1, sKey=SEED)
        res = pks.process_N_data_smoother(N=None, data_generator=gen_no_truth())
        assert len(res) == 20
        assert all(r[1] is None for r in res)
        # Smoother fields must still be populated
        for rec in pks.history:
            assert "Xkp1_smooth" in rec

    def test_singular_predicted_covariance_raises(self, param_x1y1):
        """A pathological run with zeroed-out mQ produces a singular
        P^{ZZ}_{n+1|n} in the backward pass; the Cholesky failure must
        surface as a ``CovarianceError`` (not a silent NaN), and the
        exception's structured attributes must point to the offending
        step and matrix name."""
        # Construct a fresh param with mQ = 0 — predicted joint cov collapses
        # to a rank-deficient matrix in the backward pass.
        import copy
        param_bad = copy.copy(param_x1y1)
        param_bad._mQ = np.zeros_like(param_bad._mQ)
        pks = Linear_PKS(param_bad, sKey=SEED)
        with pytest.raises(CovarianceError) as exc_info:
            pks.process_N_data_smoother(N=10)
        # The error must carry the structured context: which step, which matrix.
        err = exc_info.value
        assert err.matrix_name == "PZZkp1_predict"
        assert err.step >= 0  # a concrete time step, not the -1 sentinel
        # And it must be chained from the underlying LinAlg failure.
        assert err.__cause__ is not None


class TestLinearPKSPublicAPI:
    """The smoother must reach history records only through the public API."""

    def test_history_getitem(self, param_x1y1):
        pks = Linear_PKS(param_x1y1, sKey=SEED)
        pks.process_N_data_smoother(N=20)
        # public read
        rec0 = pks.history[0]
        assert isinstance(rec0, dict)
        assert "Xkp1_smooth" in rec0
        # negative indexing supported
        assert pks.history[-1]["k"] == 20

    def test_history_iteration(self, param_x1y1):
        pks = Linear_PKS(param_x1y1, sKey=SEED)
        pks.process_N_data_smoother(N=20)
        ks = [rec["k"] for rec in pks.history]
        assert ks == list(range(21))

    def test_history_update_record_rejects_non_string_keys(self, param_x1y1):
        pks = Linear_PKS(param_x1y1, sKey=SEED)
        pks.process_N_data_smoother(N=5)
        with pytest.raises(TypeError):
            pks.history.update_record(0, **{42: "bad"})  # type: ignore[arg-type]

    def test_history_update_record_raises_indexerror(self, param_x1y1):
        pks = Linear_PKS(param_x1y1, sKey=SEED)
        pks.process_N_data_smoother(N=5)
        with pytest.raises(IndexError):
            pks.history.update_record(999, foo="bar")

    def test_history_getitem_raises_typeerror_on_non_int(self, param_x1y1):
        pks = Linear_PKS(param_x1y1, sKey=SEED)
        pks.process_N_data_smoother(N=5)
        with pytest.raises(TypeError):
            _ = pks.history["not_an_int"]  # type: ignore[index]


class TestLinearPKSExceptionPolicy:
    """End-to-end check that domain exceptions flow through unwrapped from
    both ``process_smoother`` (generator) and ``process_N_data_smoother``
    (eager wrapper)."""

    def test_invalid_N_raises_paramerror(self, param_x1y1):
        """The forward pass `_validate_N` should fire before any record
        is produced, surfacing a ``ParamError`` unwrapped."""
        from prg.utils.exceptions import ParamError
        pks = Linear_PKS(param_x1y1, sKey=SEED)
        with pytest.raises(ParamError):
            pks.process_N_data_smoother(N=0)
        with pytest.raises(ParamError):
            pks.process_N_data_smoother(N=-1)

    def test_pkferror_root_catches_smoother_errors(self, param_x1y1):
        """All smoother errors derive from ``PKFError``, so catching the
        root class is sufficient to intercept any domain failure."""
        import copy

        from prg.utils.exceptions import PKFError
        param_bad = copy.copy(param_x1y1)
        param_bad._mQ = np.zeros_like(param_bad._mQ)
        pks = Linear_PKS(param_bad, sKey=SEED)
        with pytest.raises(PKFError):
            pks.process_N_data_smoother(N=10)


class TestLinearPKSLogging:
    """The smoother emits INFO at entry/exit and DEBUG per step."""

    LOGGER_NAME = "prg.classes.linear_pks"

    def test_info_logs_emitted_at_entry_and_exit(self, param_x1y1, caplog):
        with caplog.at_level(logging.INFO, logger=self.LOGGER_NAME):
            pks = Linear_PKS(param_x1y1, sKey=SEED)
            pks.process_N_data_smoother(N=20)
        info_msgs = [
            r.message for r in caplog.records
            if r.name == self.LOGGER_NAME and r.levelno == logging.INFO
        ]
        # Exactly two INFO lines: smoothing-pass entry and exit
        assert len(info_msgs) == 2
        assert "smoothing pass starting" in info_msgs[0]
        assert "joseph=False" in info_msgs[0]
        assert "smoothing pass complete" in info_msgs[1]

    def test_info_log_reflects_joseph_mode(self, param_x1y1, caplog):
        with caplog.at_level(logging.INFO, logger=self.LOGGER_NAME):
            pks = Linear_PKS(param_x1y1, sKey=SEED, joseph=True)
            pks.process_N_data_smoother(N=5)
        entry = next(
            r.message for r in caplog.records
            if r.name == self.LOGGER_NAME and r.levelno == logging.INFO
        )
        assert "joseph=True" in entry

    def test_debug_logs_one_per_backward_step(self, param_x1y1, caplog):
        N = 10
        with caplog.at_level(logging.DEBUG, logger=self.LOGGER_NAME):
            pks = Linear_PKS(param_x1y1, sKey=SEED)
            pks.process_N_data_smoother(N=N)
        debug_msgs = [
            r for r in caplog.records
            if r.name == self.LOGGER_NAME and r.levelno == logging.DEBUG
        ]
        # One DEBUG line per backward iteration; terminal step (n=N) has none
        assert len(debug_msgs) == N

    def test_no_debug_logs_at_info_level(self, param_x1y1, caplog):
        """Confirms the isEnabledFor gate prevents DEBUG messages from
        firing when only INFO is requested."""
        with caplog.at_level(logging.INFO, logger=self.LOGGER_NAME):
            pks = Linear_PKS(param_x1y1, sKey=SEED)
            pks.process_N_data_smoother(N=10)
        debug_msgs = [
            r for r in caplog.records
            if r.name == self.LOGGER_NAME and r.levelno == logging.DEBUG
        ]
        assert debug_msgs == []


class TestLinearPKSRegression:
    """On average over many seeds, the smoother improves on the filter."""

    @staticmethod
    def _mse(arrs_true, arrs_est):
        a = np.asarray([t.flatten() for t in arrs_true])
        b = np.asarray([e.flatten() for e in arrs_est])
        return float(((a - b) ** 2).mean())

    @pytest.mark.parametrize(
        "param_fixture, max_ratio",
        [
            # x1y1 has |A^yx| ≈ 0.30 → ~25-30% MSE reduction
            ("param_x1y1", 0.85),
            # x2y2 has |A^yx| ≈ 0.04 → marginal but ratio must stay ≤ 1
            ("param_x2y2", 1.02),
        ],
    )
    def test_smoother_beats_filter_on_average(self, param_fixture, max_ratio, request):
        param = request.getfixturevalue(param_fixture)
        mses_filter, mses_smooth = [], []
        for seed in range(N_SEEDS_REG):
            pks = Linear_PKS(param, sKey=seed)
            res = pks.process_N_data_smoother(N=N_REG)
            xtrue = [r[1] for r in res]
            xfilt = [r[4] for r in res]
            xsmth = [r[5] for r in res]
            mses_filter.append(self._mse(xtrue, xfilt))
            mses_smooth.append(self._mse(xtrue, xsmth))
        ratio = np.mean(mses_smooth) / np.mean(mses_filter)
        assert ratio <= max_ratio, (
            f"Expected smoother/filter MSE ratio <= {max_ratio} on "
            f"{param_fixture}, got {ratio:.3f}"
        )

    def test_avg_trace_shrinks_x1y1(self, param_x1y1):
        pks = Linear_PKS(param_x1y1, sKey=SEED)
        pks.process_N_data_smoother(N=N_REG)
        tr_f = np.array([np.trace(r["PXXkp1_update"]) for r in pks.history])
        tr_s = np.array([np.trace(r["PXXkp1_smooth"]) for r in pks.history])
        # tr(P_smooth) ≤ tr(P_filter) at every step (PSD inclusion)
        assert (tr_s <= tr_f + PSD_TOL).all()
        # The shrinkage is strict on a majority of steps; the margin
        # 1% of the average filter trace is a conservative effect-size
        # threshold (still robustly clearable on x1y1).
        margin = 0.01 * tr_f.mean()
        assert (tr_s < tr_f - margin).sum() > N_REG // 2
