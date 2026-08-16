"""Tests for missing-observation (all-NaN) handling in the Linear PKF.

In a pairwise model ``y`` is a component of the Markov chain, so a gap cannot
be handled by the classical "skip the update" recipe: the correct treatment
marginalises over the missing ``y``, which leaves nonzero Y and cross blocks
in the joint covariance. These tests validate the implementation against an
independent exact joint-Gaussian reference recursion.
"""

import numpy as np
import pytest

from prg.classes.linear_pkf import Linear_PKF
from prg.classes.linear_pks import Linear_PKS
from prg.classes.nonlinear_epkf import NonLinear_EPKF
from prg.classes.nonlinear_pf import NonLinear_PF
from prg.classes.nonlinear_ppf import NonLinear_PPF
from prg.classes.nonlinear_upkf import NonLinear_UPKF
from prg.learning.em_partial_dynamics import estimate_dynamics_em
from prg.learning.em_partial_noise import estimate_noise_em
from prg.utils.exceptions import FilterError, ParamError

SEED = 1234
N_STEPS = 120
ATOL = 1e-10


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _simulate(param, n):
    """Simulate via the model's own generator (control terms, augmented noise
    and singular covariances handled natively); returns (X_true, Y) lists of
    ``(dim, 1)`` columns for k = 0..n. Deterministic for a given SEED."""
    records = Linear_PKF(param, sKey=SEED).simulate_N_data(n)
    X = [x for _, x, _ in records]
    Y = [y for _, _, y in records]
    return X, Y


def _generator(X, Y, gaps):
    """Yield (k, x_true, y) with y set to all-NaN on gap steps (k >= 1)."""
    for k in range(len(X)):
        y = Y[k].copy()
        if k in gaps:
            y[:] = np.nan
        yield k, X[k].copy(), y


def _reference(param, hist0, Y, gaps):
    """Exact joint-Gaussian recursion, seeded from the filter's own k=0
    posterior. Returns lists of (X_est, PXX_est) for k = 1..n."""
    p = param.dim_x
    A = param.A
    Qeff = param.B @ param.mQ @ param.B.T
    dim_xy = A.shape[0]

    z = np.zeros((dim_xy, 1))
    P = np.zeros((dim_xy, dim_xy))
    z[:p] = hist0["Xkp1_update"]
    z[p:] = hist0["ykp1"]
    P[:p, :p] = hist0["PXXkp1_update"]

    out = []
    for k in range(1, len(Y)):
        z = A @ z
        P = A @ P @ A.T + Qeff
        if k in gaps:
            pass  # marginalisation: posterior = prior, FULL covariance kept
        else:
            S = P[p:, p:]
            G = P[:, p:] @ np.linalg.inv(S)      # joint gain [Pxy; Pyy] S^-1
            z = z + G @ (Y[k] - z[p:])
            P = P - G @ P[p:, :]
        out.append((z[:p].copy(), P[:p, :p].copy()))
    return out


def _run(param, gaps, n=N_STEPS):
    X, Y = _simulate(param, n)
    pkf = Linear_PKF(param, sKey=SEED)
    results = list(pkf.process_filter(data_generator=_generator(X, Y, gaps)))
    return pkf, results, X, Y


# Named gap patterns for the exactness test. Deliberately covers the
# configurations most likely to regress: the first possible gap (k=1, seeded
# straight from _firstEstimate), a long burst of consecutive gaps (the
# gap->gap carry and covariance inflation), and a trailing gap (the last
# record has Skp1 = None — exactly what runner diagnostics key on).
GAP_PATTERNS = {
    "no-gaps": set(),
    "every-3rd": set(range(2, N_STEPS, 3)),
    "every-2nd": set(range(2, N_STEPS, 2)),
    "first-possible": {1},
    "consecutive-burst": set(range(40, 50)),
    "trailing": {N_STEPS},
}


# ---------------------------------------------------------------------------
# correctness against the exact reference
# ---------------------------------------------------------------------------

class TestMissingObsCorrectness:

    @pytest.mark.parametrize(
        "fixture", ["param_x1y1", "param_x2y2", "param_x2y2_augmented"]
    )
    @pytest.mark.parametrize(
        "gaps", list(GAP_PATTERNS.values()), ids=list(GAP_PATTERNS.keys())
    )
    def test_matches_exact_reference(self, fixture, gaps, request):
        """With and without gaps, the filter must match the exact joint
        recursion seeded from its own first estimate."""
        param = request.getfixturevalue(fixture)
        pkf, results, X, Y = _run(param, gaps)

        ref = _reference(param, pkf.history[0], Y, gaps)
        assert len(results) == N_STEPS + 1
        for k in range(1, N_STEPS + 1):
            x_ref, pxx_ref = ref[k - 1]
            rec = pkf.history[k]
            np.testing.assert_allclose(
                rec["Xkp1_update"], x_ref, atol=ATOL,
                err_msg=f"state mismatch at k={k} (gap={k in gaps})",
            )
            np.testing.assert_allclose(
                rec["PXXkp1_update"], pxx_ref, atol=ATOL,
                err_msg=f"covariance mismatch at k={k} (gap={k in gaps})",
            )

    def test_gap_step_has_no_innovation_fields(self, param_x1y1):
        gaps = {3, 7}
        pkf, _, _, _ = _run(param_x1y1, gaps, n=10)
        for k in gaps:
            rec = pkf.history[k]
            assert rec["Skp1"] is None
            assert rec["Kkp1"] is None
            assert rec["ikp1"] is None
            assert np.isnan(rec["ykp1"]).all()
        # observed steps keep their innovation fields
        assert pkf.history[5]["Skp1"] is not None

    def test_gap_widens_covariance(self, param_x1y1):
        """At a gap step the posterior equals the prior exactly (no update),
        and it is wider than the posterior the same step would have had with
        its observation (conditioning on y strictly reduces uncertainty).

        Note: comparing against the PREVIOUS step's posterior would not be an
        invariant — a strongly contracting, low-noise model can shrink the
        trace across a gap even under exact marginalisation.
        """
        gaps = {5}
        pkf_gap, _, _, _ = _run(param_x1y1, gaps, n=10)
        pkf_full, _, _, _ = _run(param_x1y1, set(), n=10)

        rec = pkf_gap.history[5]
        # No update: posterior == prior, exactly.
        np.testing.assert_array_equal(rec["Xkp1_update"], rec["Xkp1_predict"])
        np.testing.assert_array_equal(
            rec["PXXkp1_update"], rec["PXXkp1_predict"]
        )
        # Same data up to k=4 (same seed), so the priors at k=5 coincide;
        # the observed run then conditions on y_5 and must end up tighter.
        tr_gap = float(np.trace(rec["PXXkp1_update"]))
        tr_obs = float(np.trace(pkf_full.history[5]["PXXkp1_update"]))
        assert tr_gap > tr_obs


# ---------------------------------------------------------------------------
# input validation
# ---------------------------------------------------------------------------

class TestMissingObsValidation:

    def test_partial_nan_raises(self, param_x2y2):
        X, Y = _simulate(param_x2y2, 5)
        Y[3][0, 0] = np.nan  # only one of two components missing

        pkf = Linear_PKF(param_x2y2, sKey=SEED)
        gen = ((k, X[k], Y[k]) for k in range(len(X)))
        with pytest.raises(ParamError, match="[Pp]artially missing"):
            list(pkf.process_filter(data_generator=gen))

    def test_first_obs_nan_raises(self, param_x1y1):
        X, Y = _simulate(param_x1y1, 5)
        Y[0][:] = np.nan

        pkf = Linear_PKF(param_x1y1, sKey=SEED)
        gen = ((k, X[k], Y[k]) for k in range(len(X)))
        with pytest.raises(ParamError, match="first observation"):
            list(pkf.process_filter(data_generator=gen))

    def test_partial_nan_first_obs_gets_partial_message(self, param_x2y2):
        """A partially NaN y_0 must get the partial-missing diagnosis, not the
        fully-missing one."""
        X, Y = _simulate(param_x2y2, 5)
        Y[0][0, 0] = np.nan  # only one of two components missing

        pkf = Linear_PKF(param_x2y2, sKey=SEED)
        gen = ((k, X[k], Y[k]) for k in range(len(X)))
        with pytest.raises(ParamError, match="[Pp]artially missing"):
            list(pkf.process_filter(data_generator=gen))

    def test_none_observation_raises(self, param_x1y1):
        X, Y = _simulate(param_x1y1, 5)

        pkf = Linear_PKF(param_x1y1, sKey=SEED)
        gen = ((k, X[k], None if k == 3 else Y[k]) for k in range(len(X)))
        with pytest.raises(ParamError, match="None"):
            list(pkf.process_filter(data_generator=gen))

    def test_wrong_size_gap_marker_raises(self, param_x2y2):
        X, Y = _simulate(param_x2y2, 5)
        bad_gap = np.full((3, 1), np.nan)  # dim_y is 2

        pkf = Linear_PKF(param_x2y2, sKey=SEED)
        gen = ((k, X[k], bad_gap if k == 3 else Y[k]) for k in range(len(X)))
        with pytest.raises(ParamError, match="size"):
            list(pkf.process_filter(data_generator=gen))

    def test_empty_generator_raises_filter_error(self, param_x1y1):
        pkf = Linear_PKF(param_x1y1, sKey=SEED)
        with pytest.raises(FilterError, match="no items"):
            pkf.process_N_data(N=None, data_generator=iter([]))


# ---------------------------------------------------------------------------
# downstream guards: smoothers and EM reject gapped data explicitly
# ---------------------------------------------------------------------------

class TestMissingObsDownstreamGuards:

    def test_smoother_rejects_gaps(self, param_x1y1):
        X, Y = _simulate(param_x1y1, 20)

        pks = Linear_PKS(param_x1y1, sKey=SEED)
        with pytest.raises(FilterError, match="missing observations"):
            pks.process_N_data_smoother(
                N=None, data_generator=_generator(X, Y, gaps={6})
            )

    def test_em_noise_rejects_gaps(self, param_x1y1):
        X, Y = _simulate(param_x1y1, 20)
        records = list(_generator(X, Y, gaps={6}))

        with pytest.raises(ParamError, match="missing observations"):
            estimate_noise_em(param_x1y1, records)

    def test_em_dynamics_rejects_gaps(self, param_x1y1):
        X, Y = _simulate(param_x1y1, 20)
        records = list(_generator(X, Y, gaps={6}))

        with pytest.raises(ParamError, match="missing observations"):
            estimate_dynamics_em(param_x1y1, records)


# ---------------------------------------------------------------------------
# nonlinear filters: NaN observations are rejected loudly
# ---------------------------------------------------------------------------

class TestNonlinearFiltersRejectNaN:
    """The nonlinear filters do not support gaps; a NaN observation must
    raise instead of silently propagating NaN means past the covariance
    checks (which never involve ``y``)."""

    @pytest.mark.parametrize(
        "fixture, make_filter",
        [
            ("param_nl_x2y1", lambda p: NonLinear_EPKF(p, sKey=SEED)),
            ("param_nl_x2y1",
             lambda p: NonLinear_UPKF(p, sigmaSet="wan2000", sKey=SEED)),
            ("param_nl_x2y1",
             lambda p: NonLinear_PPF(p, n_particles=100, sKey=SEED)),
            # NonLinear_PF only accepts classic (non-pairwise) models
            ("param_nl_classic_x1y1",
             lambda p: NonLinear_PF(p, n_particles=100, sKey=SEED)),
        ],
        ids=["EPKF", "UPKF", "PPF", "PF"],
    )
    @pytest.mark.parametrize("bad_k", [0, 3], ids=["first-obs", "mid-run"])
    def test_nan_observation_raises(self, fixture, make_filter, bad_k, request):
        param_nl = request.getfixturevalue(fixture)
        records = make_filter(param_nl).simulate_N_data(6)
        poisoned = [
            (
                k,
                x,
                np.full_like(np.asarray(y, dtype=float), np.nan)
                if k == bad_k
                else y,
            )
            for (k, x, y) in records
        ]
        filt = make_filter(param_nl)
        with pytest.raises(ParamError, match="NaN"):
            list(filt.process_filter(data_generator=iter(poisoned)))
