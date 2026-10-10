"""Partial EM for the linear-Gaussian pairwise model — couple dynamics blocks.

This estimator performs **partial** maximum-likelihood estimation by EM of the two
couple-defining transition blocks — the **back-action** ``A_xy`` (Y -> X) and the
**observation memory** ``A_yy`` (Y -> Y) — while the classically-recoverable blocks
``A_xx``, ``A_yx`` and the process-noise covariance ``Q`` are held **fixed**. It is
the dynamics counterpart of :mod:`prg.learning.em_partial_noise` (which fixes ``A``
and learns ``Q``), and reproduces Figs. 2-3 of the companion paper *"Smoothing,
Learning, and Testing the Gaussian Pairwise Markov Model"* (Sec. IV).

Why these two blocks
--------------------
The classical state-space model is the special case ``A_xy = 0`` (no back-action)
and ``R_xy = 0`` (uncorrelated noise). Recovering ``A_xy`` and ``A_yy`` from data,
starting from the *classical* initialisation ``A_xy = A_yy = 0``, is the sharpest
demonstration of the couple model's added content: EM discovers the couple
structure from a model that ignores it.

Identifiability (why the *Y*-column and not the *X*-column)
-----------------------------------------------------------
The latent state carries a gauge freedom: ``X_n -> T X_n`` (``T`` invertible), with
``(A, Q)`` transformed accordingly, leaves the law of ``y_{1:N}`` unchanged. The
observation memory ``A_yy`` couples **observed** coordinates and is gauge-invariant,
hence fully identifiable; the back-action ``A_xy`` is identifiable only up to the
gauge — pinned here by holding ``A_xx`` and ``Q`` fixed. Empirically ``A_yy`` is
recovered more tightly than ``A_xy`` (the latent-mediated block). Learning the
*X*-column (``A_xx``, ``A_yx``) instead would re-open the gauge; see the
identifiability discussion in :mod:`prg.learning.em_partial_noise`.

Model
-----
Pairwise couple ``Z = (X, Y)`` with ``X`` hidden, ``Y`` observed::

    Z_{n+1} = A Z_n + W_n,    W_n ~ N(0, Q),    Z = [X; Y]

with ``A = [[A_xx, A_xy], [A_yx, A_yy]]``. We take ``B = I`` so ``Q`` is the
effective process covariance (the ``*_AQ_pairwise`` models).

E-step
------
Run the **variational** linear smoother (``method="VAR"``) to get the smoothed
latent means ``x̂_n = E[X_n | y_{1:N}]`` (``Xkp1_smooth``). The regressor of the
learned column is the **observed** ``y_n``, so — unlike the noise M-step — no
smoothed covariance/cross-covariance is needed: only the means enter.

M-step (closed form, ``A_xx``, ``A_yx``, ``Q`` fixed)
-----------------------------------------------------
Regress the residual of each row of ``Z_{n+1}`` (after removing the fixed
X-driven part) on ``y_n``. With ``S_yy = Σ_n y_n y_n^T``::

    A_xy = [ Σ_n (x̂_{n+1} - A_xx x̂_n) y_n^T ] S_yy^{-1}
    A_yy = [ Σ_n (y_{n+1}   - A_yx x̂_n) y_n^T ] S_yy^{-1}

(The process-noise weighting ``Q^{-1}`` cancels because ``Q`` is constant in ``n``,
full rank, and both rows share the regressor ``y_n``, leaving an ordinary
least-squares normal equation.) Convergence is monitored by the observed-data
log-likelihood (forward innovation decomposition), which EM increases monotonically.

When only **one** block is learned (the other held fixed), ``Q^{-1}`` no longer
cancels if ``Q_xy != 0``: the fixed row's residual is informative about the learned
row's noise. The restricted M-step regresses the *conditional* residual instead.
With ``A_xy`` fixed (e.g. ``A_xy = 0``, the ``H0`` fit)::

    e_n  = x̂_{n+1} - A_xx x̂_n - A_xy y_n
    A_yy = [ Σ_n (y_{n+1} - A_yx x̂_n - Q_yx Q_xx^{-1} e_n) y_n^T ] S_yy^{-1}

and symmetrically, with ``A_yy`` fixed::

    f_n  = y_{n+1} - A_yx x̂_n - A_yy y_n
    A_xy = [ Σ_n (x̂_{n+1} - A_xx x̂_n - Q_xy Q_yy^{-1} f_n) y_n^T ] S_yy^{-1}

Testing for back-action
------------------------
:func:`back_action_lrt` runs two fits — ``A_xy`` free vs. ``A_xy = 0`` (``A_yy`` a
free nuisance in both) — and forms the likelihood-ratio statistic
``Λ = 2[ℓ(A_xy free) - ℓ(A_xy = 0)]``, asymptotically ``χ²`` with ``dim_x·dim_y``
degrees of freedom under ``H0: A_xy = 0``. EM is slow along weakly observable
directions of ``A_xy`` and its plateau rule stops short of the maximum there, which
biases ``Λ`` downward; the EM end points are therefore only warm starts, and both
log-likelihoods are maximised directly (BFGS on the exact ``y``-likelihood of the
pairwise Kalman filter) before ``Λ`` is formed.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize
from scipy.stats import chi2

from prg.classes.linear_pks import Linear_PKS
from prg.classes.param_linear import ParamLinear
from prg.learning.em_partial_noise import _gaussian_loglik
from prg.models.linear._amq import LinearAmQ
from prg.utils.exceptions import NumericalError, ParamError

__all__ = [
    "BackActionLRT",
    "EMDynamicsResult",
    "back_action_lrt",
    "estimate_dynamics_em",
]


@dataclass(frozen=True)
class EMDynamicsResult:
    """Result of :func:`estimate_dynamics_em`.

    Attributes
    ----------
    A : numpy.ndarray
        Full ``(dim_xy, dim_xy)`` transition with the learned Y-column filled in;
        the fixed ``A_xx``, ``A_yx`` blocks are unchanged from the input.
    A_xy : numpy.ndarray
        Estimated back-action block, shape ``(dim_x, dim_y)``.
    A_yy : numpy.ndarray
        Estimated observation-memory block, shape ``(dim_y, dim_y)``.
    loglik : list[float]
        Observed-data log-likelihood at each iteration (``loglik[t]`` uses the
        blocks of iteration ``t``); monotone non-decreasing if EM behaves.
    n_iter : int
        Number of EM iterations performed.
    converged : bool
        Whether the stopping criterion was met before ``max_iter``.
    """

    A: np.ndarray
    A_xy: np.ndarray
    A_yy: np.ndarray
    loglik: list[float]
    n_iter: int
    converged: bool


@dataclass(frozen=True)
class BackActionLRT:
    """Result of :func:`back_action_lrt` (test of ``H0: A_xy = 0``).

    Attributes
    ----------
    stat : float
        Likelihood-ratio statistic ``Λ = 2[ℓ(free) - ℓ(restricted)]`` (clipped at
        ``0`` for numerical safety).
    dof : int
        Degrees of freedom ``dim_x · dim_y`` (the number of free entries of
        ``A_xy``).
    pvalue : float
        Upper-tail ``χ²_dof`` probability ``P(χ²_dof > stat)``.
    loglik_free, loglik_restricted : float
        Maximised observed-data log-likelihoods of the two nested models.
    A_xy : numpy.ndarray
        Back-action estimated under the free (``H1``) model.
    """

    stat: float
    dof: int
    pvalue: float
    loglik_free: float
    loglik_restricted: float
    A_xy: np.ndarray


def _linear_param(dim_x, dim_y, A, Q, mz0, Pz0) -> ParamLinear:
    """Rebuild a consistent linear pairwise ``ParamLinear`` from ``A`` and ``Q``.

    The transition must be injected through :class:`LinearAmQ` (which regenerates
    the model's transition callables from ``A``), not by overwriting the ``A``
    attribute of an existing param: the forward filter uses the callables while the
    variational smoother reads the matrix, so a partial override desynchronises
    them and freezes the observed-data log-likelihood.
    """
    model = LinearAmQ(
        dim_x, dim_y,
        A=np.asarray(A, dtype=float),
        mQ=_effective_Q(Q),
        mz0=mz0, Pz0=Pz0, pairwiseModel=True,
    )
    kw = model.get_params().copy()
    kw.pop("dim_x")
    kw.pop("dim_y")
    return ParamLinear(0, dim_x, dim_y, **kw)


def _effective_Q(Q) -> np.ndarray:
    """Symmetrised, slightly regularised process covariance used by the E-step."""
    Q = np.asarray(Q, dtype=float)
    return 0.5 * (Q + Q.T) + 1e-9 * np.eye(Q.shape[0])


def _y_loglik(Y, A, Q, mz0, Pz0, dim_x) -> float:
    """Exact observed-data log-likelihood of ``y_{0:N}`` (pairwise Kalman filter).

    Same convention as :func:`prg.learning.em_partial_noise._gaussian_loglik` on the
    library filter's history: the prior term of ``y_0`` (which does not depend on
    ``A`` or ``Q``) is left out, so the two agree to rounding. ``Q`` is used as given
    (pass it through :func:`_effective_Q` to match the E-step). Returns ``-inf`` if
    an innovation covariance is not positive definite.

    Once ``y_n`` is conditioned on, only ``x_n | y_{0:n} ~ N(m_n, P_n)`` remains
    uncertain. The covariance recursion does not depend on the data; it is frozen
    once ``P_n`` reaches its fixed point (to rounding), which leaves only the mean
    recursion to run for the rest of the record.
    """
    Y = np.asarray(Y, dtype=float)
    A = np.asarray(A, dtype=float)
    Q = np.asarray(Q, dtype=float)
    dx = dim_x
    A_x = A[:, :dx]
    AyY = Y[:-1] @ A[:, dx:].T          # y-driven part of the predicted mean
    m = np.asarray(mz0, dtype=float).ravel()
    P = np.asarray(Pz0, dtype=float)
    quad = logdet = 0.0
    frozen = False
    with np.errstate(all="ignore"):
        try:
            # condition the prior on y_0
            G = P[:dx, dx:] @ np.linalg.inv(P[dx:, dx:])
            m_x = m[:dx] + G @ (Y[0] - m[dx:])
            P_x = P[:dx, :dx] - G @ P[dx:, :dx]
            for n in range(1, len(Y)):
                if not frozen:
                    Pp = A_x @ P_x @ A_x.T + Q
                    c = np.linalg.cholesky(Pp[dx:, dx:])
                    S_inv = np.linalg.inv(c)
                    S_inv = S_inv.T @ S_inv
                    G = Pp[:dx, dx:] @ S_inv
                    P_new = Pp[:dx, :dx] - G @ Pp[dx:, :dx]
                    step_logdet = 2.0 * float(np.log(np.diag(c)).sum())
                    frozen = bool(np.max(np.abs(P_new - P_x), initial=0.0)
                                  <= 1e-15 * np.max(np.abs(P_new), initial=0.0))
                    P_x = P_new
                mp = A_x @ m_x + AyY[n - 1]
                r = Y[n] - mp[dx:]
                quad += r @ S_inv @ r
                logdet += step_logdet
                m_x = mp[:dx] + G @ r
        except np.linalg.LinAlgError:
            return -np.inf
    ll = -0.5 * ((len(Y) - 1) * Y.shape[1] * np.log(2.0 * np.pi) + logdet + quad)
    return float(ll) if np.isfinite(ll) else -np.inf


def _maximize_blocks(Y, A, Q, mz0, Pz0, dim_x, learn_back_action, learn_obs_memory, starts):
    """Directly maximise :func:`_y_loglik` over the learned blocks (BFGS).

    At least one block must be learned. ``A`` supplies the fixed blocks; each matrix
    in ``starts`` is a full transition whose learned blocks seed one BFGS run.
    Returns ``(A_best, loglik_best)``, the best over the starts themselves and the
    BFGS end points.
    """
    rows = slice(None) if (learn_back_action and learn_obs_memory) else (
        slice(None, dim_x) if learn_back_action else slice(dim_x, None))
    A = np.asarray(A, dtype=float)
    shape = A[rows, dim_x:].shape

    def build(theta):
        A_t = A.copy()
        A_t[rows, dim_x:] = theta.reshape(shape)
        return A_t

    def neg_ll(theta):
        ll = _y_loglik(Y, build(theta), Q, mz0, Pz0, dim_x)
        return -ll if np.isfinite(ll) else 1e300

    best_A, best_ll = None, -np.inf
    for A0 in starts:
        theta0 = np.asarray(A0, dtype=float)[rows, dim_x:].ravel()
        with np.errstate(all="ignore"):
            res = minimize(neg_ll, theta0, method="BFGS", options={"gtol": 1e-7})
        for theta in (theta0, res.x):
            ll = -neg_ll(theta)
            if ll > best_ll:
                best_A, best_ll = build(theta), ll
    return best_A, float(best_ll)


def estimate_dynamics_em(
    param: ParamLinear,
    data,
    *,
    learn_back_action: bool = True,
    learn_obs_memory: bool = True,
    max_iter: int = 100,
    tol: float = 1e-6,
    sKey: int | None = None,
    verbose: int = 0,
) -> EMDynamicsResult:
    """Estimate the couple dynamics blocks ``A_xy`` and/or ``A_yy`` by EM.

    The blocks ``A_xx``, ``A_yx`` and the noise ``Q`` are held **fixed** at the
    values carried by ``param``; the learned blocks are *initialised* from
    ``param.A`` (use ``A_xy = A_yy = 0`` for the classical initialisation). Any
    block whose ``learn_*`` flag is ``False`` is kept at its initial value — in
    particular ``learn_back_action=False`` with ``param``'s ``A_xy = 0`` fits the
    classical (no-back-action) sub-model, the ``H0`` case of
    :func:`back_action_lrt`.

    Parameters
    ----------
    param : ParamLinear
        Linear-Gaussian pairwise parameters; supplies the fixed ``A_xx``,
        ``A_yx``, ``mQ`` (with ``B = I``) and the initial learned blocks.
    data : sequence of tuple
        Re-iterable ``(k, x_true_or_None, y_obs)`` triples, as produced by
        :meth:`Linear_PKF.simulate_N_data`. Only ``y_obs`` is used.
    learn_back_action, learn_obs_memory : bool
        Whether to update ``A_xy`` / ``A_yy`` (default both ``True``).
    max_iter : int
        Maximum number of EM iterations.
    tol : float
        Stop when the relative increase of the observed-data log-likelihood (the
        quantity EM maximises) falls below ``tol`` — a plateau criterion that
        avoids chasing the weakly-identified back-action along a near-flat
        likelihood ridge.
    sKey : int, optional
        Seed forwarded to the internal smoother (does not affect the linear
        E-step; present for parity).
    verbose : int
        ``0`` silent, ``1`` per-iteration log-likelihood, ``2`` also the blocks.

    Returns
    -------
    EMDynamicsResult

    Raises
    ------
    ParamError
        If ``data`` has fewer than two steps.
    NumericalError
        If the observed second-moment ``S_yy = Σ_n y_n y_n^T`` is singular.
    """
    records = list(data)
    if len(records) < 2:
        raise ParamError(f"EM needs at least 2 time steps (got {len(records)}).")
    if any(np.isnan(np.asarray(rec[2], dtype=float)).any() for rec in records):
        raise ParamError(
            "estimate_dynamics_em: missing observations (NaN in y) are not "
            "supported."
        )

    dim_x, dim_y = int(param.dim_x), int(param.dim_y)
    A = np.asarray(param.A, dtype=float).copy()
    A_xx = A[:dim_x, :dim_x].copy()   # fixed
    A_yx = A[dim_x:, :dim_x].copy()   # fixed
    Q = np.asarray(param.mQ, dtype=float)
    Qe = _effective_Q(Q)              # the covariance the E-step uses
    Q_xx, Q_xy = Qe[:dim_x, :dim_x], Qe[:dim_x, dim_x:]
    Q_yx, Q_yy = Qe[dim_x:, :dim_x], Qe[dim_x:, dim_x:]
    mz0 = np.asarray(param.mz0, dtype=float)
    Pz0 = np.asarray(param.Pz0, dtype=float)

    loglik: list[float] = []
    converged = False
    it = 0
    for it in range(1, max_iter + 1):
        # ---- E-step: VAR smoother with the current A ----
        em_param = _linear_param(dim_x, dim_y, A, Q, mz0, Pz0)
        smoother = Linear_PKS(em_param, sKey=sKey, method="VAR")
        smoother.process_N_data_smoother(N=None, data_generator=iter(records))
        hist = smoother.history._history

        loglik.append(_gaussian_loglik(smoother.history))

        xh = np.array([np.asarray(rec["Xkp1_smooth"]).reshape(dim_x) for rec in hist])
        y = np.array([np.asarray(rec["ykp1"]).reshape(dim_y) for rec in hist])
        X0, X1 = xh[:-1], xh[1:]           # (T, dim_x)
        Y0, Y1 = y[:-1], y[1:]             # (T, dim_y)
        S_yy = Y0.T @ Y0                    # (dim_y, dim_y)

        # ---- M-step: regression of each learned residual row on the observed y_n ----
        # Residuals with the learned column removed; when only one block is learned,
        # the fixed row's full residual is regressed out through Q (GLS, see module doc).
        R_x = X1 - X0 @ A_xx.T                              # (T, dim_x)
        R_y = Y1 - X0 @ A_yx.T                              # (T, dim_y)
        if learn_back_action and not learn_obs_memory:
            E_y = R_y - Y0 @ A[dim_x:, dim_x:].T
            R_x = R_x - E_y @ np.linalg.solve(Q_yy, Q_yx)   # - Q_xy Q_yy^{-1} f_n
        elif learn_obs_memory and not learn_back_action:
            E_x = R_x - Y0 @ A[:dim_x, dim_x:].T
            R_y = R_y - E_x @ np.linalg.solve(Q_xx, Q_xy)   # - Q_yx Q_xx^{-1} e_n
        A_new = A.copy()
        try:
            if learn_back_action:
                A_new[:dim_x, dim_x:] = np.linalg.solve(S_yy.T, (R_x.T @ Y0).T).T
            if learn_obs_memory:
                A_new[dim_x:, dim_x:] = np.linalg.solve(S_yy.T, (R_y.T @ Y0).T).T
        except np.linalg.LinAlgError as exc:
            raise NumericalError(
                "Observed second moment S_yy = sum_n y_n y_n^T is singular; "
                "cannot solve the M-step normal equations.",
                matrix_name="S_yy",
            ) from exc

        learned_old = np.concatenate([A[:, dim_x:].ravel()])
        learned_new = np.concatenate([A_new[:, dim_x:].ravel()])
        dA = float(np.linalg.norm(learned_new - learned_old)) / (
            float(np.linalg.norm(learned_old)) + 1e-300
        )
        dll = (
            abs(loglik[-1] - loglik[-2]) / (abs(loglik[-2]) + 1e-300)
            if len(loglik) >= 2 else np.inf
        )
        if verbose >= 1:
            msg = f"[EM dyn] iter {it:3d}  loglik={loglik[-1]:.6f}  |dA|={dA:.3e}"
            if verbose >= 2:
                msg += (f"  A_xy={A_new[:dim_x, dim_x:].ravel()}"
                        f"  A_yy={A_new[dim_x:, dim_x:].ravel()}")
            print(msg)

        A = A_new
        if dll < tol:  # observed-data log-likelihood has plateaued
            converged = True
            break

    return EMDynamicsResult(
        A=A,
        A_xy=A[:dim_x, dim_x:].copy(),
        A_yy=A[dim_x:, dim_x:].copy(),
        loglik=loglik,
        n_iter=it,
        converged=converged,
    )


def back_action_lrt(
    param: ParamLinear,
    data,
    *,
    max_iter: int = 100,
    tol: float = 1e-6,
    sKey: int | None = None,
    polish: bool = True,
) -> BackActionLRT:
    """Likelihood-ratio test of ``H0: A_xy = 0`` (no measurement back-action).

    Fits two nested partial models on the same record — ``A_xy`` free vs.
    ``A_xy = 0``, with ``A_yy`` a free nuisance in both — and forms
    ``Λ = 2[ℓ(free) - ℓ(restricted)]``. ``param`` should carry the classical
    initialisation ``A_xy = 0`` (its ``A_xy`` block seeds the free fit and *is*
    the fixed value of the restricted fit). Under ``H0``, ``Λ`` is asymptotically
    ``χ²`` with ``dim_x · dim_y`` degrees of freedom.

    Each fit is first run by partial EM (:func:`estimate_dynamics_em` with
    ``max_iter`` / ``tol``); with ``polish=True`` (default) the EM end points only
    seed a direct maximisation (BFGS) of the exact ``y``-log-likelihood over the
    learned blocks — the free fit started both from its EM end point and from the
    restricted optimum with ``A_xy = 0``, so ``ℓ(free) >= ℓ(restricted)``. EM's
    plateau rule stops short of the maximum along weakly observable directions of
    ``A_xy`` and, unpolished, biases ``Λ`` downward (a conservative test);
    ``polish=False`` returns that EM-only statistic.

    Returns
    -------
    BackActionLRT
    """
    dim_x, dim_y = int(param.dim_x), int(param.dim_y)
    records = list(data)
    free = estimate_dynamics_em(
        param, records, learn_back_action=True, learn_obs_memory=True,
        max_iter=max_iter, tol=tol, sKey=sKey,
    )
    restricted = estimate_dynamics_em(
        param, records, learn_back_action=False, learn_obs_memory=True,
        max_iter=max_iter, tol=tol, sKey=sKey,
    )
    if polish:
        Y = np.array([np.asarray(rec[2], dtype=float).reshape(dim_y) for rec in records])
        Qe = _effective_Q(param.mQ)
        args = (Y, np.asarray(param.A, dtype=float), Qe, param.mz0, param.Pz0, dim_x)
        A_r, ll_r = _maximize_blocks(*args, False, True, [restricted.A])
        A_f, ll_f = _maximize_blocks(*args, True, True, [free.A, A_r])
        A_xy = A_f[:dim_x, dim_x:].copy()
    else:
        ll_f, ll_r = free.loglik[-1], restricted.loglik[-1]
        A_xy = free.A_xy
    stat = max(2.0 * (ll_f - ll_r), 0.0)
    dof = dim_x * dim_y
    return BackActionLRT(
        stat=float(stat),
        dof=int(dof),
        pvalue=float(chi2.sf(stat, dof)),
        loglik_free=float(ll_f),
        loglik_restricted=float(ll_r),
        A_xy=A_xy,
    )
