"""Partial EM for the linear-Gaussian pairwise model — noise covariance only.

This estimator performs **partial** maximum-likelihood estimation by EM: the
couple transition matrix ``A`` is assumed **known** (fixed), and only the joint
process-noise covariance ``Q`` is estimated.

Why "partial" / why fix ``A``
-----------------------------
Joint EM on ``(A, Q)`` of a linear-Gaussian state-space model is **not
identifiable**: any invertible change of the hidden-X coordinates ``X' = T X``
leaves the law of the observations unchanged (a gauge freedom), giving a
continuum of equivalent ``(A, Q)``. Fixing ``A`` removes that gauge, and the
M-step below is then a unique closed form (a property of the constrained
optimisation given the E-step moments).

Removing the gauge is **not** enough to identify the full joint ``Q``
---------------------------------------------------------------------
Even with ``A`` fixed and the couple observable (X reconstructible from the
Y-sequence — ``A_yx != 0`` in the scalar case, a *necessary* condition: with
``A_yx = 0`` the likelihood is exactly flat in ``Q_xx``), the cross-noise
``Q_xy`` is **effectively non-identified from the hidden state**. The joint-``Q``
Fisher information is numerically singular along a ``Q_xy``-dominated direction:
``Q_xy`` trades off against the diagonal blocks along a near-flat likelihood
ridge (bounded only by the PSD constraint on ``Q``). Verified numerically:

- *Conditional* on the diagonal blocks ``Q_xx``, ``Q_yy`` (or with X observed),
  ``Q_xy`` is **sharply** identified — the complete-data estimate is precise even
  at small ``N``.
- *Jointly* (blocks free, X hidden) the per-observation information along the
  ridge does not grow with ``N``; the EM endpoint for ``Q_xy`` tracks the
  **initialisation**, not the data, and does not approach the truth as ``N``
  grows.

The X- and Y-noise *blocks* remain well identified. Pass ``block_diagonal=True``
to estimate the well-conditioned block-diagonal sub-model (``Q_xy = 0``), which
recovers cleanly at modest ``N``.

Model
-----
Pairwise couple ``Z = (X, Y)`` with ``X`` hidden, ``Y`` observed::

    Z_{n+1} = A Z_n + W_n,    W_n ~ N(0, Q),    Z = [X; Y]

``A`` is ``(dim_xy, dim_xy)`` (blocks ``[[A_xx, A_xy], [A_yx, A_yy]]``); the
X-block occupies indices ``[0:dim_x]``. We take the noise-injection factor
``B = I`` so ``Q`` is the effective process covariance (the case of the
``*_AQ_pairwise`` models).

E-step
------
Run the **variational** linear smoother (``method="VAR"``, the only variant that
exposes the lag-one cross-covariance ``Mk_smooth``) to get, per step ``n``:

- ``x̂_n = E[X_n | y_{1:N}]``                (``Xkp1_smooth``)
- ``P_n = Cov(X_n | y_{1:N})``              (``PXXkp1_smooth``)
- ``M_n = Cov(X_{n+1}, X_n | y_{1:N})``     (``Mk_smooth``)

Because ``Y`` is **observed**, the joint posterior moments collapse to the data
on every Y-row/column: only the X-block carries uncertainty.

M-step (unique closed form, A fixed)
------------------------------------
``Q̂ = (1/T) Σ_n E[(Z_{n+1} - A Z_n)(Z_{n+1} - A Z_n)^T | y_{1:N}]`` which is PSD
by construction. With ``ẑ_n = [x̂_n; y_n]`` and the embedding ``E_x = [I_x; 0]``::

    Σ_{n,n}   = ẑ_n ẑ_n^T   + E_x P_n     E_x^T
    Σ_{n,n+1} = ẑ_n ẑ_{n+1}^T + E_x M_n^T E_x^T
    Q̂ = (1/T) Σ_n [ Σ_{n+1,n+1} - A Σ_{n,n+1} - Σ_{n,n+1}^T A^T + A Σ_{n,n} A^T ]

Convergence is monitored by the observed-data log-likelihood (forward
innovation decomposition), which EM increases monotonically.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.linalg import LinAlgError, cho_factor, cho_solve

from prg.classes.linear_pks import Linear_PKS
from prg.classes.param_linear import ParamLinear
from prg.utils.exceptions import NumericalError, ParamError

__all__ = ["EMNoiseResult", "estimate_noise_em"]


@dataclass(frozen=True)
class EMNoiseResult:
    """Result of :func:`estimate_noise_em`.

    Attributes
    ----------
    Q : numpy.ndarray
        Estimated joint process-noise covariance, shape ``(dim_xy, dim_xy)``,
        symmetric PSD.
    loglik : list[float]
        Observed-data log-likelihood at each iteration (``loglik[t]`` uses the
        Q of iteration ``t``); monotone non-decreasing if EM behaves.
    n_iter : int
        Number of EM iterations performed.
    converged : bool
        Whether the stopping criterion (log-likelihood / ``Q`` change below
        ``tol``) was met before ``max_iter``.
    """

    Q: np.ndarray
    loglik: list[float]
    n_iter: int
    converged: bool


def _embed_xx(P_xx: np.ndarray, dim_x: int, dim_xy: int) -> np.ndarray:
    """Embed a ``(dim_x, dim_x)`` block into the X-X corner of a
    ``(dim_xy, dim_xy)`` zero matrix (the only block carrying X-uncertainty,
    since Y is observed)."""
    out = np.zeros((dim_xy, dim_xy))
    out[:dim_x, :dim_x] = P_xx
    return out


def _gaussian_loglik(history) -> float:
    """Observed-data log-likelihood from the forward innovation decomposition:
    ``Σ_n log N(i_n; 0, S_n)`` using ``ikp1`` / ``Skp1`` in the filter history.

    The first record (``k=0``) is the prior, not a one-step prediction: it
    carries placeholder ``ikp1=0`` / ``Skp1=I`` (a Q-independent constant), so it
    is skipped — the sum runs over the genuine innovations ``k=1..N``."""
    ll = 0.0
    log2pi = float(np.log(2.0 * np.pi))
    for rec in history._history[1:]:
        i = rec.get("ikp1")
        S = rec.get("Skp1")
        if i is None or S is None:
            continue
        i = np.asarray(i).reshape(-1, 1)
        S = np.asarray(S)
        try:
            cf = cho_factor(S, lower=True, check_finite=False)
        except LinAlgError as exc:
            raise NumericalError(
                "Innovation covariance S is not positive-definite while "
                "evaluating the log-likelihood.",
                matrix_name="Skp1",
            ) from exc
        logdet = 2.0 * float(np.sum(np.log(np.diag(cf[0]))))
        quad = float((i.T @ cho_solve(cf, i, check_finite=False)).item())
        ll += -0.5 * (S.shape[0] * log2pi + logdet + quad)
    return ll


def estimate_noise_em(
    param: ParamLinear,
    data,
    *,
    Q_init: np.ndarray | None = None,
    block_diagonal: bool = False,
    max_iter: int = 200,
    tol: float = 1e-7,
    sKey: int | None = None,
    verbose: int = 0,
) -> EMNoiseResult:
    """Estimate only the joint noise covariance ``Q`` by EM, with ``A`` fixed.

    Parameters
    ----------
    param : ParamLinear
        Linear-Gaussian pairwise parameters supplying the **known** transition
        ``A`` (and ``mz0`` / ``Pz0``, kept fixed). The noise-injection factor is
        taken as ``B = I``, so ``Q`` is the effective process covariance.
    data : sequence of tuple
        Re-iterable sequence of ``(k, x_true_or_None, y_obs)`` triples — the
        format produced by :meth:`Linear_PKF.simulate_N_data` and consumed by
        the smoother's ``data_generator``. Only ``y_obs`` (the observed Y) is
        used; ``x_true`` is ignored (X is hidden).
    Q_init : numpy.ndarray, optional
        Initial ``(dim_xy, dim_xy)`` covariance. Defaults to ``param.mQ``.
    block_diagonal : bool, optional
        If ``True``, constrain ``Q`` to be block-diagonal (``Q_xy = 0``): only
        the X-noise and Y-noise blocks are estimated. This is the
        **well-identified** sub-problem. In the full joint ``Q`` the cross-noise
        ``Q_xy`` is effectively non-identified from the hidden state (a near-flat
        likelihood ridge; the EM endpoint tracks the initial ``Q`` rather than
        converging to the truth), whereas the block-diagonal blocks recover
        cleanly at modest ``N``. Defaults to ``False``.
    max_iter : int
        Maximum number of EM iterations.
    tol : float
        Stop when both the relative log-likelihood increase and the relative
        ``Q`` change fall below ``tol``.
    sKey : int, optional
        Seed forwarded to the internal smoother (does not affect the linear
        E-step result; present for reproducibility/parity).
    verbose : int
        ``0`` silent, ``1`` per-iteration log-likelihood, ``2`` also ``Q``.

    Returns
    -------
    EMNoiseResult

    Raises
    ------
    ParamError
        If ``data`` is empty / malformed or ``Q_init`` has the wrong shape.
    NumericalError
        On a non-PSD intermediate covariance during the E-step or
        log-likelihood evaluation.

    Notes
    -----
    Observability (``A`` fixed + X reconstructible from Y, e.g. ``A_yx != 0`` in
    the scalar case) is *necessary*: with ``A_yx = 0`` the likelihood is flat in
    ``Q_xx``. It is not *sufficient* for the full joint ``Q`` — even when
    observable, the cross-noise ``Q_xy`` direction is only weakly identified (a
    near-flat ridge). Use ``block_diagonal=True`` for the well-conditioned
    sub-model. See the module docstring for the full identifiability discussion.
    """
    records = list(data)
    if len(records) < 2:
        raise ParamError("EM needs at least 2 time steps (got "
                         f"{len(records)}).")

    dim_x, dim_y = int(param.dim_x), int(param.dim_y)
    dim_xy = dim_x + dim_y
    A = np.asarray(param.A, dtype=float)

    if Q_init is None:
        Q = np.asarray(param.mQ, dtype=float).copy()
    else:
        Q = np.asarray(Q_init, dtype=float).copy()
    if Q.shape != (dim_xy, dim_xy):
        raise ParamError(
            f"Q_init must be {(dim_xy, dim_xy)}, got {Q.shape}."
        )
    Q = 0.5 * (Q + Q.T)
    if float(np.min(np.linalg.eigvalsh(Q))) <= 0.0:
        raise ParamError("Q_init must be positive-definite.")
    if block_diagonal:
        Q[:dim_x, dim_x:] = 0.0
        Q[dim_x:, :dim_x] = 0.0

    # Boilerplate kwargs to rebuild a ParamLinear each iteration (A fixed, B=I).
    base_kwargs = {
        "A": A,
        "B": np.eye(dim_xy),
        "mz0": np.asarray(param.mz0, dtype=float),
        "Pz0": np.asarray(param.Pz0, dtype=float),
        "augmented": bool(param.augmented),
        "pairwiseModel": bool(param.pairwiseModel),
        "g": param.g, "f": param.f, "h": param.h, "jacobiens_g": param.jacobiens_g,
        "alpha": param.alpha, "beta": param.beta, "kappa": param.kappa,
        "lambda_": param.lambda_,
    }

    loglik: list[float] = []
    converged = False
    it = 0
    # Build the working ParamLinear once (A fixed, B=I); only mQ changes per
    # iteration. A fresh Linear_PKS is built each iter so it re-reads mQ
    # (Linear_PKF caches B·mQ·Bᵀ at construction).
    em_param = ParamLinear(0, dim_x, dim_y, **{**base_kwargs, "mQ": Q})

    for it in range(1, max_iter + 1):
        # ---- build a smoother with the current Q, run E-step ----
        em_param.mQ = Q
        smoother = Linear_PKS(em_param, sKey=sKey, method="VAR")
        smoother.process_N_data_smoother(N=None, data_generator=iter(records))
        hist = smoother.history._history
        R = len(hist)

        ll = _gaussian_loglik(smoother.history)
        loglik.append(ll)

        # ---- assemble joint smoothed moments and the M-step numerator ----
        # ẑ_n, Cov_n (XX = P_n), Cross_n (XX = M_n^T) for n = 0..R-1.
        zhat = np.empty((R, dim_xy, 1))
        cov = [None] * R          # Cov(Z_n | Y) = embed_xx(P_n)
        cross = [None] * (R - 1)  # Cov(Z_n, Z_{n+1} | Y) = embed_xx(M_n^T)
        for n in range(R):
            rec = hist[n]
            xh = np.asarray(rec["Xkp1_smooth"]).reshape(dim_x, 1)
            y = np.asarray(rec["ykp1"]).reshape(dim_y, 1)
            zhat[n] = np.vstack([xh, y])
            cov[n] = _embed_xx(np.asarray(rec["PXXkp1_smooth"]), dim_x, dim_xy)
            if n < R - 1:
                M = np.asarray(rec["Mk_smooth"])              # Cov(X_{n+1}, X_n)
                cross[n] = _embed_xx(M.T, dim_x, dim_xy)      # Cov(X_n, X_{n+1})

        T = R - 1  # number of transitions
        Q_new = np.zeros((dim_xy, dim_xy))
        for n in range(T):
            S_nn = zhat[n] @ zhat[n].T + cov[n]
            S_npn = zhat[n] @ zhat[n + 1].T + cross[n]          # Σ_{n,n+1}
            S_pp = zhat[n + 1] @ zhat[n + 1].T + cov[n + 1]
            A_Snpn = A @ S_npn
            Q_new += S_pp - A_Snpn - A_Snpn.T + A @ S_nn @ A.T
        Q_new = Q_new / T
        Q_new = 0.5 * (Q_new + Q_new.T)  # symmetrise for numerical PSD
        if block_diagonal:
            # Constrained M-step: ML for a block-diagonal Q is just the diagonal
            # blocks of the unconstrained residual covariance (cross-block = 0).
            Q_new[:dim_x, dim_x:] = 0.0
            Q_new[dim_x:, :dim_x] = 0.0

        # ---- convergence: relative loglik increase and relative Q change ----
        dQ = float(np.linalg.norm(Q_new - Q)) / (float(np.linalg.norm(Q)) + 1e-300)
        dll = (
            abs(loglik[-1] - loglik[-2]) / (abs(loglik[-2]) + 1e-300)
            if len(loglik) >= 2 else np.inf
        )
        if verbose >= 1:
            msg = f"[EM noise] iter {it:3d}  loglik={ll:.6f}  |dQ|={dQ:.3e}"
            if verbose >= 2:
                msg += f"\n{np.array2string(Q_new, precision=4)}"
            print(msg)

        Q = Q_new
        if dQ < tol and dll < tol:
            converged = True
            break

    return EMNoiseResult(Q=Q, loglik=loglik, n_iter=it, converged=converged)
