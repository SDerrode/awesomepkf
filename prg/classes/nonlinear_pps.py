"""
####################################################################
Pairwise Particle Smoother (PPS) — FFBSm on top of the PPF.
####################################################################

Implements Forward Filtering, Backward Smoothing (FFBSm) (Doucet et al. 2000,
Klaas et al. 2006) for the pairwise particle filter. Unlike the Kalman-family
smoothers (PKS / EPKS / UPKS / UKS), there is no closed-form covariance
recursion: the smoother reweights the **forward** particle cloud using a
backward-in-time weight recursion based on the transition density evaluated
between every pair of forward particles.

Complexity is :math:`O(N \\cdot n_p^2)` per smoother run (vs
:math:`O(N \\cdot n_p)` for the forward filter). For the test grid
``n_p = 300, N = 300``, the backward pass evaluates ``~3 \\cdot 10^7``
Gaussian densities — well within tractable limits.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator

import numpy as np
from scipy.linalg import LinAlgError, cho_factor, cho_solve
from scipy.special import logsumexp

from prg.classes.nonlinear_ppf import NonLinear_PPF
from prg.utils.exceptions import CovarianceError, FilterError

logger = logging.getLogger(__name__)

__all__ = ["NonLinear_PPS"]


class NonLinear_PPS(NonLinear_PPF):
    """
    Pairwise Particle Smoother (PPS) using FFBSm.

    Two-pass smoother extending :class:`NonLinear_PPF`. The forward pass
    is the standard PPF (with ``store_particles=True`` forced so each
    history record carries the per-step particle cloud and weights). The
    backward pass implements the Forward Filtering, Backward Smoothing
    (FFBSm) recursion:

    .. math::

        \\tilde w_{i,n} \\;=\\; w_{i,n} \\sum_j \\tilde w_{j,n+1}\\,
            \\frac{p(\\xi_{j,n+1} \\mid \\xi_{i,n},\\, y_n)}
                  {\\sum_l w_{l,n}\\, p(\\xi_{j,n+1} \\mid \\xi_{l,n},\\, y_n)}

    Smoothed moments per step are recovered as standard weighted
    statistics on the forward particle cloud:

    .. math::

        \\hat X_{n|N} \\;=\\; \\sum_i \\tilde w_{i,n}\\, \\xi_{i,n},
        \\quad
        P^{xx}_{n|N} \\;=\\; \\sum_i \\tilde w_{i,n}\\,
            (\\xi_{i,n} - \\hat X_{n|N})(\\xi_{i,n} - \\hat X_{n|N})^\\top.

    Note: there is no Joseph form for particle smoothers — that is a
    Kalman-family numerical safeguard. The corresponding numerical
    safeguard for particle smoothers is the log-sum-exp normalisation
    of the backward kernel, applied below.

    Parameters
    ----------
    Forwarded to :class:`NonLinear_PPF`. ``store_particles`` is forced
    to ``True``; user-supplied value (if any) is overridden.

    Complexity
    ----------
    Forward: :math:`O(N \\cdot n_p)`. Backward: :math:`O(N \\cdot n_p^2)`.
    Total memory: history stores ``N`` particle clouds of shape
    ``(n_p, dim_x, 1)`` and weight vectors of shape ``(n_p,)``; this is
    additional to the regular PKFStep fields.

    History schema additions
    ------------------------
    Each forward record (from the PPF) carries ``particles`` and
    ``weights``. The backward pass adds three more keys:

    - ``Xkp1_smooth`` of shape ``(dim_x, 1)``: smoothed posterior mean.
    - ``PXXkp1_smooth`` of shape ``(dim_x, dim_x)``: smoothed covariance.
    - ``w_smooth`` of shape ``(n_p,)``: smoothed weights
      :math:`\\tilde w_{i,n}` (sum to 1).

    Unlike the Kalman smoothers, there is **no** ``Gk_smooth`` (no gain
    matrix in the particle backward recursion). All four fields are
    written via the public :meth:`HistoryTracker.update_record` API.
    """

    def __init__(self, *args, **kwargs) -> None:
        # Force particle storage on — the backward pass needs the cloud.
        kwargs["store_particles"] = True
        super().__init__(*args, **kwargs)

    # ------------------------------------------------------------------
    # Smoother
    # ------------------------------------------------------------------

    def process_smoother(
        self,
        N: int | None = None,
        data_generator: Iterator[tuple[int, np.ndarray, np.ndarray]] | None = None,
    ) -> Iterator[
        tuple[
            int,
            np.ndarray | None,
            np.ndarray,
            np.ndarray,
            np.ndarray,
            np.ndarray,
        ]
    ]:
        """
        Run the PPS as a generator.

        First exhausts the PPF forward (each history record now carries
        ``particles`` and ``weights``), then performs the FFBSm
        backward recursion in place and yields the augmented tuples in
        chronological order.

        Yields
        ------
        k, x_true, y_observed, X_predict, X_update, X_smooth
            ``X_smooth`` is the smoothed posterior mean
            :math:`\\sum_i \\tilde w_{i,n} \\xi_{i,n}`.

        Raises
        ------
        ParamError, InvertibilityError, NumericalError, StepValidationError, FilterError
            Propagated unwrapped from the PPF forward pass.
        CovarianceError
            If the transition-noise covariance block ``mQ[:p, :p]`` is
            not positive definite (its Cholesky is required to evaluate
            the backward kernel).
        """
        # 1) Forward pass — drains PPF into self.history with particles/weights
        for _ in self.process_filter(N=N, data_generator=data_generator):
            pass

        N_records: int = len(self.history)
        if N_records == 0:
            raise FilterError("NonLinear_PPS: forward pass yielded no records.")

        logger.info(
            "NonLinear_PPS backward pass starting (N_records=%d, n_particles=%d).",
            N_records,
            self.n_particles,
        )

        # 2) Pre-compute the inverse of the X-marginal transition-noise
        # covariance. Same convention as the PPF: assumes B = I, so the
        # marginal X-noise covariance is just the top-left p×p block of mQ.
        Sigma_xx: np.ndarray = self.param.mQ[: self.dim_x, : self.dim_x]
        try:
            c_sig, low_sig = cho_factor(Sigma_xx)
            Sigma_xx_inv: np.ndarray = cho_solve(
                (c_sig, low_sig), self.eye_dim_x
            )
        except (LinAlgError, ValueError) as e:
            raise CovarianceError(
                "NonLinear_PPS: Cholesky factorisation of mQ[:p,:p] failed — "
                "cannot evaluate the backward transition kernel.",
                matrix_name="mQ[:p,:p]",
                step=-1,
            ) from e

        # 3) Terminal step: smoothed = filtered
        last = self.history[N_records - 1]
        self._record_smoothed(
            N_records - 1, last["particles"], last["weights"].copy(),
        )

        # 4) Backward FFBSm recursion
        zeros_batched: np.ndarray = np.zeros(
            (self.n_particles, self.dim_xy, 1)
        )

        for idx in range(N_records - 2, -1, -1):
            cur = self.history[idx]
            nxt = self.history[idx + 1]

            particles_n: np.ndarray = cur["particles"]         # (n_p, dim_x, 1)
            w_n: np.ndarray = cur["weights"]                   # (n_p,)
            Yn: np.ndarray = cur["ykp1"]                       # (dim_y, 1)
            particles_npo: np.ndarray = nxt["particles"]       # (n_p, dim_x, 1)
            w_smooth_npo: np.ndarray = nxt["w_smooth"]         # (n_p,)

            # Propagate each particle through g with y_n inserted, take X-part:
            # mu_x[i] = [ g((particles_n[i], y_n), 0) ]_X
            y_tiled = np.tile(Yn, (self.n_particles, 1, 1))    # (n_p, dim_y, 1)
            z_in = np.concatenate([particles_n, y_tiled], axis=1)  # (n_p, dim_xy, 1)
            z_out = self.param.g(z_in, zeros_batched, self.dt)
            mu_x: np.ndarray = z_out[:, : self.dim_x, :]       # (n_p, dim_x, 1)

            # Pairwise log-density matrix
            #   log_D[i, j] = -0.5 (xi_{j,n+1} - mu_x[i])^T Sigma_xx^{-1} (xi_{j,n+1} - mu_x[i])
            # diff[i, j, :, 0] = xi_{j,n+1} - mu_x[i]
            diff = particles_npo[None, :, :, :] - mu_x[:, None, :, :]
            # quad[i, j] via two einsum-friendly contractions
            #   tmp[i, j, k] = Sigma_xx_inv[k, l] * diff[i, j, l, 0]
            #   quad[i, j]   = diff[i, j, k, 0] * tmp[i, j, k]
            tmp = np.einsum("kl,ijl->ijk", Sigma_xx_inv, diff[..., 0])
            quad = np.einsum("ijk,ijk->ij", diff[..., 0], tmp)
            log_D = -0.5 * quad                                # (n_p, n_p)

            # Backward kernel normaliser per column j:
            #   log Z_j = logsumexp_l ( log w_n[l] + log_D[l, j] )
            log_w_n = np.where(
                w_n > 0, np.log(np.maximum(w_n, 1e-300)), -np.inf,
            )
            log_inner = log_w_n[:, None] + log_D                # (n_p, n_p)
            log_Z = logsumexp(log_inner, axis=0)                # (n_p,)

            # FFBSm smoothed weights:
            #   tilde_w_n[i] = w_n[i] * sum_j tilde_w_npo[j] * D[i, j] / Z_j
            # Compute normalised backward kernel exp(log_D - log_Z[None, :])
            # (avoids forming the full Z divisor explicitly).
            log_norm_D = log_D - log_Z[None, :]                 # (n_p, n_p)
            # backward_sum_i = sum_j tilde_w_npo[j] * exp(log_norm_D[i, j])
            backward_sum = np.einsum(
                "ij,j->i", np.exp(log_norm_D), w_smooth_npo,
            )
            w_smooth_n = w_n * backward_sum

            total = w_smooth_n.sum()
            if not np.isfinite(total) or total <= 0.0:
                # Total degeneracy of the smoother weights — fall back to
                # uniform and log a warning. Mirrors the parent's
                # _safe_normalize_log_weights pattern.
                logger.warning(
                    "NonLinear_PPS step %d: smoothed weight total degenerate "
                    "(%.3g) — falling back to uniform.",
                    cur["k"], total,
                )
                w_smooth_n = np.full(self.n_particles, 1.0 / self.n_particles)
            else:
                w_smooth_n = w_smooth_n / total

            self._record_smoothed(idx, particles_n, w_smooth_n)

            if logger.isEnabledFor(logging.DEBUG):
                ess_smooth = 1.0 / np.sum(w_smooth_n**2)
                logger.debug(
                    "Step %d: ESS_smooth=%.1f/%d  weight_max=%.3g",
                    cur["k"], ess_smooth, self.n_particles, w_smooth_n.max(),
                )

        logger.info(
            "NonLinear_PPS backward pass complete (N_records=%d).", N_records,
        )

        for entry in self.history:
            yield (
                entry["k"],
                entry["xkp1"],
                entry["ykp1"],
                entry["Xkp1_predict"],
                entry["Xkp1_update"],
                entry["Xkp1_smooth"],
            )

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _record_smoothed(
        self,
        idx: int,
        particles: np.ndarray,
        w_smooth: np.ndarray,
    ) -> None:
        """
        Compute and record the smoothed mean and covariance for step
        ``idx`` from the particle cloud and smoothed weights.
        """
        # Smoothed mean: weighted average
        mean = np.einsum("i,ijk->jk", w_smooth, particles)        # (dim_x, 1)
        # Smoothed covariance: weighted sample covariance
        diff = particles - mean                                   # (n_p, dim_x, 1)
        cov = np.einsum("i,ijk,ilk->jl", w_smooth, diff, diff)
        # Safety symmetrisation against einsum-level floating asymmetry
        cov = 0.5 * (cov + cov.T)
        self.history.update_record(
            idx,
            w_smooth=w_smooth,
            Xkp1_smooth=mean,
            PXXkp1_smooth=cov,
        )

    def process_N_data_smoother(
        self,
        N: int | None,
        data_generator: Iterator | None = None,
    ) -> list[
        tuple[int, np.ndarray | None, np.ndarray, np.ndarray, np.ndarray, np.ndarray]
    ]:
        """
        Eager version of :meth:`process_smoother`. Same exception
        propagation policy as the Kalman-family smoothers.
        """
        try:
            return list(self.process_smoother(N=N, data_generator=data_generator))
        except RuntimeError as e:
            raise FilterError("Unexpected runtime error in PPS process_smoother.") from e
