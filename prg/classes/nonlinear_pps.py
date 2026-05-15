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

See ``Report/NonLinearSmoothingReport/Sections/Section6_PPS.tex`` for the
formal derivation, the pairwise transition-density formula, and the
Monte-Carlo convergence proof against :class:`Linear_PKS` on linear-gaussian
pairwise models.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator

import numpy as np
from scipy.linalg import LinAlgError, cho_factor, cho_solve
from scipy.special import logsumexp

from prg.classes.nonlinear_ppf import NonLinear_PPF
from prg.classes.param_linear import ParamLinear
from prg.classes.param_nonlinear import ParamNonLinear
from prg.utils.display import rich_show_fields
from prg.utils.exceptions import CovarianceError, FilterError
# NOTE: ParamError, InvertibilityError, NumericalError and StepValidationError
# may propagate from the inherited forward pass (cf. NonLinear_PPF.process_filter);
# they are listed in the Raises docstrings but not imported here as they are
# only re-raised, never constructed in this module.

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

    The smoothed covariance is validated through the same
    :meth:`_check_covariance` PSD diagnostic as the Kalman-family
    smoothers — a degenerate cloud (catastrophic ESS) that produces a
    non-PSD sample cov is caught rather than silently written to the
    history.

    There is no Joseph form for particle smoothers — Joseph is a
    Kalman-family numerical safeguard. The corresponding particle-side
    safeguard chain is log-sum-exp normalisation + degenerate-uniform
    fallback (see :ref:`Numerical safeguards` below).

    Parameters
    ----------
    param : ParamLinear | ParamNonLinear
        Forwarded to :class:`NonLinear_PPF` (which itself accepts both
        because the PPF reads only ``mQ`` and ``g`` from the param).
    n_particles : int, optional
        Number of particles. Default 300.
    resample_threshold, resample_method, sKey, verbose, particle_clip
        Forwarded to :class:`NonLinear_PPF`.
    store_particles : ignored
        The PPS forces ``store_particles=True`` internally because the
        backward pass needs the cloud — any user-supplied value is
        overridden.

    Cost
    ----
    Forward: :math:`O(N \\cdot n_p)`. Backward: :math:`O(N \\cdot n_p^2)`.
    Memory: ``N`` particle clouds of shape ``(n_p, dim_x, 1)`` plus the
    weight vectors of shape ``(n_p,)``, in addition to the regular
    PKFStep fields.

    Numerical safeguards
    --------------------
    The backward kernel evaluation uses ``scipy.special.logsumexp`` to
    avoid underflow when the forward and next-step particle clouds are
    far apart in state space. If the total of the smoothed weights
    underflows to zero or becomes non-finite at some step, the
    smoother falls back to uniform weights at that step with a
    ``WARNING`` log, mirroring the parent's
    :meth:`_safe_normalize_log_weights` pattern. The terminal
    sample covariance is checked via :meth:`_check_covariance`
    (Tikhonov-regularised if needed) so a degenerate cloud cannot
    silently write a non-PSD covariance to the history.

    Terminal-step caveat
    --------------------
    Unlike the Kalman-family smoothers (where :math:`\\hat X_{N|N}` and
    :math:`P^{xx}_{N|N}` from the forward equal the smoothed boundary
    value exactly), the PPS smoothed mean at step ``N`` uses the **raw
    particle cloud** weighted by ``w_smooth = weights``, whereas the
    PPF ``Xkp1_update`` uses the **Rao-Blackwellised estimator**
    :math:`\\sum_i w_i \\mu'_{x,i}`. Both target the same posterior but
    differ by Monte-Carlo variance of order :math:`O(\\sigma / \\sqrt{n_p})`.
    Boundary condition that **does** hold strictly: ``w_smooth[N] =
    weights[N]`` (initial condition of FFBSm).

    History schema additions
    ------------------------
    The forward PPF, when run via this class, attaches two keys to
    every history record (via the public
    :meth:`HistoryTracker.update_record` API):

    - ``particles`` of shape ``(n_p, dim_x, 1)``: the particle cloud
      (post-resampling, if resampling fired at this step).
    - ``weights`` of shape ``(n_p,)``: the corresponding particle
      weights (uniform after resampling; non-uniform otherwise).

    The backward pass then adds three further keys:

    - ``Xkp1_smooth`` of shape ``(dim_x, 1)``: smoothed posterior mean.
    - ``PXXkp1_smooth`` of shape ``(dim_x, dim_x)``: smoothed covariance.
    - ``w_smooth`` of shape ``(n_p,)``: smoothed weights
      :math:`\\tilde w_{i,n}` summing to 1.

    Unlike the Kalman smoothers, there is **no** ``Gk_smooth`` (no gain
    matrix in the particle backward recursion).
    """

    def __init__(
        self,
        param: ParamLinear | ParamNonLinear,
        *args,
        **kwargs,
    ) -> None:
        # Force particle storage on — the backward pass needs the cloud.
        kwargs["store_particles"] = True
        super().__init__(param, *args, **kwargs)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _propagate_particles_at(
        self,
        particles_n: np.ndarray,
        Yn: np.ndarray,
        z_buffer: np.ndarray,
        zeros_batched: np.ndarray,
    ) -> np.ndarray:
        """
        Vectorised application of ``g`` to the particle cloud with the
        observation ``y_n`` inserted between the X and noise blocks of
        the augmented state — mirrors the forward pass exactly.

        Parameters
        ----------
        particles_n : np.ndarray, shape ``(n_p, dim_x, 1)``
        Yn : np.ndarray, shape ``(dim_y, 1)``
        z_buffer : np.ndarray, shape ``(n_p, dim_xy, 1)``
            Pre-allocated scratch buffer (no per-call allocation).
        zeros_batched : np.ndarray, shape ``(n_p, dim_xy, 1)``
            Pre-allocated zero process-noise buffer.

        Returns
        -------
        z_out : np.ndarray, shape ``(n_p, dim_xy, 1)``
            ``g((particles_n[i], Yn), 0, dt)`` for each particle ``i``.
            The caller slices the first ``dim_x`` rows for ``mu_x``.
        """
        z_buffer[:, : self.dim_x] = particles_n
        z_buffer[:, self.dim_x :] = Yn   # broadcasts over the n_p axis
        return self.param.g(z_buffer, zeros_batched, self.dt)

    def _record_smoothed(
        self,
        idx: int,
        particles: np.ndarray,
        w_smooth: np.ndarray,
        k: int,
    ) -> None:
        """
        Compute and record the smoothed mean and covariance for step
        ``idx`` from the particle cloud and smoothed weights.

        Runs ``_check_covariance`` on the result so a degenerate cloud
        (catastrophic ESS) cannot silently write a non-PSD covariance.
        """
        # Smoothed mean: weighted average over particles
        mean = np.einsum("i,ijk->jk", w_smooth, particles)        # (dim_x, 1)
        # Smoothed covariance: weighted sample covariance
        diff = particles - mean                                   # (n_p, dim_x, 1)
        cov = np.einsum("i,ijk,ilk->jl", w_smooth, diff, diff)
        # Safety symmetrisation against einsum-level floating asymmetry
        cov = 0.5 * (cov + cov.T)
        self._check_covariance(cov, k, name="PXXkp1_smooth")
        self.history.update_record(
            idx,
            w_smooth=w_smooth,
            Xkp1_smooth=mean,
            PXXkp1_smooth=cov,
        )

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
        ParamError
            If ``N`` is not a strictly positive integer or ``None``
            (propagated from the PPF forward pass).
        InvertibilityError
            If a covariance matrix in the forward PPF (e.g. ``R``)
            cannot be inverted (propagated).
        CovarianceError
            (a) at construction-time if the X-marginal transition-noise
            covariance ``mQ[:p,:p]`` is not positive definite — its
            Cholesky is required to evaluate the backward kernel
            (carries ``step=-1`` and ``matrix_name="mQ[:p,:p]"``);
            (b) at every backward step if the resulting smoothed
            covariance :math:`P^{xx}_{n|N}` is not PSD (carries the
            offending ``step`` and ``matrix_name="PXXkp1_smooth"``).
        StepValidationError
            If the forward pass cannot construct a valid ``PKFStep``
            (propagated).
        NumericalError
            Base class of ``CovarianceError`` / ``InvertibilityError`` —
            catch this to intercept any matrix-level failure.
        FilterError
            If the forward pass yielded no records (defensive guard) or
            on unexpected propagated errors.
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
            N_records - 1,
            last["particles"],
            last["weights"].copy(),
            last["k"],
        )

        # 4) Pre-allocate scratch buffers reused at every backward step.
        zeros_batched: np.ndarray = np.zeros(
            (self.n_particles, self.dim_xy, 1)
        )
        z_buffer: np.ndarray = np.zeros((self.n_particles, self.dim_xy, 1))

        # 5) Backward FFBSm recursion
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
            z_out = self._propagate_particles_at(
                particles_n, Yn, z_buffer, zeros_batched,
            )
            mu_x: np.ndarray = z_out[:, : self.dim_x, :]       # (n_p, dim_x, 1)

            # Pairwise log-density matrix
            #   log_D[i, j] = -0.5 (xi_{j,n+1} - mu_x[i])^T Sigma_xx^{-1} (xi_{j,n+1} - mu_x[i])
            # diff[i, j, :, 0] = xi_{j,n+1} - mu_x[i]
            diff = particles_npo[None, :, :, :] - mu_x[:, None, :, :]
            # Fused contraction with path optimisation — ~20-30% faster than
            # two sequential einsums on the test grid.
            quad = np.einsum(
                "ijk,kl,ijl->ij",
                diff[..., 0], Sigma_xx_inv, diff[..., 0],
                optimize=True,
            )
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
            log_norm_D = log_D - log_Z[None, :]                 # (n_p, n_p)
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

            self._record_smoothed(idx, particles_n, w_smooth_n, cur["k"])

            if logger.isEnabledFor(logging.DEBUG):
                ess_smooth = 1.0 / np.sum(w_smooth_n**2)
                logger.debug(
                    "Step %d: ESS_smooth=%.1f/%d  weight_max=%.3g",
                    cur["k"], ess_smooth, self.n_particles, w_smooth_n.max(),
                )

            if self.verbose > 1:
                rich_show_fields(
                    self.history[idx], title=f"PPS smoothed step {cur['k']}"
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

    def process_N_data_smoother(
        self,
        N: int | None,
        data_generator: Iterator | None = None,
    ) -> list[
        tuple[int, np.ndarray | None, np.ndarray, np.ndarray, np.ndarray, np.ndarray]
    ]:
        """
        Eager version of :meth:`process_smoother` — runs ``N`` steps and
        returns all outputs as a list.

        Exception-handling policy
        -------------------------
        All domain-specific exceptions raised by :meth:`process_smoother`
        — ``ParamError``, ``InvertibilityError``, ``CovarianceError``,
        ``NumericalError``, ``StepValidationError`` and direct
        ``FilterError`` — propagate up **unwrapped**. Their structured
        ``step`` / ``matrix_name`` attributes are preserved.

        Only an opaque ``RuntimeError`` raised by Python's generator
        machinery (e.g. an inadvertent ``StopIteration`` re-raised by
        PEP 479) is wrapped as ``FilterError`` and chained via ``from``
        for traceability.

        Raises
        ------
        ParamError, InvertibilityError, CovarianceError, NumericalError, StepValidationError
            Propagated unwrapped from :meth:`process_smoother`.
        FilterError
            Either propagated from :meth:`process_smoother` (forward
            empty or unexpected error), or wrapping an opaque
            ``RuntimeError`` from the generator machinery.
        """
        try:
            return list(self.process_smoother(N=N, data_generator=data_generator))
        except RuntimeError as e:
            raise FilterError("Unexpected runtime error in PPS process_smoother.") from e
