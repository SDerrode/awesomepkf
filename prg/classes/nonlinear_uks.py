"""
####################################################################
Unscented Kalman Smoother (UKS) — classical RTS sigma-point smoother
####################################################################

Unlike the pairwise smoothers (Linear_PKS / NonLinear_EPKS / NonLinear_UPKS),
the classical UKS operates on the X-only Markov chain: the model is
``x_{n+1} = f(x_n) + w_n``, ``y_n = h(x_n) + e_n`` with independent
process and observation noises. The smoothing gain is therefore a
``(dim_x, dim_x)`` matrix (not ``(dim_x, dim_x + dim_y)``).
"""

from __future__ import annotations

import logging
from collections.abc import Iterator

import numpy as np
from scipy.linalg import LinAlgError, cho_factor, cho_solve

from prg.classes.nonlinear_ukf import NonLinear_UKF
from prg.classes.param_linear import ParamLinear
from prg.classes.param_nonlinear import ParamNonLinear
from prg.utils.display import rich_show_fields
from prg.utils.exceptions import CovarianceError, FilterError

# NOTE: ParamError, InvertibilityError, NumericalError and StepValidationError
# may propagate from the inherited forward pass (cf. NonLinear_UKF.process_filter);
# they are listed in the Raises docstrings but not imported here as they are
# only re-raised, never constructed in this module.

logger = logging.getLogger(__name__)

__all__ = ["NonLinear_UKS"]


class NonLinear_UKS(NonLinear_UKF):
    """
    Unscented Kalman Smoother (UKS) for classical (non-pairwise)
    state-space models.

    Two-pass smoother extending :class:`NonLinear_UKF`. The forward pass
    is the standard UKF; the backward pass is the classical Rauch-Tung-
    Striebel recursion with sigma-point cross-covariance:

    .. math::

        C_n \\;=\\;
            \\mathrm{Cov}(X_n, X_{n+1} \\mid y_{1:n})\\,
            (P^{xx}_{n+1|n})^{-1}
        \\;\\in\\; \\RR^{p \\times p}.

    The cross-covariance is estimated from sigma points regenerated at
    ``(X_{n|n}, P^{xx}_{n|n})`` and propagated through ``f``. The
    predicted covariance ``P^{xx}_{n+1|n}`` — which includes the
    additive process-noise term ``Q_x`` — is read from the forward
    history record, so the smoothing gain remains consistent with the
    forward filter.

    Parameters
    ----------
    param : ParamLinear | ParamNonLinear
        Must expose ``f`` (state transition) and ``h`` (observation),
        i.e. ``param.pairwiseModel`` must be ``False``.
    sigmaSet : str
        Key of the sigma-point set in :attr:`SigmaPointsSet.registry`.
        Forwarded to :class:`NonLinear_UKF`.
    sKey, verbose
        Standard.
    joseph : bool, optional
        If ``True``, use the Joseph form
        :math:`(I_p, -C_n)\\, \\Omega_n\\, (I_p, -C_n)^\\top + C_n P^{xx}_{n+1|N} C_n^\\top`
        with :math:`\\Omega_n` the joint covariance of
        :math:`(X_n, X_{n+1})` conditional on :math:`y_{1:n}`. Default
        ``False``. Empirically equivalent to the standard form to
        ``~1e-10`` on the test fixtures.

    Cost
    ----
    The backward pass regenerates sigma points (dimension ``dim_x``
    only, not augmented with the observation or process noise — much
    cheaper than the UPKS regeneration). Total cost is roughly the
    forward UKF + one extra Cholesky of size ``dim_x`` and
    ``n_\\sigma`` calls to ``param.f`` per backward step.

    History schema additions
    ------------------------
    Each forward record is augmented with three keys by the backward
    pass: ``Xkp1_smooth`` of shape ``(dim_x, 1)``, ``PXXkp1_smooth`` of
    shape ``(dim_x, dim_x)``, and ``Gk_smooth`` of shape
    ``(dim_x, dim_x)``. **Note the shape difference with the pairwise
    smoothers**: the UKS gain is square ``dim_x × dim_x``, not
    ``dim_x × dim_xy`` — a direct consequence of the X-only Markov
    structure (no future-observation contribution in the residual). At
    the terminal step ``n = N`` the gain is undefined; a zero matrix
    of the correct shape is stored as a placeholder. All three fields
    are written via the public :meth:`HistoryTracker.update_record` API.
    """

    def __init__(
        self,
        param: ParamLinear | ParamNonLinear,
        sigmaSet: str,
        sKey: int | None = None,
        verbose: int = 0,
        joseph: bool = False,
    ) -> None:
        super().__init__(param, sigmaSet, sKey, verbose)
        self.joseph: bool = joseph

    # ------------------------------------------------------------------
    # Helper: sigma-point propagation through f at step n
    # ------------------------------------------------------------------

    def _propagate_sigma_f_at(
        self,
        Xf_n: np.ndarray,
        Pf_n: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Regenerate ``dim_x``-dimensional sigma points around
        ``(X_{n|n}, P^{xx}_{n|n})`` and propagate them through ``f``.

        Returns
        -------
        sigma_pred : np.ndarray, shape ``(n_sigma, dim_x, 1)``
            Sigma points before propagation — used for the X-side of
            the cross-covariance.
        sigma_f : np.ndarray, shape ``(n_sigma, dim_x, 1)``
            Sigma points after propagation through ``f``.
        """
        sigma_list = self.sigma_pred_set._sigma_point(Xf_n, Pf_n)
        sigma_pred = np.array(sigma_list)             # (n_sigma, dim_x, 1)
        n_sigma = sigma_pred.shape[0]
        zeros_x = np.zeros((n_sigma, self.dim_x, 1))
        sigma_f = self.param.f(sigma_pred, zeros_x, self.dt)
        return sigma_pred, sigma_f

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
        Run the UKS as a generator.

        Exhausts the UKF forward (populating :attr:`history`), then runs
        the backward sigma-point RTS recursion in place via
        :meth:`HistoryTracker.update_record`, and yields the augmented
        tuples in chronological order.

        Yields
        ------
        k, x_true, y_observed, X_predict, X_update, X_smooth

        Raises
        ------
        ParamError
            If ``N`` is not a strictly positive integer or ``None``
            (propagated from the forward pass via ``_validate_N``), or
            if the constructor-time sigma-point set name was unknown
            (propagated from :class:`NonLinear_UKF`).
        InvertibilityError
            If the forward-pass innovation covariance is not invertible
            (propagated).
        CovarianceError
            (a) from forward covariance checks; (b) from backward
            Cholesky failure of :math:`P^{xx}_{n+1|n}`; (c) from PSD
            violation of :math:`P^{xx}_{n|N}`. In cases (b) and (c) the
            exception carries ``step`` and ``matrix_name`` attributes.
        StepValidationError
            If the forward pass cannot construct a valid ``PKFStep``
            (propagated).
        NumericalError
            Base class of ``CovarianceError`` / ``InvertibilityError`` —
            catch this to intercept any matrix-level failure regardless
            of subtype.
        FilterError
            If the forward pass yielded no records (defensive guard) or
            on unexpected propagated errors.
        """
        # 1) Forward pass — UKF populates self.history
        for _ in self.process_filter(N=N, data_generator=data_generator):
            pass

        N_records: int = len(self.history)
        if N_records == 0:
            raise FilterError("NonLinear_UKS: forward pass yielded no records.")

        logger.info(
            "NonLinear_UKS backward pass starting (N_records=%d, joseph=%s).",
            N_records,
            self.joseph,
        )

        # Terminal step: smoother = filter at n = N. Unlike the pairwise
        # variants the smoothing gain at the terminal step is (dim_x, dim_x).
        last_idx = N_records - 1
        last = self.history[last_idx]
        self.history.update_record(
            last_idx,
            Xkp1_smooth=last["Xkp1_update"].copy(),
            PXXkp1_smooth=last["PXXkp1_update"].copy(),
            Gk_smooth=np.zeros((self.dim_x, self.dim_x)),
        )

        if self.joseph:
            # Joint covariance Omega_n = Cov((X_n, X_{n+1}) | y_{1:n})
            # of shape (2*dim_x, 2*dim_x)
            dim_jnt: int = 2 * self.dim_x
            Omega = np.zeros((dim_jnt, dim_jnt))
            J = np.zeros((self.dim_x, dim_jnt))
            J[: self.dim_x, : self.dim_x] = self.eye_dim_x

        Wm = self.sigma_pred_set.Wm
        Wc = self.sigma_pred_set.Wc

        for i in range(N_records - 2, -1, -1):
            cur = self.history[i]
            nxt = self.history[i + 1]

            Xf_n: np.ndarray = cur["Xkp1_update"]
            Pf_n: np.ndarray = cur["PXXkp1_update"]

            Xs_npo: np.ndarray = nxt["Xkp1_smooth"]
            Ps_npo: np.ndarray = nxt["PXXkp1_smooth"]
            Pp_npo: np.ndarray = nxt["PXXkp1_predict"]   # already includes Q_x

            # Regenerate sigma points (dim_x only) + propagate through f
            sigma_pred, sigma_f = self._propagate_sigma_f_at(Xf_n, Pf_n)

            # Backward-recomputed predicted mean (used self-consistently for
            # both the cross-covariance diffs and the mean-update residual).
            Xhat_npo = np.sum(Wm[:, None, None] * sigma_f, axis=0)

            # Cross-covariance Cov(X_n, X_{n+1} | y_{1:n}) — shape (dim_x, dim_x).
            # Note: additive Q_x does NOT enter the cross-covariance — it is
            # noise injected after f and independent of X_n.
            diffs_X = sigma_pred - Xf_n
            diffs_F = sigma_f - Xhat_npo
            cross = np.einsum("i,ijk,ilk->jl", Wc, diffs_X, diffs_F)

            # Smoothing gain. Pp_npo comes from the forward (includes Q_x),
            # so the gain is consistent with the forward predicted covariance.
            try:
                c, low = cho_factor(Pp_npo)
                Cn: np.ndarray = cho_solve((c, low), cross.T).T
            except (LinAlgError, ValueError) as e:
                raise CovarianceError(
                    f"Step {cur['k']}: Cholesky factorisation failed for "
                    f"PXXkp1_predict in UKS backward pass.",
                    matrix_name="PXXkp1_predict",
                    step=cur["k"],
                ) from e

            # Smoothed mean
            Xs_n: np.ndarray = Xf_n + Cn @ (Xs_npo - Xhat_npo)

            # Smoothed covariance
            if self.joseph:
                Omega[: self.dim_x, : self.dim_x] = Pf_n
                Omega[: self.dim_x, self.dim_x :] = cross
                Omega[self.dim_x :, : self.dim_x] = cross.T
                Omega[self.dim_x :, self.dim_x :] = Pp_npo
                J[:, self.dim_x :] = -Cn
                Ps_n: np.ndarray = J @ Omega @ J.T + Cn @ Ps_npo @ Cn.T
            else:
                Ps_n = Pf_n + Cn @ (Ps_npo - Pp_npo) @ Cn.T

            Ps_n = 0.5 * (Ps_n + Ps_n.T)
            self._check_covariance(Ps_n, cur["k"], name="PXXkp1_smooth")

            self.history.update_record(
                i,
                Xkp1_smooth=Xs_n,
                PXXkp1_smooth=Ps_n,
                Gk_smooth=Cn,
            )

            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(
                    "Step %d: |Cn|_F=%.3g, tr(P_smooth)=%.3g, tr(P_filt)=%.3g.",
                    cur["k"],
                    float(np.linalg.norm(Cn)),
                    float(np.trace(Ps_n)),
                    float(np.trace(Pf_n)),
                )

            if self.verbose > 1:
                rich_show_fields(
                    self.history[i], title=f"UKS smoothed step {cur['k']}"
                )

        logger.info(
            "NonLinear_UKS backward pass complete (N_records=%d).", N_records,
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
        Eager version of :meth:`process_smoother`. Same exception
        propagation policy as :class:`Linear_PKS.process_N_data_smoother`.
        """
        try:
            return list(self.process_smoother(N=N, data_generator=data_generator))
        except RuntimeError as e:
            raise FilterError("Unexpected runtime error in UKS process_smoother.") from e
