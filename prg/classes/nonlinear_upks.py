"""
####################################################################
Unscented Pairwise Kalman Smoother (UPKS) — RTS-pairwise smoother on
top of the Unscented Pairwise Kalman Filter (UPKF).
####################################################################
"""

from __future__ import annotations

import logging
from collections.abc import Iterator

import numpy as np
from scipy.linalg import LinAlgError, cho_factor, cho_solve

from prg.classes.nonlinear_upkf import NonLinear_UPKF
from prg.classes.param_linear import ParamLinear
from prg.classes.param_nonlinear import ParamNonLinear
from prg.utils.display import rich_show_fields
from prg.utils.exceptions import CovarianceError, FilterError
# NOTE: ParamError, InvertibilityError, NumericalError and StepValidationError
# may propagate from the inherited forward pass.

logger = logging.getLogger(__name__)

__all__ = ["NonLinear_UPKS"]


class NonLinear_UPKS(NonLinear_UPKF):
    """
    Unscented Pairwise Kalman Smoother (UPKS).

    Two-pass smoother extending :class:`NonLinear_UPKF`. The forward
    pass is the standard UPKF (sigma-point propagation around the
    filtered state). The backward pass mirrors the linear PKS recursion
    with **sigma-point computed** cross-covariance and predicted joint
    covariance replacing their analytical-Jacobian counterparts of the
    EPKS.

    Sigma points are regenerated in the backward pass at the same
    augmented mean ``(X_{n|n}, 0_{noise})`` and augmented covariance
    ``diag(P^{xx}_{n|n}, mQ)`` as the forward used. The pairwise
    observation ``y_n`` is inserted between the state and the noise
    components before evaluating ``g``, exactly like the forward.

    The smoothing gain is

    .. math::

        G_n \\;=\\; \\mathrm{Cov}(X_n, Z_{n+1} \\mid y_{1:n})\\,
                    (P^{ZZ}_{n+1|n})^{-1}
        \\;\\in\\; \\RR^{p \\times (p+q)},

    with both factors estimated from sigma points. The two covariance
    update forms — standard and Joseph — are available via the
    ``joseph`` flag, identical to :class:`Linear_PKS` and
    :class:`NonLinear_EPKS`.

    Parameters
    ----------
    param : ParamLinear | ParamNonLinear
    sigmaSet : str
        Key of the sigma-point set in :attr:`SigmaPointsSet.registry`
        (e.g. ``"wan2000"``, ``"cpkf"``, ``"lerner2002"``, ``"ito2000"``).
        Forwarded to :class:`NonLinear_UPKF`.
    sKey : int, optional
    verbose : int, optional
    joseph : bool, optional
        If ``True``, use the Joseph form of the covariance update.
        Empirically equivalent to the standard form to ``~1e-10`` on the
        test fixtures.

    Cost
    ----
    The backward pass regenerates and propagates a new set of sigma
    points at every step (the forward does not store them). Total cost
    is therefore roughly twice the forward UPKF — ``2 N \\cdot n_\\sigma``
    calls to ``param.g``, where ``n_\\sigma`` depends on the chosen
    sigma-point set (typically ``2 \\cdot (2 p + q) + 1`` for
    ``wan2000``).

    History schema additions
    ------------------------
    Each forward record is augmented with three keys by the backward
    pass: ``Xkp1_smooth`` of shape ``(dim_x, 1)``, ``PXXkp1_smooth`` of
    shape ``(dim_x, dim_x)``, and ``Gk_smooth`` of shape
    ``(dim_x, dim_xy)``. At the terminal step ``n = N`` the smoothing
    gain is undefined; a zero matrix of the correct shape is stored as
    a placeholder. All three fields are written via the public
    :meth:`HistoryTracker.update_record` API.
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
    # Helper: sigma-point propagation at step n
    # ------------------------------------------------------------------

    def _propagate_sigma_at(
        self,
        Xf_n: np.ndarray,
        Pf_n: np.ndarray,
        Yn: np.ndarray,
        Pa_base: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Regenerate sigma points around ``(X_{n|n}, 0_{noise})``, propagate
        them through ``g`` with the observation ``y_n`` inserted, and
        return the propagated stack along with the X-part of the input
        sigma stack (needed for the cross-covariance).

        Mirrors the forward pass code path of
        :meth:`NonLinear_UPKF.process_filter` exactly.

        Returns
        -------
        sigma_X : np.ndarray, shape ``(n_sigma, dim_x, 1)``
            X-part of the sigma points before propagation — used for the
            X-side of the cross-covariance.
        sigma_propag : np.ndarray, shape ``(n_sigma, dim_xy, 1)``
            Sigma points after propagation through ``g``.
        """
        n_aug: int = 2 * self.dim_x + self.dim_y

        za = np.zeros((n_aug, 1))
        za[: self.dim_x] = Xf_n

        Pa = Pa_base.copy()
        Pa[: self.dim_x, : self.dim_x] = Pf_n

        sigma_without_y = self.sigma_point_set_obj._sigma_point(za, Pa)
        sigma_stack = np.array(sigma_without_y)         # (n_sigma, n_aug, 1)
        n_sigma = sigma_stack.shape[0]

        y_tiled = np.tile(Yn, (n_sigma, 1, 1))           # (n_sigma, dim_y, 1)
        sigma_with_y = np.concatenate(
            [
                sigma_stack[:, : self.dim_x, :],         # X part
                y_tiled,                                 # observation
                sigma_stack[:, self.dim_x :, :],         # noise part
            ],
            axis=1,
        )                                                # (n_sigma, 2*dim_xy, 1)

        z_batch, noise_batch = np.split(
            sigma_with_y, [self.dim_xy], axis=1
        )
        sigma_propag = self.param.g(z_batch, noise_batch, self.dt)

        sigma_X = sigma_stack[:, : self.dim_x, :]
        return sigma_X, sigma_propag

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
        Run the UPKS as a generator.

        First exhausts the UPKF forward (populating :attr:`history`),
        then performs the backward sigma-point RTS recursion in place
        via :meth:`HistoryTracker.update_record`, and yields the
        augmented tuples in chronological order.

        Yields
        ------
        k, x_true, y_observed, X_predict, X_update, X_smooth

        Raises
        ------
        ParamError
            From ``_validate_N`` (forward).
        InvertibilityError
            Propagated from the forward UPKF.
        CovarianceError
            (a) from forward covariance checks; (b) from backward
            Cholesky failure of :math:`P^{ZZ}_{n+1|n}`; (c) from PSD
            violation of :math:`P^{xx}_{n|N}`.
        StepValidationError, NumericalError, FilterError
            Propagated from the forward pass or raised on a defensive
            empty-history guard.
        """
        # 1) Forward pass — drains UPKF into self.history
        for _ in self.process_filter(N=N, data_generator=data_generator):
            pass

        N_records: int = len(self.history)
        if N_records == 0:
            raise FilterError("NonLinear_UPKS: forward pass yielded no records.")

        logger.info(
            "NonLinear_UPKS backward pass starting (N_records=%d, joseph=%s).",
            N_records,
            self.joseph,
        )

        # Terminal step initialisation
        last = self.history[N_records - 1]
        self.history.update_record(
            N_records - 1,
            Xkp1_smooth=last["Xkp1_update"].copy(),
            PXXkp1_smooth=last["PXXkp1_update"].copy(),
            Gk_smooth=np.zeros((self.dim_x, self.dim_xy)),
        )

        # Pre-allocated buffers — augmented covariance base (the noise block
        # is constant; only the X block is overwritten each iteration).
        n_aug: int = 2 * self.dim_x + self.dim_y
        Pa_base = np.zeros((n_aug, n_aug))
        Pa_base[self.dim_x :, self.dim_x :] = self.param.mQ

        P_zz_smooth = self.zeros_dim_xy_xy.copy()
        delta_Z = self.zeros_dim_xy_1.copy()

        if self.joseph:
            dim_jnt: int = self.dim_x + self.dim_xy
            Omega = np.zeros((dim_jnt, dim_jnt))
            J = np.zeros((self.dim_x, dim_jnt))
            J[: self.dim_x, : self.dim_x] = self.eye_dim_x

        Wm = self.sigma_point_set_obj.Wm
        Wc = self.sigma_point_set_obj.Wc

        for i in range(N_records - 2, -1, -1):
            cur = self.history[i]
            nxt = self.history[i + 1]

            Xf_n: np.ndarray = cur["Xkp1_update"]
            Pf_n: np.ndarray = cur["PXXkp1_update"]
            Yn: np.ndarray = cur["ykp1"]

            Xs_npo: np.ndarray = nxt["Xkp1_smooth"]
            Ps_npo: np.ndarray = nxt["PXXkp1_smooth"]
            Ynpo: np.ndarray = nxt["ykp1"]  # y_{n+1}

            # Regenerate + propagate sigma points at step n's linearisation
            # point (same as the forward).
            sigma_X, sigma_propag = self._propagate_sigma_at(
                Xf_n, Pf_n, Yn, Pa_base
            )

            # Predicted joint mean from sigma points (use this consistently
            # for both the cross-covariance diffs AND the mean-update Δ).
            Zhat_npo = np.sum(
                Wm[:, None, None] * sigma_propag, axis=0
            )                                            # (dim_xy, 1)

            # Predicted joint covariance from sigma points
            diffs_Z = sigma_propag - Zhat_npo            # (n_sigma, dim_xy, 1)
            P_zz_npo = np.einsum(
                "i,ijk,ilk->jl", Wc, diffs_Z, diffs_Z
            )
            P_zz_npo = 0.5 * (P_zz_npo + P_zz_npo.T)

            # Cross-covariance Cov(X_n, Z_{n+1} | y_{1:n}) — shape (dim_x, dim_xy)
            diffs_X = sigma_X - Xf_n                     # (n_sigma, dim_x, 1)
            cross_X = np.einsum(
                "i,ijk,ilk->jl", Wc, diffs_X, diffs_Z
            )

            # Smoothing gain
            try:
                c, low = cho_factor(P_zz_npo)
                Gn: np.ndarray = cho_solve((c, low), cross_X.T).T
            except (LinAlgError, ValueError) as e:
                raise CovarianceError(
                    f"Step {cur['k']}: Cholesky factorisation failed for "
                    f"PZZkp1_predict in UPKS backward pass.",
                    matrix_name="PZZkp1_predict",
                    step=cur["k"],
                ) from e

            # Mean update — uses backward-recomputed Zhat to stay
            # self-consistent with the cross-covariance diffs.
            #     delta_Z = (X_{n+1|N} - Xhat_{n+1|n}^bw ; y_{n+1} - Yhat_{n+1|n}^bw)
            delta_Z[: self.dim_x] = Xs_npo - Zhat_npo[: self.dim_x]
            delta_Z[self.dim_x :] = Ynpo - Zhat_npo[self.dim_x :]
            Xs_n: np.ndarray = Xf_n + Gn @ delta_Z

            # Covariance update
            if self.joseph:
                Omega[: self.dim_x, : self.dim_x] = Pf_n
                Omega[: self.dim_x, self.dim_x :] = cross_X
                Omega[self.dim_x :, : self.dim_x] = cross_X.T
                Omega[self.dim_x :, self.dim_x :] = P_zz_npo
                J[:, self.dim_x :] = -Gn
                Gn_x: np.ndarray = Gn[:, : self.dim_x]
                Ps_n: np.ndarray = J @ Omega @ J.T + Gn_x @ Ps_npo @ Gn_x.T
            else:
                P_zz_smooth[: self.dim_x, : self.dim_x] = Ps_npo
                Delta_P_ZZ: np.ndarray = P_zz_smooth - P_zz_npo
                Ps_n = Pf_n + Gn @ Delta_P_ZZ @ Gn.T

            Ps_n = 0.5 * (Ps_n + Ps_n.T)
            self._check_covariance(Ps_n, cur["k"], name="PXXkp1_smooth")

            self.history.update_record(
                i,
                Xkp1_smooth=Xs_n,
                PXXkp1_smooth=Ps_n,
                Gk_smooth=Gn,
            )

            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(
                    "Step %d: |Gn|_F=%.3g, tr(P_smooth)=%.3g, tr(P_filt)=%.3g.",
                    cur["k"],
                    float(np.linalg.norm(Gn)),
                    float(np.trace(Ps_n)),
                    float(np.trace(Pf_n)),
                )

            if self.verbose > 1:
                rich_show_fields(
                    self.history[i], title=f"UPKS smoothed step {cur['k']}"
                )

        logger.info(
            "NonLinear_UPKS backward pass complete (N_records=%d).", N_records,
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
            raise FilterError("Unexpected runtime error in UPKS process_smoother.") from e
