"""
####################################################################
Extended Pairwise Kalman Smoother (EPKS) — RTS pairwise smoother on
top of the Extended Pairwise Kalman Filter (EPKF).
####################################################################
"""

from __future__ import annotations

import logging
from collections.abc import Iterator

import numpy as np
from scipy.linalg import LinAlgError, cho_factor, cho_solve

from prg.classes.nonlinear_epkf import NonLinear_EPKF
from prg.classes.param_linear import ParamLinear
from prg.classes.param_nonlinear import ParamNonLinear
from prg.utils.display import rich_show_fields
from prg.utils.exceptions import CovarianceError, FilterError, ParamError
# NOTE: InvertibilityError, NumericalError and StepValidationError may propagate
# from the inherited forward pass (cf. NonLinear_EPKF.process_filter); they are
# listed in the Raises docstrings but not imported here.

logger = logging.getLogger(__name__)

__all__ = ["NonLinear_EPKS"]


class NonLinear_EPKS(NonLinear_EPKF):
    """
    Extended Pairwise Kalman Smoother (EPKS).

    Two-pass smoother extending :class:`NonLinear_EPKF`. The forward pass
    is the standard EPKF (first-order linearisation around the filtered
    state). The backward pass is the linear PKS recursion applied to the
    *per-step linearised* dynamics: at step ``n``, the Jacobian
    :math:`\\mathbf{F}_{n+1}` evaluated at :math:`(X_{n|n}, y_n)` plays
    the role of the constant ``A`` matrix of the linear case.

    Mathematically equivalent to running the classical RTS smoother on
    the augmented state ``Z = (X, Y)`` with the EKF's per-step
    linearisation, just as the EPKF forward equals the augmented-state
    EKF forward (see :cite:`Derrode2025NonLinearPKF`, §2 Rem PKFasKF).

    Implementation notes
    --------------------
    The Jacobian is **recomputed** in the backward pass at the same
    linearisation point :math:`(X_{n|n}, y_n)` as the forward used. The
    linearisation point is stored in the forward record (``Xkp1_update``
    and ``ykp1``), so reuse is exact. Cost: one extra Jacobian
    evaluation per step — same order of magnitude as the forward.

    Parameters
    ----------
    param : ParamLinear | ParamNonLinear
        Forwarded to :class:`NonLinear_EPKF`.
    sKey : int, optional
    verbose : int, optional
    joseph : bool, optional
        If ``True``, use the Joseph form of the covariance update
        (explicitly symmetric / PSD-preserving for any gain). Default
        ``False``. The two forms agree to ``~1e-13`` in double precision
        when the EPKF's per-step Jacobian is well-conditioned.

    History schema additions
    ------------------------
    Each forward record is augmented with three keys by the backward
    pass: ``Xkp1_smooth``, ``PXXkp1_smooth``, ``Gk_smooth`` (the same
    schema as :class:`Linear_PKS`). The smoothing gain ``Gk_smooth`` has
    shape ``(dim_x, dim_xy)``.
    """

    def __init__(
        self,
        param: ParamLinear | ParamNonLinear,
        sKey: int | None = None,
        verbose: int = 0,
        joseph: bool = False,
    ) -> None:
        super().__init__(param, sKey, verbose)
        self.joseph: bool = joseph

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _jacobian_at(self, z_lin: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Evaluate ``param.jacobiens_g`` at the augmented state ``z_lin``
        and return the two ``(dim_xy, dim_xy)`` matrices ``(An, Bn)``.

        Mirrors the shape validation performed in
        :meth:`NonLinear_EPKF.process_filter` so the backward pass
        cannot silently drift from the forward.
        """
        An, Bn = self.param.jacobiens_g(z_lin, self.zeros_dim_xy_1, self.dt)
        expected_shape = (self.dim_xy, self.dim_xy)
        if An.ndim == 2:
            if An.shape != expected_shape or Bn.shape != expected_shape:
                raise ParamError(
                    f"Jacobian returned matrices of wrong shape in backward "
                    f"pass: An={An.shape}, Bn={Bn.shape}, expected "
                    f"{expected_shape}."
                )
            return An, Bn
        # Batched 3D Jacobian (rare): take the first slice, consistent
        # with what the forward pass does implicitly.
        if An.shape[1:] != expected_shape or Bn.shape[1:] != expected_shape:
            raise ParamError(
                f"Jacobian returned matrices of wrong shape in backward "
                f"pass: An={An.shape}, Bn={Bn.shape}, expected "
                f"(N, {self.dim_xy}, {self.dim_xy})."
            )
        return An[0], Bn[0]

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
        Run the EPKS as a generator.

        First exhausts the EPKF forward (populating :attr:`history`),
        then performs the backward RTS recursion in place via
        :meth:`HistoryTracker.update_record`, and yields the augmented
        tuples in chronological order.

        Yields
        ------
        k, x_true, y_observed, X_predict, X_update, X_smooth
            ``X_smooth`` is the smoothed posterior mean
            :math:`E[X_n | y_{1:N}]` under the per-step EPKF
            linearisation.

        Raises
        ------
        ParamError
            From ``_validate_N`` (forward) or from shape validation of
            the Jacobian returned by ``param.jacobiens_g``.
        InvertibilityError
            Propagated from the forward EPKF.
        CovarianceError
            (a) from forward covariance checks; (b) from backward
            Cholesky failure of :math:`P^{ZZ}_{n+1|n}`; (c) from PSD
            violation of :math:`P^{xx}_{n|N}`.
        StepValidationError, NumericalError, FilterError
            Propagated from the forward pass or raised on a defensive
            "forward produced no records" guard.
        """
        # 1) Forward pass — drains EPKF into self.history
        for _ in self.process_filter(N=N, data_generator=data_generator):
            pass

        N_records: int = len(self.history)
        if N_records == 0:
            raise FilterError("NonLinear_EPKS: forward pass yielded no records.")

        logger.info(
            "NonLinear_EPKS backward pass starting (N_records=%d, joseph=%s).",
            N_records,
            self.joseph,
        )

        # 2) Terminal step — smoother = filter at n = N
        last = self.history[N_records - 1]
        self.history.update_record(
            N_records - 1,
            Xkp1_smooth=last["Xkp1_update"].copy(),
            PXXkp1_smooth=last["PXXkp1_update"].copy(),
            Gk_smooth=np.zeros((self.dim_x, self.dim_xy)),
        )

        # Pre-allocated scratch buffers (writable blocks only — see
        # Linear_PKS for the invariant rationale).
        P_aug: np.ndarray = self.zeros_dim_xy_xy.copy()
        P_zz_smooth: np.ndarray = self.zeros_dim_xy_xy.copy()
        delta_Z: np.ndarray = self.zeros_dim_xy_1.copy()
        z_lin: np.ndarray = self.zeros_dim_xy_1.copy()

        if self.joseph:
            dim_jnt: int = self.dim_x + self.dim_xy
            Omega = np.zeros((dim_jnt, dim_jnt))
            J = np.zeros((self.dim_x, dim_jnt))
            J[: self.dim_x, : self.dim_x] = self.eye_dim_x

        for i in range(N_records - 2, -1, -1):
            cur = self.history[i]
            nxt = self.history[i + 1]

            Xf_n: np.ndarray = cur["Xkp1_update"]
            Pf_n: np.ndarray = cur["PXXkp1_update"]
            Yn: np.ndarray = cur["ykp1"]

            Xp_npo: np.ndarray = nxt["Xkp1_predict"]
            ikp1: np.ndarray = nxt["ikp1"]
            Xs_npo: np.ndarray = nxt["Xkp1_smooth"]
            Ps_npo: np.ndarray = nxt["PXXkp1_smooth"]

            # Re-linearise at the exact point the forward used at step n.
            # NOTE: in the EPKF forward, the Jacobian was evaluated at
            # z_iterated = (X_{n|n}, y_n) just before predicting n -> n+1.
            z_lin[: self.dim_x] = Xf_n
            z_lin[self.dim_x :] = Yn
            An, Bn = self._jacobian_at(z_lin)

            # Predicted joint covariance P^{ZZ}_{n+1|n}:
            #   P_{n+1|n} = A diag(P^{xx}_{n|n}, 0) A^T + B Q B^T
            P_aug[: self.dim_x, : self.dim_x] = Pf_n
            P_zz_npo: np.ndarray = An @ P_aug @ An.T + Bn @ self.param.mQ @ Bn.T
            P_zz_npo = 0.5 * (P_zz_npo + P_zz_npo.T)  # forward also symmetrises

            # Cross-cov Cov(X_n, Z_{n+1} | y_{1:n}) = [P^{xx}_{n|n}, 0] A^T
            cross_X: np.ndarray = (P_aug @ An.T)[: self.dim_x, :]

            # Gain
            try:
                c, low = cho_factor(P_zz_npo)
                Gn: np.ndarray = cho_solve((c, low), cross_X.T).T
            except (LinAlgError, ValueError) as e:
                raise CovarianceError(
                    f"Step {cur['k']}: Cholesky factorisation failed for "
                    f"PZZkp1_predict in EPKS backward pass.",
                    matrix_name="PZZkp1_predict",
                    step=cur["k"],
                ) from e

            # Smoothed mean
            delta_Z[: self.dim_x] = Xs_npo - Xp_npo
            delta_Z[self.dim_x :] = ikp1
            Xs_n: np.ndarray = Xf_n + Gn @ delta_Z

            # Smoothed covariance
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
                Gk_smooth=Gn.copy(),
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
                    self.history[i], title=f"EPKS smoothed step {cur['k']}"
                )

        logger.info(
            "NonLinear_EPKS backward pass complete (N_records=%d).", N_records,
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
            raise FilterError("Unexpected runtime error in EPKS process_smoother.") from e
