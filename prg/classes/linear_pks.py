"""
####################################################################
Linear Pairwise Kalman Smoother (PKS) — RTS-pairwise backward pass
####################################################################
"""

from __future__ import annotations

import logging
from collections.abc import Iterator

import numpy as np
from scipy.linalg import LinAlgError, cho_factor, cho_solve

from prg.classes.linear_pkf import Linear_PKF
from prg.classes.param_linear import ParamLinear
from prg.classes.param_nonlinear import ParamNonLinear
from prg.utils.display import rich_show_fields
from prg.utils.exceptions import CovarianceError, FilterError

# NOTE: ParamError, InvertibilityError, NumericalError and StepValidationError
# may propagate from the inherited forward pass (cf. Linear_PKF.process_filter);
# they are listed in the Raises docstrings but not imported here as they are
# only re-raised, never constructed in this module.

logger = logging.getLogger(__name__)

__all__ = ["Linear_PKS"]


class Linear_PKS(Linear_PKF):
    """
    Linear Pairwise Kalman Smoother (PKS).

    Two-pass smoother for the linear pairwise state-space model

    .. math::

        Z_{n+1} = A\\, Z_n + B\\, W_{n+1}, \\quad
        Z_n = (X_n^T,\\, Y_n^T)^T.

    The forward pass is the standard linear PKF (inherited from
    :class:`Linear_PKF`). The backward pass performs a Rauch-Tung-Striebel
    recursion at the **joint** ``Z = (X, Y)`` level: the pairwise model is
    Markov in ``Z``, not in ``X`` alone (because ``Y_{n+1}`` can depend
    directly on ``X_n`` via the bottom-left block of ``A``). The smoothing
    gain therefore couples ``X_n`` to **both** the smoothed ``X_{n+1}``
    and the next-step innovation ``y_{n+1} - yhat_{n+1|n}``.

    Two covariance update forms are available, selected at construction
    time via the ``joseph`` flag:

    - ``joseph=False`` (default) — standard form
      :math:`P^{xx}_{n|N} = P^{xx}_{n|n} + G_n (P^{ZZ}_{n+1|N} - P_{n+1|n}) G_n^T`,
    - ``joseph=True`` — Joseph form
      :math:`P^{xx}_{n|N} = (I_p, -G_n) \\Omega_n (I_p, -G_n)^T
                          + G_n^x\\, P^{xx}_{n+1|N}\\, (G_n^x)^T`,
      with :math:`\\Omega_n` the joint covariance of
      :math:`(X_n, Z_{n+1})` conditional on :math:`y_{1:n}`
      (shape :math:`(p + p + q) \\times (p + p + q)`).

    The two forms are mathematically equivalent at the optimal gain and
    agree empirically to ``~1e-10`` in double precision on the test
    fixtures (test tolerance enforced by
    :class:`TestLinearPKSJosephForm.JOSEPH_EQ_TOL`). The Joseph variant
    becomes valuable for the nonlinear extensions (EPKF / UPKF) where
    matrices become less well-conditioned.

    Parameters
    ----------
    param : ParamLinear | ParamNonLinear
        Forwarded to :class:`Linear_PKF`. Although the class is named
        "linear", the parent accepts non-linear param objects too — only
        the constant ``A`` matrix from the param is used.
    sKey : int, optional
        Random seed for reproducibility (default ``None``).
    verbose : int, optional
        Verbosity level (0, 1, 2; default 0). ``verbose > 1`` displays
        each smoothed history record via :func:`rich_show_fields`.
    joseph : bool, optional
        If ``True``, use the Joseph form of the covariance update
        (explicitly symmetric / PSD-preserving for any gain). Default
        ``False`` (standard RTS form).

    History schema additions
    ------------------------
    Each history record (initially populated by the forward filter from a
    :class:`prg.classes.pkf.PKFStep` dataclass) is augmented with three
    keys by the backward pass:

    - ``Xkp1_smooth`` : ``(dim_x, 1)`` smoothed mean ``E[X_n | y_{1:N}]``.
    - ``PXXkp1_smooth`` : ``(dim_x, dim_x)`` smoothed covariance.
    - ``Gk_smooth`` : ``(dim_x, dim_xy)`` smoothing gain :math:`G_n`. At
      the terminal step ``n = N`` the gain is undefined (no future to
      condition on); a zero matrix of the correct shape is stored as
      placeholder.

    All three fields are written via the public
    :meth:`HistoryTracker.update_record` API.
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
        Run the linear PKS as a generator.

        First exhausts the forward filter (which populates :attr:`history`),
        then performs the backward RTS recursion in place on the history
        records via :meth:`HistoryTracker.update_record`, and finally
        yields the augmented tuples in chronological order.

        Parameters
        ----------
        N : int, optional
            Maximum number of forward steps. ``None`` runs until the data
            generator is exhausted; the actual number of yielded tuples
            equals the number of forward records.
        data_generator : Iterator, optional
            External data generator yielding ``(k, x_true, y_obs)``. If
            ``None`` (default), the internal simulator generator is used.

        Yields
        ------
        k, x_true, y_observed, X_predict, X_update, X_smooth
            ``X_smooth`` is the posterior mean
            :math:`E[X_n | y_{1:N}]`.

        Raises
        ------
        ParamError
            If ``N`` is not a strictly positive integer or ``None``
            (propagated from the forward pass via ``_validate_N``).
        InvertibilityError
            If the forward-pass innovation covariance ``Skp1`` is not
            invertible (propagated from :meth:`Linear_PKF.process_filter`).
        CovarianceError
            If (a) any covariance matrix in the forward pass is invalid,
            (b) the Cholesky factorisation of :math:`P^{ZZ}_{n+1|n}` fails
            during the backward pass, or (c) the resulting smoothed
            covariance :math:`P^{xx}_{n|N}` is not PSD. In cases (b) and
            (c) the exception carries ``step`` and ``matrix_name``
            attributes set to the offending location.
        StepValidationError
            If the forward pass cannot construct a valid ``PKFStep``
            (propagated).
        NumericalError
            Base class of ``CovarianceError`` / ``InvertibilityError`` —
            catch this to intercept any matrix-level failure regardless
            of subtype.
        FilterError
            If the forward pass yielded no records (defensive guard) or
            if the forward pass raises an unexpected ``FilterError``
            (propagated).
        """
        # 1) Forward pass — drains the filter into self.history
        for _ in self.process_filter(N=N, data_generator=data_generator):
            pass

        N_records: int = len(self.history)
        if N_records == 0:
            raise FilterError("Linear_PKS: forward pass yielded no records.")

        logger.info(
            "Linear_PKS backward pass starting (N_records=%d, joseph=%s).",
            N_records,
            self.joseph,
        )

        # 2) Backward pass — initialise from the terminal step.
        # At n = N the gain is undefined; zeros of the right shape are
        # stored as a placeholder (see docstring).
        last = self.history[N_records - 1]
        self.history.update_record(
            N_records - 1,
            Xkp1_smooth=last["Xkp1_update"].copy(),
            PXXkp1_smooth=last["PXXkp1_update"].copy(),
            Gk_smooth=np.zeros((self.dim_x, self.dim_xy)),
        )

        # Pre-allocated scratch buffers — only the writable blocks are
        # touched in the hot loop; the surrounding zero blocks are part of
        # the structural invariants of P_aug and P_zz_smooth and must NOT
        # be overwritten elsewhere.
        P_aug: np.ndarray = self.zeros_dim_xy_xy.copy()
        P_zz_smooth: np.ndarray = self.zeros_dim_xy_xy.copy()
        delta_Z: np.ndarray = self.zeros_dim_xy_1.copy()

        if self.joseph:
            # Joint covariance Omega_n = Cov((X_n, Z_{n+1}) | y_{1:n})
            dim_jnt: int = self.dim_x + self.dim_xy
            Omega = np.zeros((dim_jnt, dim_jnt))
            J = np.zeros((self.dim_x, dim_jnt))
            J[: self.dim_x, : self.dim_x] = self.eye_dim_x  # I_p block (fixed)

        for i in range(N_records - 2, -1, -1):
            cur = self.history[i]      # time step n
            nxt = self.history[i + 1]  # time step n+1

            Xf_n: np.ndarray = cur["Xkp1_update"]
            Pf_n: np.ndarray = cur["PXXkp1_update"]

            Xp_npo: np.ndarray = nxt["Xkp1_predict"]
            ikp1: np.ndarray = nxt["ikp1"]
            Xs_npo: np.ndarray = nxt["Xkp1_smooth"]
            Ps_npo: np.ndarray = nxt["PXXkp1_smooth"]

            # Rebuild the full Z-level predicted covariance at step n+1:
            #     P_{n+1|n} = A diag(P^{xx}_{n|n}, 0) A^T + B Q B^T.
            P_aug[: self.dim_x, : self.dim_x] = Pf_n
            P_zz_npo: np.ndarray = self._A @ P_aug @ self._AT + self._BmQBT

            # Cross-covariance (dim_x rows, dim_xy cols):
            #     Cov(X_n, Z_{n+1} | y_{1:n}) = [P^{xx}_{n|n}, 0] A^T = Pf_n M^T.
            cross_X: np.ndarray = (P_aug @ self._AT)[: self.dim_x, :]

            # Gain via Cholesky solve (numerically stabler than explicit inverse).
            # The cho_factor try/except subsumes any ill-conditioning the
            # generic invertibility check would catch, so we rely solely on it.
            try:
                c, low = cho_factor(P_zz_npo)
                Gn: np.ndarray = cho_solve((c, low), cross_X.T).T
            except (LinAlgError, ValueError) as e:
                raise CovarianceError(
                    f"Step {cur['k']}: Cholesky factorisation failed for "
                    f"PZZkp1_predict in backward pass.",
                    matrix_name="PZZkp1_predict",
                    step=cur["k"],
                ) from e

            # ── Smoothed mean (same formula for both covariance variants) ──
            # delta_Z = ((X_{n+1|N} - X_{n+1|n}); innovation y_{n+1} - yhat_{n+1|n})
            delta_Z[: self.dim_x] = Xs_npo - Xp_npo
            delta_Z[self.dim_x :] = ikp1
            Xs_n: np.ndarray = Xf_n + Gn @ delta_Z

            # ── Smoothed covariance ────────────────────────────────────────
            if self.joseph:
                Omega[: self.dim_x, : self.dim_x] = Pf_n
                Omega[: self.dim_x, self.dim_x :] = cross_X
                Omega[self.dim_x :, : self.dim_x] = cross_X.T
                Omega[self.dim_x :, self.dim_x :] = P_zz_npo
                J[:, self.dim_x :] = -Gn          # J = (I_p, -G_n)
                Gn_x: np.ndarray = Gn[:, : self.dim_x]
                Ps_n: np.ndarray = J @ Omega @ J.T + Gn_x @ Ps_npo @ Gn_x.T
            else:
                # Standard form: Pf + G (P^ZZ_smooth - P^ZZ_pred) G^T
                P_zz_smooth[: self.dim_x, : self.dim_x] = Ps_npo
                Delta_P_ZZ: np.ndarray = P_zz_smooth - P_zz_npo
                Ps_n = Pf_n + Gn @ Delta_P_ZZ @ Gn.T

            # Floating-point symmetry filter (independent of standard/Joseph)
            Ps_n = 0.5 * (Ps_n + Ps_n.T)

            self._check_covariance(Ps_n, cur["k"], name="PXXkp1_smooth")

            self.history.update_record(
                i,
                Xkp1_smooth=Xs_n,
                PXXkp1_smooth=Ps_n,
                Gk_smooth=Gn,
            )

            # Per-step DEBUG trace — gated to avoid formatting cost when off
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(
                    "Step %d: |Gn|_F=%.3g, tr(P_smooth)=%.3g, tr(P_filt)=%.3g.",
                    cur["k"],
                    float(np.linalg.norm(Gn)),
                    float(np.trace(Ps_n)),
                    float(np.trace(Pf_n)),
                )

            # Rich display of the smoothed step, mirroring the forward pass
            if self.verbose > 1:
                rich_show_fields(
                    self.history[i], title=f"Smoothed step {cur['k']}"
                )

        logger.info(
            "Linear_PKS backward pass complete (N_records=%d).", N_records,
        )

        # 3) Yield records in chronological order, including the smoother fields
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

        Mirror of :meth:`PKF.process_N_data` for the smoothing pass.

        Exception-handling policy
        -------------------------
        All domain-specific exceptions raised by :meth:`process_smoother`
        — ``ParamError``, ``InvertibilityError``, ``CovarianceError``,
        ``NumericalError``, ``StepValidationError`` and direct
        ``FilterError`` — propagate up **unwrapped**. The structured
        ``step`` / ``matrix_name`` attributes are preserved.

        Only an opaque ``RuntimeError`` raised by Python's generator
        machinery (e.g. a non-domain ``StopIteration`` re-raised by
        PEP 479) is wrapped as ``FilterError`` and chained via
        ``from`` for traceability.

        Raises
        ------
        ParamError, InvertibilityError, CovarianceError, NumericalError, StepValidationError
            Propagated unwrapped from :meth:`process_smoother`.
        FilterError
            Either propagated from :meth:`process_smoother` (forward pass
            empty or unexpected error), or wrapping an opaque
            ``RuntimeError`` from the generator machinery.
        """
        try:
            return list(self.process_smoother(N=N, data_generator=data_generator))
        except RuntimeError as e:
            raise FilterError("Unexpected runtime error in process_smoother.") from e
