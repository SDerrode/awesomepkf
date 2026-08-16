"""
####################################################################
Linear Pairwise Kalman Smoother (PKS)
####################################################################

Five pairwise linear smoothers share the **same** forward Pairwise Kalman
Filter (PKF) and differ only by their *smoothing pass*: RTS, BF, MBF, MF and
DWY (cf. Geng et al., 2023). This module provides the shared infrastructure
(``_LinearPKSBase``) and the first variant (RTS, ``Linear_PKS_RTS``), plus a
backward-compatible façade ``Linear_PKS`` selecting a variant via ``method=``.

Adding a new variant is a two-step operation:

1. write a free function ``_<name>_pass(s, N_records)`` operating in place on
   ``s.history`` (``s`` being a ``_LinearPKSBase`` instance), then
2. register it in ``_SMOOTHING_PASSES`` and expose a thin ``Linear_PKS_<NAME>``
   subclass.

Every pass must write ``Xkp1_smooth`` and ``PXXkp1_smooth`` to each history
record (the public contract consumed by the yield/emit step and downstream
scripts); it may add variant-specific fields (e.g. ``Gk_smooth`` for RTS).
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

__all__ = [
    "Linear_PKS",
    "Linear_PKS_BF",
    "Linear_PKS_DWY",
    "Linear_PKS_MBF",
    "Linear_PKS_MF",
    "Linear_PKS_RTS",
    "Linear_PKS_VAR",
]


# ======================================================================
# Shared infrastructure
# ======================================================================
class _LinearPKSBase(Linear_PKF):
    """
    Shared base for the linear pairwise smoothers.

    Factorises everything common to the five variants:

    - the forward drain (running :meth:`Linear_PKF.process_filter` into
      :attr:`history`),
    - the public smoother API (:meth:`process_smoother`,
      :meth:`process_N_data_smoother`) and the chronological emit step,
    - the ``joseph`` flag and the parent forward-pass wiring.

    The only variant-specific part is the abstract hook
    :meth:`_smoothing_pass`, which each variant overrides to fill the
    ``Xkp1_smooth`` / ``PXXkp1_smooth`` fields of every history record.

    Parameters
    ----------
    param : ParamLinear | ParamNonLinear
        Forwarded to :class:`Linear_PKF` (only the constant ``A`` matrix is
        used; non-linear param objects are accepted).
    sKey : int, optional
        Random seed for reproducibility (default ``None``).
    verbose : int, optional
        Verbosity level (0, 1, 2; default 0). ``verbose > 1`` displays each
        smoothed record via :func:`rich_show_fields`.
    joseph : bool, optional
        If ``True``, variants that support it use the Joseph (PSD-preserving)
        covariance update. Default ``False``.
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
    # Public smoother API
    # ------------------------------------------------------------------
    def process_smoother(
        self,
        N: int | None = None,
        data_generator: Iterator[tuple[int, np.ndarray, np.ndarray]] | None = None,
        u: np.ndarray | None = None,
    ) -> Iterator[
        tuple[int, np.ndarray | None, np.ndarray, np.ndarray, np.ndarray, np.ndarray]
    ]:
        """
        Run the linear PKS as a generator.

        First exhausts the forward filter (which populates :attr:`history`),
        then runs the variant-specific :meth:`_smoothing_pass` in place on the
        history records, and finally yields the augmented tuples in
        chronological order.

        Optional deterministic control
        ------------------------------
        When ``u`` is given (and ``param.G`` is set), the couple obeys
        ``Z_{n+1} = A Z_n + G u_n + B W_n``. A deterministic control shifts only
        **means**, never covariances, so it is handled by *superposition*
        (exact for the linear-Gaussian model) without touching any of the six
        backward passes: build the nominal trajectory ``μ`` (``μ[0]=0``,
        ``μ[n+1] = A μ[n] + G u[n]``), run the control-free smoother on the
        **shifted** observations ``Ỹ_n = Y_n − μ[n][dim_x:]``, then add ``μ``
        back to the means (covariances unchanged).

        Parameters
        ----------
        N : int, optional
            Maximum number of forward steps. ``None`` runs until the data
            generator is exhausted.
        data_generator : Iterator, optional
            External data generator yielding ``(k, x_true, y_obs)``. If
            ``None`` (default), the internal simulator generator is used.
        u : np.ndarray, optional
            Deterministic control sequence, shape ``(R, dim_u)`` or
            ``(R, dim_u, 1)``. ``None`` (default) or ``param.G is None`` ⇒
            behaviour is **exactly** unchanged.

        Yields
        ------
        k, x_true, y_observed, X_predict, X_update, X_smooth
            ``X_smooth`` is the posterior mean :math:`E[X_n | y_{1:N}]`.

        Raises
        ------
        ParamError
            If ``N`` is not a strictly positive integer or ``None``
            (propagated from the forward pass via ``_validate_N``).
        InvertibilityError
            If the forward-pass innovation covariance ``Skp1`` is not
            invertible (propagated from :meth:`Linear_PKF.process_filter`).
        CovarianceError
            If a covariance matrix is invalid, a Cholesky factorisation fails,
            or a smoothed covariance is not PSD (carries ``step`` /
            ``matrix_name`` attributes where applicable).
        StepValidationError
            If the forward pass cannot construct a valid ``PKFStep``.
        NumericalError
            Base class of ``CovarianceError`` / ``InvertibilityError``.
        FilterError
            If the forward pass yielded no records, or on an unexpected
            ``FilterError`` (propagated).
        """
        # ── Control-driven path: mean-trajectory compensation ────────────────
        u_norm = self._normalise_control(u)
        if u_norm is not None:
            mu = self._build_control_trajectory(N, data_generator, u_norm)
            self._ctrl_mu = mu
            shifted_gen: Iterator | None = iter(self._shift_records())
        else:
            mu = None
            shifted_gen = data_generator

        # 1) Forward pass — drains the filter into self.history
        for _ in self.process_filter(N=N, data_generator=shifted_gen):
            pass

        N_records: int = len(self.history)
        if N_records == 0:
            raise FilterError("Linear_PKS: forward pass yielded no records.")

        # Missing observations (all-NaN y): only the joint-level RTS pass
        # handles them; the other variants consume the innovation fields
        # (None on a gap step) or treat every recorded y as observed data.
        gap_steps = [rec["k"] for rec in self.history if rec["ikp1"] is None]
        if gap_steps and not self._supports_gaps():
            shown = ", ".join(str(k) for k in gap_steps[:8])
            more = ", ..." if len(gap_steps) > 8 else ""
            raise FilterError(
                f"Linear_PKS: this smoother variant does not support missing "
                f"observations (all-NaN y) — {len(gap_steps)} gap step(s) at "
                f"k = {shown}{more}. Use the RTS pass (method='RTS')."
            )

        logger.info(
            "Linear_PKS smoothing pass starting (N_records=%d, method=%s, joseph=%s).",
            N_records,
            getattr(self, "method", self.__class__.__name__),
            self.joseph,
        )

        # 2) Variant-specific smoothing pass (writes *_smooth into history)
        self._smoothing_pass(N_records)

        logger.info("Linear_PKS smoothing pass complete (N_records=%d).", N_records)

        # 2b) Un-shift the history: add μ back to the means (covariances stay).
        if mu is not None:
            self._unshift_history(mu)

        # 3) Yield records in chronological order, including the smoother fields
        yield from self._emit()

    def process_N_data_smoother(
        self,
        N: int | None,
        data_generator: Iterator | None = None,
        u: np.ndarray | None = None,
    ) -> list[
        tuple[int, np.ndarray | None, np.ndarray, np.ndarray, np.ndarray, np.ndarray]
    ]:
        """
        Eager version of :meth:`process_smoother` — runs ``N`` steps and
        returns all outputs as a list (mirror of :meth:`PKF.process_N_data`).

        Domain-specific exceptions propagate **unwrapped** (preserving their
        ``step`` / ``matrix_name`` attributes). Only an opaque ``RuntimeError``
        from the generator machinery is wrapped as ``FilterError``.

        Parameters
        ----------
        N, data_generator, u
            See :meth:`process_smoother`.

        Raises
        ------
        ParamError, InvertibilityError, CovarianceError, NumericalError, StepValidationError
            Propagated unwrapped from :meth:`process_smoother`.
        FilterError
            Propagated, or wrapping an opaque generator ``RuntimeError``.
        """
        try:
            return list(
                self.process_smoother(N=N, data_generator=data_generator, u=u)
            )
        except RuntimeError as e:
            raise FilterError("Unexpected runtime error in process_smoother.") from e

    # ------------------------------------------------------------------
    # Control superposition helpers (linear-Gaussian, exact)
    # ------------------------------------------------------------------
    def _build_control_trajectory(
        self,
        N: int | None,
        data_generator: Iterator | None,
        u_norm: np.ndarray,
    ) -> np.ndarray:
        """Materialise the data and build the nominal control trajectory ``μ``.

        Stores the raw records on ``self._ctrl_records`` (a list of
        ``(k, x_true, y_obs)`` triples) for :meth:`_shift_records`, and returns
        ``μ`` of shape ``(R, dim_xy, 1)`` with ``μ[0]=0`` and
        ``μ[n+1] = A μ[n] + G u[n]`` (so the prior ``mz0`` is left unchanged).
        """
        if data_generator is not None:
            records = list(data_generator)
            if N is not None:
                records = records[: N + 1]
        else:
            # No external data: simulate the control-driven data natively first.
            records = self.simulate_N_data(N, u=u_norm)

        R = len(records)
        G = self.param.G
        dim_u = G.shape[1]
        mu = np.zeros((R, self.dim_xy, 1))
        for n in range(R - 1):
            un = u_norm[n] if n < u_norm.shape[0] else np.zeros((dim_u, 1))
            mu[n + 1] = self._A @ mu[n] + G @ un

        self._ctrl_records = records
        return mu

    def _shift_records(self) -> list[tuple[int, np.ndarray | None, np.ndarray]]:
        """Build the shifted data ``Ỹ_n = Y_n − μ[n][dim_x:]`` for the filter.

        The X-truth column is passed through unchanged (it is ground truth, not
        an estimate); only the observation ``Y`` is shifted by the Y-block of
        ``μ`` so the control-free smoother sees a centred sequence.
        """
        mu = self._ctrl_mu
        dx = self.dim_x
        shifted: list[tuple[int, np.ndarray | None, np.ndarray]] = []
        for n, (k, x_true, y_obs) in enumerate(self._ctrl_records):
            y_arr = np.asarray(y_obs, dtype=float).reshape(self.dim_y, 1)
            shifted.append((k, x_true, y_arr - mu[n][dx:]))
        return shifted

    def _unshift_history(self, mu: np.ndarray) -> None:
        """Add ``μ`` back to the X means and restore the original observations.

        Covariances and innovations are untouched (a deterministic control
        shifts means only).
        """
        dx = self.dim_x
        for n in range(len(self.history)):
            rec = self.history[n]
            mu_x = mu[n][:dx]
            mu_y = mu[n][dx:]
            self.history.update_record(
                n,
                Xkp1_predict=rec["Xkp1_predict"] + mu_x,
                Xkp1_update=rec["Xkp1_update"] + mu_x,
                Xkp1_smooth=rec["Xkp1_smooth"] + mu_x,
                ykp1=rec["ykp1"] + mu_y,
            )

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _emit(
        self,
    ) -> Iterator[
        tuple[int, np.ndarray | None, np.ndarray, np.ndarray, np.ndarray, np.ndarray]
    ]:
        """Yield the augmented history records in chronological order."""
        for entry in self.history:
            yield (
                entry["k"],
                entry["xkp1"],
                entry["ykp1"],
                entry["Xkp1_predict"],
                entry["Xkp1_update"],
                entry["Xkp1_smooth"],
            )

    def _smoothing_pass(self, N_records: int) -> None:
        """Variant hook — fill ``Xkp1_smooth`` / ``PXXkp1_smooth`` in history."""
        raise NotImplementedError(
            "A smoother variant must implement _smoothing_pass()."
        )

    def _supports_gaps(self) -> bool:
        """Whether this smoother variant handles missing observations.

        Only the joint-level RTS pass does; the other variants consume the
        innovation fields (``None`` on gap steps) or treat every recorded
        ``y`` as observed data.
        """
        return False


# ======================================================================
# Smoothing passes (one free function per variant)
# ======================================================================
def _joint_posterior(
    s: _LinearPKSBase, rec: dict
) -> tuple[np.ndarray, np.ndarray]:
    """
    Joint posterior ``(z, P)`` of the couple at a recorded forward step.

    Observed step: ``[Xkp1_update; y]`` with covariance
    ``blkdiag(PXXkp1_update, 0)`` — conditioning on the exactly observed ``Y``
    zeroes its covariance. Missing-observation step: the full joint recorded
    by the forward pass (``Zkp1_joint`` / ``PZZkp1_joint``), which is NOT
    block-diagonal.
    """
    if rec.get("PZZkp1_joint") is not None:
        return rec["Zkp1_joint"].copy(), rec["PZZkp1_joint"].copy()
    if rec["ikp1"] is None and rec["k"] > 0:
        raise FilterError(
            f"Step {rec['k']}: gap record without a stored joint posterior "
            f"(history predates missing-observation support?)."
        )
    z = np.vstack((rec["Xkp1_update"], rec["ykp1"]))
    P = np.zeros((s.dim_xy, s.dim_xy))
    P[: s.dim_x, : s.dim_x] = rec["PXXkp1_update"]
    return z, P


def _joint_predicted_mean(s: _LinearPKSBase, rec: dict) -> np.ndarray:
    """
    Predicted joint mean ``[X_pred; Y_pred]`` at a recorded step.

    Observed step: ``Y_pred = y - innovation``. Missing-observation step: the
    recorded joint posterior IS the predicted joint (no update happened).
    """
    if rec.get("PZZkp1_joint") is not None:
        return rec["Zkp1_joint"].copy()
    return np.vstack((rec["Xkp1_predict"], rec["ykp1"] - rec["ikp1"]))


def _rts_backward_pass(s: _LinearPKSBase, N_records: int) -> None:
    """
    Rauch--Tung--Striebel pairwise backward pass, in place on ``s.history``.

    Backward RTS recursion carried at the **joint** ``Z = (X, Y)`` level,
    which handles missing observations exactly. With the joint posterior
    ``(z_n, P_n)`` of each forward step (see :func:`_joint_posterior`), the
    recursion is the textbook RTS on ``Z``:

    .. math::
        P_{n+1|n} &= A P_n A^T + B Q B^T \\\\
        G_n       &= P_n A^T P_{n+1|n}^{-1} \\\\
        z_{n|N}   &= z_n + G_n (z_{n+1|N} - z_{n+1|n}) \\\\
        P_{n|N}   &= P_n + G_n (P_{n+1|N} - P_{n+1|n}) G_n^T

    On an observed step ``P_n = blkdiag(P^{xx}_{n|n}, 0)``, so the ``Y`` rows
    of ``G_n`` vanish and the recursion reduces exactly to the historical
    X-marginal form (the smoothed ``Y_n`` stays the observed ``y_n``). On a
    gap step ``P_n`` is the full predicted joint, and the nonzero Y/cross
    blocks propagate the future information into the missing ``Y_n`` as well.

    Two covariance update forms (selected by ``s.joseph``): standard (above)
    or Joseph ``(I, -G_n) \\Omega_n (I, -G_n)^T + G_n P^{ZZ}_{n+1|N} G_n^T``
    with ``\\Omega_n = [[P_n, P_n A^T], [A P_n, P_{n+1|n}]]`` — algebraically
    identical, PSD-preserving.

    Writes ``Xkp1_smooth``, ``PXXkp1_smooth`` and ``Gk_smooth`` (the ``X``
    rows of the joint gain, shape ``dim_x x dim_xy``, unchanged from the
    historical convention) to each record.

    Parameters
    ----------
    s : _LinearPKSBase
        Smoother instance providing ``_A``, ``_AT``, ``_BmQBT``, ``dim_x``,
        ``dim_xy``, ``joseph``, ``history``.
    N_records : int
        Number of forward records (history length).

    Raises
    ------
    CovarianceError
        If the Cholesky factorisation of ``P^{ZZ}_{n+1|n}`` fails, or the
        smoothed covariance is not PSD (carries ``step`` / ``matrix_name``).
    """
    # Backward pass — initialise from the terminal step. At n = N the gain is
    # undefined (no future to condition on); a zero placeholder is stored.
    last = s.history[N_records - 1]
    zs_npo, Ps_npo = _joint_posterior(s, last)  # smoothed == filtered at n=N
    s.history.update_record(
        N_records - 1,
        Xkp1_smooth=last["Xkp1_update"].copy(),
        PXXkp1_smooth=last["PXXkp1_update"].copy(),
        Gk_smooth=np.zeros((s.dim_x, s.dim_xy)),
    )

    d: int = s.dim_xy

    if s.joseph:
        Omega = np.zeros((2 * d, 2 * d))
        J = np.zeros((d, 2 * d))
        J[:, :d] = np.eye(d)  # (I, -G_n) — identity block fixed

    for i in range(N_records - 2, -1, -1):
        cur = s.history[i]      # time step n
        nxt = s.history[i + 1]  # time step n+1

        z_n, P_n = _joint_posterior(s, cur)
        zp_npo: np.ndarray = _joint_predicted_mean(s, nxt)

        # Cov(Z_n, Z_{n+1} | data so far) = P_n A^T, and the joint prior
        Cn: np.ndarray = P_n @ s._AT
        P_zz_npo: np.ndarray = s._A @ Cn + s._BmQBT

        # Gain via Cholesky solve (numerically stabler than explicit inverse).
        try:
            c, low = cho_factor(P_zz_npo)
            Gn: np.ndarray = cho_solve((c, low), Cn.T).T  # (dim_xy, dim_xy)
        except (LinAlgError, ValueError) as e:
            raise CovarianceError(
                f"Step {cur['k']}: Cholesky factorisation failed for "
                f"PZZkp1_predict in backward pass.",
                matrix_name="PZZkp1_predict",
                step=cur["k"],
            ) from e

        # ── Smoothed joint mean (same formula for both covariance variants) ──
        zs_n: np.ndarray = z_n + Gn @ (zs_npo - zp_npo)

        # ── Smoothed joint covariance ──
        if s.joseph:
            Omega[:d, :d] = P_n
            Omega[:d, d:] = Cn
            Omega[d:, :d] = Cn.T
            Omega[d:, d:] = P_zz_npo
            J[:, d:] = -Gn
            Ps_n: np.ndarray = J @ Omega @ J.T + Gn @ Ps_npo @ Gn.T
        else:
            Ps_n = P_n + Gn @ (Ps_npo - P_zz_npo) @ Gn.T

        # Floating-point symmetry filter (independent of standard/Joseph)
        Ps_n = 0.5 * (Ps_n + Ps_n.T)

        Ps_x: np.ndarray = Ps_n[: s.dim_x, : s.dim_x].copy()
        s._check_covariance(Ps_x, cur["k"], name="PXXkp1_smooth")

        s.history.update_record(
            i,
            Xkp1_smooth=zs_n[: s.dim_x].copy(),
            PXXkp1_smooth=Ps_x,
            Gk_smooth=Gn[: s.dim_x, :].copy(),
        )

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "Step %d: |Gn|_F=%.3g, tr(P_smooth)=%.3g, tr(P_filt)=%.3g.",
                cur["k"],
                float(np.linalg.norm(Gn)),
                float(np.trace(Ps_x)),
                float(np.trace(P_n[: s.dim_x, : s.dim_x])),
            )

        if s.verbose > 1:
            rich_show_fields(s.history[i], title=f"Smoothed step {cur['k']}")

        zs_npo, Ps_npo = zs_n, Ps_n


def _mbf_backward_pass(s: _LinearPKSBase, N_records: int) -> None:
    r"""
    Modified Bryson--Frazier (Bierman) pairwise smoothing pass, in place.

    Adjoint smoother at the **filtered** ``X`` level (dimension ``dim_x``). It
    propagates the adjoint pair :math:`(\lambda_n, \Lambda_n)` from which the
    smoothed law is recovered by

    .. math::
        \bar x_{n|N} = \bar x_{n|n} + P^{xx}_{n|n}\,\lambda_n,\quad
        P^{xx}_{n|N} = P^{xx}_{n|n} - P^{xx}_{n|n}\,\Lambda_n\,P^{xx}_{n|n}.

    Backward recursion (``n = N \to 1``), with the constant columns-X block
    :math:`M = (A^{xx}; A^{yx})`, the forward innovation ``ikp1``, its
    covariance ``Skp1`` and the filter gain ``Kkp1``:

    - measurement update (to the couple adjoint)
      :math:`\mu_n = (\lambda_n;\, S_n^{-1}\iota_n - K_n^\top\lambda_n)`,
      :math:`N_n = \mathrm{diag}(0, S_n^{-1}) + (I; -K_n^\top)\,\Lambda_n\,(I, -K_n)`;
    - time update :math:`\lambda_{n-1} = M^\top\mu_n`,
      :math:`\Lambda_{n-1} = M^\top N_n M`.

    The factored form of ``N_n`` keeps :math:`\Lambda_n \succeq 0` by
    construction (the rearward analogue of the Joseph form), so this variant is
    numerically robust and never forms the degenerate joint covariance. On the
    linear-Gaussian model it returns the RTS estimate (see ``test_mbf_equals_rts``).

    Writes ``Xkp1_smooth`` / ``PXXkp1_smooth`` to each record.

    Raises
    ------
    CovarianceError
        If the innovation covariance ``Skp1`` is not Cholesky-factorisable, or a
        smoothed covariance is not PSD (carries ``step`` / ``matrix_name``).
    """
    dx = s.dim_x
    M: np.ndarray = s._A[:, :dx]   # columns-X of A, shape (dim_xy, dim_x)
    MT: np.ndarray = M.T
    NN: int = N_records - 1

    # Terminal step: lambda_N = Lambda_N = 0 → smoothed = filtered.
    last = s.history[NN]
    s.history.update_record(
        NN,
        Xkp1_smooth=last["Xkp1_update"].copy(),
        PXXkp1_smooth=last["PXXkp1_update"].copy(),
    )
    lam: np.ndarray = np.zeros((dx, 1))
    Lam: np.ndarray = np.zeros((dx, dx))

    for n in range(NN, 0, -1):
        rec = s.history[n]
        Sn, Kn, inn = rec["Skp1"], rec["Kkp1"], rec["ikp1"]

        try:
            c, low = cho_factor(Sn)
            Sinv_inn: np.ndarray = cho_solve((c, low), inn)            # S^{-1} iota_n
            Sinv: np.ndarray = cho_solve((c, low), s.eye_dim_y)        # S^{-1}
        except (LinAlgError, ValueError) as e:
            raise CovarianceError(
                f"Step {rec['k']}: Cholesky factorisation failed for the "
                f"innovation covariance Skp1 in the MBF pass.",
                matrix_name="Skp1",
                step=rec["k"],
            ) from e

        # Measurement update → couple adjoint (mu_n, N_n), factored PSD form.
        F: np.ndarray = np.vstack((s.eye_dim_x, -Kn.T))    # [I_p; -K_n^T], (dz, dx)
        mu: np.ndarray = np.vstack((lam, Sinv_inn - Kn.T @ lam))
        Nmat: np.ndarray = F @ Lam @ F.T
        Nmat[dx:, dx:] += Sinv                              # + diag(0, S^{-1})

        # Time update n → n-1.
        lam = MT @ mu
        Lam = MT @ Nmat @ M

        # Recover the smoothed law at n-1 (filtered-level).
        prev = s.history[n - 1]
        Xf: np.ndarray = prev["Xkp1_update"]
        Pf: np.ndarray = prev["PXXkp1_update"]
        Xs: np.ndarray = Xf + Pf @ lam
        Ps: np.ndarray = Pf - Pf @ Lam @ Pf
        Ps = 0.5 * (Ps + Ps.T)
        s._check_covariance(Ps, prev["k"], name="PXXkp1_smooth")
        s.history.update_record(n - 1, Xkp1_smooth=Xs, PXXkp1_smooth=Ps)


def _bf_backward_pass(s: _LinearPKSBase, N_records: int) -> None:
    r"""
    Bryson--Frazier (pure) pairwise smoothing pass, in place.

    Adjoint smoother at the **couple** level: it propagates the predicted-couple
    adjoint :math:`(\mu_n, N_n) \in \mathbb{R}^{p+q}` and recovers the smoothed
    law from the *predicted* joint moments

    .. math::
        \bar z_{n|N} = \bar z_{n|n-1} + P_{n|n-1}\,\mu_n,\quad
        P^{ZZ}_{n|N} = P_{n|n-1} - P_{n|n-1}\,N_n\,P_{n|n-1},

    the smoothed ``X`` law being the ``x`` / ``xx`` block. Backward recursion
    (``n = N \to 1``) with :math:`\Psi_n = (I; -K_n^\top) M^\top`:

    .. math::
        \mu_n = (0; S_n^{-1}\iota_n) + \Psi_n \mu_{n+1},\quad
        N_n = \mathrm{diag}(0, S_n^{-1}) + \Psi_n N_{n+1} \Psi_n^\top.

    The terminal step ``n = 0`` (no prediction) is recovered at the filtered
    level via :math:`\lambda_0 = M^\top\mu_1`. Unlike MBF, the covariance update
    is a *subtraction* on the joint :math:`P_{n|n-1}`; it is the (less robust)
    standard counterpart, kept for completeness. On the linear-Gaussian model it
    returns the RTS estimate (see ``test_bf_equals_rts``).

    Writes ``Xkp1_smooth`` / ``PXXkp1_smooth`` to each record.

    Raises
    ------
    CovarianceError
        If the innovation covariance ``Skp1`` is not Cholesky-factorisable, or a
        smoothed covariance is not PSD (carries ``step`` / ``matrix_name``).
    """
    dx, dz = s.dim_x, s.dim_xy
    M: np.ndarray = s._A[:, :dx]
    MT: np.ndarray = M.T
    NN: int = N_records - 1

    # Couple adjoint, initialised flat (mu_{N+1} = 0, N_{N+1} = 0).
    mu: np.ndarray = np.zeros((dz, 1))
    Nmat: np.ndarray = np.zeros((dz, dz))

    P_pred: np.ndarray = s.zeros_dim_xy_xy.copy()

    for n in range(NN, 0, -1):
        rec = s.history[n]
        Sn, Kn, inn = rec["Skp1"], rec["Kkp1"], rec["ikp1"]

        try:
            c, low = cho_factor(Sn)
            Sinv_inn: np.ndarray = cho_solve((c, low), inn)
            Sinv: np.ndarray = cho_solve((c, low), s.eye_dim_y)
        except (LinAlgError, ValueError) as e:
            raise CovarianceError(
                f"Step {rec['k']}: Cholesky factorisation failed for the "
                f"innovation covariance Skp1 in the BF pass.",
                matrix_name="Skp1",
                step=rec["k"],
            ) from e

        # Couple-adjoint recursion: Psi_n = [I; -K_n^T] M^T.
        Psi: np.ndarray = np.vstack((s.eye_dim_x, -Kn.T)) @ MT
        mu = Psi @ mu
        mu[dx:] += Sinv_inn                       # + (0; S^{-1} iota_n)
        Nmat = Psi @ Nmat @ Psi.T
        Nmat[dx:, dx:] += Sinv                    # + diag(0, S^{-1})

        # Predicted-level recovery at n. Rebuild the predicted joint moments:
        #   z_pred = (X_pred; y - iota),  P_pred = [[PXX, K S], [(K S)^T, S]].
        Xp: np.ndarray = rec["Xkp1_predict"]
        z_pred: np.ndarray = np.vstack((Xp, rec["ykp1"] - inn))
        Pxy: np.ndarray = Kn @ Sn
        P_pred[:dx, :dx] = rec["PXXkp1_predict"]
        P_pred[:dx, dx:] = Pxy
        P_pred[dx:, :dx] = Pxy.T
        P_pred[dx:, dx:] = Sn

        z_s: np.ndarray = z_pred + P_pred @ mu
        Pzz_s: np.ndarray = P_pred - P_pred @ Nmat @ P_pred
        Ps: np.ndarray = 0.5 * (Pzz_s[:dx, :dx] + Pzz_s[:dx, :dx].T)
        s._check_covariance(Ps, rec["k"], name="PXXkp1_smooth")
        s.history.update_record(n, Xkp1_smooth=z_s[:dx], PXXkp1_smooth=Ps)

    # n = 0: no prediction → recover at the filtered level via lambda_0 = M^T mu_1.
    lam0: np.ndarray = MT @ mu
    Lam0: np.ndarray = MT @ Nmat @ M
    rec0 = s.history[0]
    Xf0, Pf0 = rec0["Xkp1_update"], rec0["PXXkp1_update"]
    Ps0: np.ndarray = Pf0 - Pf0 @ Lam0 @ Pf0
    Ps0 = 0.5 * (Ps0 + Ps0.T)
    s._check_covariance(Ps0, rec0["k"], name="PXXkp1_smooth")
    s.history.update_record(0, Xkp1_smooth=Xf0 + Pf0 @ lam0, PXXkp1_smooth=Ps0)


def _cond_xy(
    mean_z: np.ndarray, Pzz: np.ndarray, y: np.ndarray, dx: int
) -> tuple[np.ndarray, np.ndarray]:
    """Gaussian conditioning of ``X`` on ``Y = y`` from the joint ``(mean_z, Pzz)``.

    Returns ``(E[X | Y=y], Var[X | Y=y])``. Used by the DWY backward filter's
    measurement update and its terminal initialisation.
    """
    mx, my = mean_z[:dx], mean_z[dx:]
    Pxx = Pzz[:dx, :dx]
    Pyy, Pyx = Pzz[dx:, dx:], Pzz[dx:, :dx]
    K = np.linalg.solve(Pyy, Pyx).T          # K = Pxy Pyy^{-1}, shape (dx, dy)
    return mx + K @ (y - my), Pxx - K @ Pyx


def _dwy_backward_filter(s: _LinearPKSBase, N_records: int) -> dict:
    """
    Backward pairwise filter on the time-reversed (complementary) couple model.

    Shared infrastructure for the **DWY** smoother (which chains a forward
    recursion, :func:`_dwy_pass`) and the **MF** two-filter smoother (which
    fuses pointwise, :func:`_mf_twofilter_pass`). Both reuse the same reversed
    model and backward filter; they differ only by how the backward law is
    combined with the forward filter (cf. report Section 2.4 / 2.5).

    Mechanics (cf. report Section 2.5):

    - Lyapunov prior covariance ``Sigma_n`` and prior mean ``m_n``;
    - backward model ``Z_n = A^b_n Z_{n+1} + c^b_n + u^b_n`` with
      ``A^b_n = Sigma_n A^T Sigma_{n+1}^{-1}``,
      ``Q^b_n = Sigma_n - A^b_n Sigma_{n+1} (A^b_n)^T``, and a **non-zero-mean**
      forcing/offset ``c^b_n = m_n - A^b_n m_{n+1}`` (the reversed process is not
      centred — this offset affects means, not covariances);
    - backward filter (``N -> 0``): ``P^b_n``, ``x^b_n = E[X_n | y_{n:N}]``,
      plus the backward-predicted joint moments ``z^b_{n-1|n}``, ``P^b_{n-1|n}``.

    Returns
    -------
    dict
        Keys ``dx, dy, dz, NN, Sig, mz, Mb, ys, xb, Pb, Ppred, zpred``.
    """
    dx, dy, dz = s.dim_x, s.dim_y, s.dim_xy
    A, AT, Qp = s._A, s._AT, s._BmQBT
    NN = N_records - 1

    Sig0 = np.asarray(s.param.Pz0, dtype=float).reshape(dz, dz)
    mz0 = np.asarray(s.param.mz0, dtype=float).reshape(dz, 1)

    # --- Lyapunov prior (covariance + mean) ---
    Sig: list = [Sig0]
    mz: list = [mz0]
    for _ in range(NN):
        Sig.append(A @ Sig[-1] @ AT + Qp)
        mz.append(A @ mz[-1])

    # --- backward (complementary) model: A^b, Q^b, M^b, offset c^b ---
    Ab: list = [None] * NN
    Qb: list = [None] * NN
    Mb: list = [None] * NN
    cb: list = [None] * NN
    for n in range(NN):
        Abn = Sig[n] @ AT @ np.linalg.inv(Sig[n + 1])
        Ab[n] = Abn
        Qb[n] = Sig[n] - Abn @ Sig[n + 1] @ Abn.T
        Mb[n] = Abn[:, :dx]                       # X-columns block
        cb[n] = mz[n] - Abn @ mz[n + 1]           # non-zero-mean offset c^b_n

    ys = [
        np.asarray(s.history[n]["ykp1"], dtype=float).reshape(dy, 1)
        for n in range(N_records)
    ]

    # --- backward filter (N -> 0) ---
    Pb: list = [None] * (NN + 1)
    xb: list = [None] * (NN + 1)
    Ppred: list = [None] * (NN + 1)   # P^b_{n-1|n}  (stored at index n-1)
    zpred: list = [None] * (NN + 1)   # z^b_{n-1|n}
    xb[NN], Pb[NN] = _cond_xy(mz[NN], Sig[NN], ys[NN], dx)
    for n in range(NN, 0, -1):
        Pzz = Mb[n - 1] @ Pb[n] @ Mb[n - 1].T + Qb[n - 1]
        zp = Ab[n - 1] @ np.vstack([xb[n], ys[n]]) + cb[n - 1]
        Ppred[n - 1] = Pzz
        zpred[n - 1] = zp
        xb[n - 1], Pb[n - 1] = _cond_xy(zp, Pzz, ys[n - 1], dx)

    return {
        "dx": dx, "dy": dy, "dz": dz, "NN": NN,
        "Sig": Sig, "mz": mz, "Mb": Mb, "ys": ys,
        "xb": xb, "Pb": Pb, "Ppred": Ppred, "zpred": zpred,
    }


def _dwy_pass(s: _LinearPKSBase, N_records: int) -> None:
    """
    Desai--Weinert--Yusypchuk (backward-RTS) pairwise smoothing pass.

    Dual of the RTS pass: a **backward** pairwise filter on the time-reversed
    (complementary) couple model (:func:`_dwy_backward_filter`), followed by a
    **forward** recursion of gain ``D_n = P^b_n (M^b_{n-1})^T (P^b_{n-1|n})^{-1}``.
    On the linear-Gaussian model it returns the same smoothed estimate as RTS
    (Geng et al., 2023) — see the equivalence test ``test_dwy_equals_rts``.

    Writes ``Xkp1_smooth``, ``PXXkp1_smooth`` and the DWY gain ``Dk_smooth``.

    Raises
    ------
    CovarianceError
        If a backward-predicted covariance is not Cholesky-factorisable, or a
        smoothed covariance is not PSD (carries ``step`` / ``matrix_name``).
    """
    bf = _dwy_backward_filter(s, N_records)
    dx, dz, NN = bf["dx"], bf["dz"], bf["NN"]
    Mb, ys = bf["Mb"], bf["ys"]
    xb, Pb, Ppred, zpred = bf["xb"], bf["Pb"], bf["Ppred"], bf["zpred"]

    # --- forward recursion (0 -> N) ---
    Xs: list = [None] * (NN + 1)
    Ps: list = [None] * (NN + 1)
    Dn_list: list = [None] * (NN + 1)
    Dn_list[0] = np.zeros((dx, dz))               # n = 0: gain undefined
    Xs[0], Ps[0] = xb[0], Pb[0]
    blk = s.zeros_dim_xy_xy.copy()
    for n in range(1, NN + 1):
        try:
            c, low = cho_factor(Ppred[n - 1])
            Dn = cho_solve((c, low), (Pb[n] @ Mb[n - 1].T).T).T
        except (LinAlgError, ValueError) as e:
            raise CovarianceError(
                f"Step {s.history[n]['k']}: Cholesky factorisation failed for "
                f"the backward-predicted covariance in the DWY pass.",
                matrix_name="PZZpred_backward",
                step=s.history[n]["k"],
            ) from e
        blk[:dx, :dx] = Ps[n - 1]
        Ps[n] = Pb[n] + Dn @ (blk - Ppred[n - 1]) @ Dn.T
        resid = np.vstack(
            [Xs[n - 1] - zpred[n - 1][:dx], ys[n - 1] - zpred[n - 1][dx:]]
        )
        Xs[n] = xb[n] + Dn @ resid
        Dn_list[n] = Dn

    # --- write smoothed quantities to history ---
    for n in range(N_records):
        Ps_n = 0.5 * (Ps[n] + Ps[n].T)
        s._check_covariance(Ps_n, s.history[n]["k"], name="PXXkp1_smooth")
        s.history.update_record(
            n, Xkp1_smooth=Xs[n], PXXkp1_smooth=Ps_n, Dk_smooth=Dn_list[n]
        )


def _mf_twofilter_pass(s: _LinearPKSBase, N_records: int) -> None:
    r"""
    Mayne--Fraser (two-filter) pairwise smoothing pass, in place.

    Fuses, in **information** form, the forward filter posterior
    :math:`p(X_n | y_{1:n})` with the backward filter posterior
    :math:`p(X_n | y_{n:N})` (shared backward filter,
    :func:`_dwy_backward_filter`), removing the doubly-counted prior-conditioned
    term :math:`p(X_n | y_n)`:

    .. math::
        (P^{xx}_{n|N})^{-1} = (P^{xx}_{n|n})^{-1} + (P^b_n)^{-1} - (P^y_n)^{-1},

    and likewise for the information vector. The two passes are **independent**
    (parallelisable) — this is the structural difference with DWY, which chains
    them. On the linear-Gaussian model it returns the RTS estimate (Geng et al.,
    2023) — see ``test_mf_equals_rts``.

    Writes ``Xkp1_smooth`` / ``PXXkp1_smooth`` to each record.

    Raises
    ------
    CovarianceError
        If any of the three :math:`p\times p` covariances (or the fused
        information matrix) is not Cholesky-factorisable, or a smoothed
        covariance is not PSD (carries ``step`` / ``matrix_name``).
    """
    bf = _dwy_backward_filter(s, N_records)
    dx = bf["dx"]
    Sig, mz, ys, xb, Pb = bf["Sig"], bf["mz"], bf["ys"], bf["xb"], bf["Pb"]
    eye_dx = s.eye_dim_x

    def _inv_pd(P: np.ndarray, k: int, name: str) -> np.ndarray:
        try:
            c, low = cho_factor(P)
            return cho_solve((c, low), eye_dx)
        except (LinAlgError, ValueError) as e:
            raise CovarianceError(
                f"Step {k}: Cholesky factorisation failed for {name} "
                f"in the MF two-filter fusion.",
                matrix_name=name,
                step=k,
            ) from e

    for n in range(N_records):
        rec = s.history[n]
        Xf, Pf = rec["Xkp1_update"], rec["PXXkp1_update"]
        Xb, Pbn = xb[n], Pb[n]
        # p(X_n | y_n): prior marginal N(m_n, Sig_n) conditioned on Y_n = y_n.
        Xy, Py = _cond_xy(mz[n], Sig[n], ys[n], dx)

        If = _inv_pd(Pf, rec["k"], "PXXkp1_update")
        Ib = _inv_pd(Pbn, rec["k"], "Pb_backward")
        Iy = _inv_pd(Py, rec["k"], "Py_prior")

        info: np.ndarray = If + Ib - Iy                  # fused information
        Ps: np.ndarray = _inv_pd(info, rec["k"], "info_smooth")
        Xs: np.ndarray = Ps @ (If @ Xf + Ib @ Xb - Iy @ Xy)
        Ps = 0.5 * (Ps + Ps.T)
        s._check_covariance(Ps, rec["k"], name="PXXkp1_smooth")
        s.history.update_record(n, Xkp1_smooth=Xs, PXXkp1_smooth=Ps)


def _lifted_pass(s: _LinearPKSBase, N_records: int) -> None:
    r"""
    Variational (lifted) pairwise smoothing pass: one block-tridiagonal solve.

    Since $\mY_n=\vy_n$ is observed exactly, fixed-interval smoothing is a single
    quadratic program in the latent trajectory $\mathbf{x}=(\mX_0,\dots,\mX_N)$.
    With $\mathbf{E}=[\mI_p;\zero]$, the columns-$\mX$ block $\mM=\mA[:,:p]$ and the
    transition-residual covariance $\mathbf{R}=\mB\calQ\mB^\top$, the information
    matrix is block tridiagonal ($p\times p$ blocks),

        J_{nn}   = E^T R^{-1} E + M^T R^{-1} M     (interior; boundary terms drop),
        J_{n,n-1} = -E^T R^{-1} M  (constant),

    plus the prior $P_0^{-1}$ from the first estimate $p(\mX_0\mid\vy_0)$. The
    smoothed means solve $J\,\hat{\mathbf{x}}=\boldsymbol\eta$ by a block Thomas
    (forward--backward) sweep; the diagonal inverse blocks $\Pxx{\nnN}$ follow from
    the Takahashi backward recursion. On the linear-Gaussian model this returns the
    RTS estimate (``test_variant_equals_rts`` with ``method="VAR"``); $J\succ0$ iff $\mS_n\succ0\ \forall n$.

    Requires $\mathbf{R}=\mB\calQ\mB^\top\succ0$ (full-rank process noise).

    Writes ``Xkp1_smooth`` / ``PXXkp1_smooth`` to each record.

    Raises
    ------
    CovarianceError
        If $\mathbf{R}$, the prior $P_0$ or a pivot $\Delta_n$ is not
        Cholesky-factorisable (carries ``step`` / ``matrix_name``).
    """
    dx, dy, dz = s.dim_x, s.dim_y, s.dim_xy
    A = s._A
    M = A[:, :dx]                       # (dz, dx) columns-X block
    Axy = A[:dx, dx:]                   # (dx, dy)
    Ayy = A[dx:, dx:]                   # (dy, dy)
    R = s._BmQBT
    NN = N_records - 1
    E = np.zeros((dz, dx))
    E[:dx, :] = s.eye_dim_x

    eye_dx = s.eye_dim_x
    k0 = s.history[0]["k"]
    try:
        cR, lowR = cho_factor(R)
        Rinv = cho_solve((cR, lowR), np.eye(dz))
    except (LinAlgError, ValueError) as e:
        raise CovarianceError(
            "Variational pass: R = B Q B^T is not positive definite "
            "(required by the lifted form).",
            matrix_name="BmQBT",
            step=k0,
        ) from e

    EtRinvE = E.T @ Rinv @ E            # (dx, dx)
    MtRinvM = M.T @ Rinv @ M            # (dx, dx)
    Loff = -(E.T @ Rinv @ M)           # constant lower block J[n, n-1]

    mu0 = s.history[0]["Xkp1_update"]
    P0 = s.history[0]["PXXkp1_update"]
    try:
        cP0, lowP0 = cho_factor(P0)
        P0inv = cho_solve((cP0, lowP0), eye_dx)
    except (LinAlgError, ValueError) as e:
        raise CovarianceError(
            "Variational pass: initial P_0 is not positive definite.",
            matrix_name="PXXkp1_update",
            step=k0,
        ) from e

    ys = [np.asarray(s.history[n]["ykp1"], dtype=float).reshape(dy, 1)
          for n in range(N_records)]

    # --- assemble block-tridiagonal J (diagonal D, constant lower Loff) and eta ---
    D = [np.zeros((dx, dx)) for _ in range(NN + 1)]
    eta = [np.zeros((dx, 1)) for _ in range(NN + 1)]
    D[0] += P0inv
    eta[0] += P0inv @ mu0
    for n in range(1, NN + 1):
        cn = np.vstack([-Axy @ ys[n - 1], ys[n] - Ayy @ ys[n - 1]])   # (dz, 1)
        Rinv_cn = Rinv @ cn
        D[n] += EtRinvE
        D[n - 1] += MtRinvM
        eta[n] += -(E.T @ Rinv_cn)
        eta[n - 1] += M.T @ Rinv_cn

    # --- forward sweep: pivots Delta_n (chol) and modified RHS ctil_n ---
    chol: list = [None] * (NN + 1)
    ctil: list = [None] * (NN + 1)
    try:
        Delta = 0.5 * (D[0] + D[0].T)
        chol[0] = cho_factor(Delta)
        ctil[0] = eta[0]
        for n in range(1, NN + 1):
            T = Loff @ cho_solve(chol[n - 1], eye_dx)     # L_n Delta_{n-1}^{-1}
            Delta = D[n] - T @ Loff.T
            Delta = 0.5 * (Delta + Delta.T)
            chol[n] = cho_factor(Delta)
            ctil[n] = eta[n] - T @ ctil[n - 1]
    except (LinAlgError, ValueError) as e:
        raise CovarianceError(
            "Variational pass: a block-tridiagonal pivot is not positive "
            "definite (S_n may be singular).",
            matrix_name="Delta",
            step=k0,
        ) from e

    # --- backward sweep: mean x_n and diagonal inverse blocks P_{n|N} ---
    Xs: list = [None] * (NN + 1)
    Ps: list = [None] * (NN + 1)
    # Mk_smooth[n] = Cov(X_{n+1}, X_n | y_{0:N}); zero placeholder at the terminal n.
    Ms: list = [np.zeros((dx, dx)) for _ in range(NN + 1)]
    Xs[NN] = cho_solve(chol[NN], ctil[NN])
    Ps[NN] = cho_solve(chol[NN], eye_dx)
    for n in range(NN - 1, -1, -1):
        Dinv = cho_solve(chol[n], eye_dx)
        Xs[n] = cho_solve(chol[n], ctil[n] - Loff.T @ Xs[n + 1])
        W = Dinv @ Loff.T                              # Delta_n^{-1} J[n,n+1]
        Ps[n] = Dinv + W @ Ps[n + 1] @ W.T
        Ms[n] = -Ps[n + 1] @ W.T                       # (J^{-1})_{n+1,n}

    for n in range(N_records):
        Psn = 0.5 * (Ps[n] + Ps[n].T)
        s._check_covariance(Psn, s.history[n]["k"], name="PXXkp1_smooth")
        s.history.update_record(
            n, Xkp1_smooth=Xs[n], PXXkp1_smooth=Psn, Mk_smooth=Ms[n]
        )


# Registry of smoothing passes. All variants produce the same smoothed estimate
# on the linear-Gaussian model (Geng et al., 2023); VAR is the single
# block-tridiagonal (variational) solve that unifies the five recursions.
_SMOOTHING_PASSES = {
    "RTS": _rts_backward_pass,
    "BF": _bf_backward_pass,
    "MBF": _mbf_backward_pass,
    "MF": _mf_twofilter_pass,
    "2F": _mf_twofilter_pass,   # alias: "two-filter" (paper name); "MF" kept for back-compat
    "DWY": _dwy_pass,
    "VAR": _lifted_pass,
}


# ======================================================================
# Variant classes + façade
# ======================================================================
class Linear_PKS_RTS(_LinearPKSBase):
    """Linear pairwise Rauch--Tung--Striebel smoother (explicit variant).

    The backward pass runs at the joint ``Z = (X, Y)`` level, so it handles
    missing observations (all-NaN ``y``) exactly."""

    def _smoothing_pass(self, N_records: int) -> None:
        _rts_backward_pass(self, N_records)

    def _supports_gaps(self) -> bool:
        return True


class Linear_PKS_BF(_LinearPKSBase):
    """Linear pairwise Bryson--Frazier smoother (couple adjoint, predicted-level
    recovery). Equivalent to RTS on the linear-Gaussian model. See
    :func:`_bf_backward_pass`."""

    def _smoothing_pass(self, N_records: int) -> None:
        _bf_backward_pass(self, N_records)


class Linear_PKS_MBF(_LinearPKSBase):
    """Linear pairwise Modified Bryson--Frazier smoother (filtered ``X`` adjoint,
    PSD-preserving factored form). Equivalent to RTS on the linear-Gaussian
    model. See :func:`_mbf_backward_pass`."""

    def _smoothing_pass(self, N_records: int) -> None:
        _mbf_backward_pass(self, N_records)


class Linear_PKS_MF(_LinearPKSBase):
    """Linear pairwise two-filter smoother (independent forward and backward
    filters fused in information form). Also selectable via ``method='2F'`` on the
    :class:`Linear_PKS` facade -- ``'2F'`` ("two-filter") is the name used in the
    companion paper, ``'MF'`` (Mayne--Fraser) is kept for backward compatibility.
    The backward pass is the *posterior* (Bayesian) time-reversal of the couple
    chain, so it generalises the classical Mayne--Fraser likelihood two-filter
    while returning the same smoothed estimate. Equivalent to RTS on the
    linear-Gaussian model. See :func:`_mf_twofilter_pass`."""

    def _smoothing_pass(self, N_records: int) -> None:
        _mf_twofilter_pass(self, N_records)


class Linear_PKS_DWY(_LinearPKSBase):
    """Linear pairwise Desai--Weinert--Yusypchuk (backward-RTS) smoother.

    Equivalent to RTS on the linear-Gaussian model (same smoothed mean and
    covariance to machine precision), obtained via the time-reversed
    complementary couple model. See :func:`_dwy_pass`.
    """

    def _smoothing_pass(self, N_records: int) -> None:
        _dwy_pass(self, N_records)


class Linear_PKS_VAR(_LinearPKSBase):
    """Linear pairwise variational smoother: the smoothed trajectory as the
    solution of a single block-tridiagonal linear system (the lifted/QP form that
    unifies the five recursions). Equivalent to RTS on the linear-Gaussian model.
    See :func:`_lifted_pass`."""

    def _smoothing_pass(self, N_records: int) -> None:
        _lifted_pass(self, N_records)


class Linear_PKS(_LinearPKSBase):
    """
    Linear Pairwise Kalman Smoother — façade selecting a smoothing variant.

    Backward-compatible: ``Linear_PKS(param, sKey=..., joseph=...)`` keeps the
    historical RTS behaviour (``method="RTS"`` is the default). Other variants
    (``"BF"``, ``"MBF"``, ``"MF"``/``"2F"``, ``"DWY"``, ``"VAR"``) are registered
    as they are implemented; on the linear-Gaussian model they all produce the same
    smoothed estimate (Geng et al., 2023) and differ only by mechanics. ``"MF"`` and
    ``"2F"`` select the same two-filter smoother (``"2F"`` is the name used in the
    companion paper; ``"MF"`` is kept for backward compatibility).

    Parameters
    ----------
    param, sKey, verbose, joseph
        See :class:`_LinearPKSBase`.
    method : str, optional
        Smoother variant key, one of :data:`_SMOOTHING_PASSES` (default
        ``"RTS"``).

    Raises
    ------
    FilterError
        If ``method`` is not a registered variant.
    """

    def __init__(
        self,
        param: ParamLinear | ParamNonLinear,
        sKey: int | None = None,
        verbose: int = 0,
        joseph: bool = False,
        method: str = "RTS",
    ) -> None:
        super().__init__(param, sKey, verbose, joseph)
        if method not in _SMOOTHING_PASSES:
            raise FilterError(
                f"Unknown smoother method {method!r}; "
                f"available: {sorted(_SMOOTHING_PASSES)}."
            )
        self.method: str = method

    def _smoothing_pass(self, N_records: int) -> None:
        _SMOOTHING_PASSES[self.method](self, N_records)

    def _supports_gaps(self) -> bool:
        return self.method == "RTS"
