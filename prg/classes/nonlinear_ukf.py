"""
####################################################################
Unscented Kalman Filter (UKF) — nonlinear, additive noise
####################################################################

Differences compared to the UPKF:
  - No pairwise structure: the state is not augmented with
    the observation.
  - Two independent sigma-point sets:
      * ``sigma_pred_set``  (dim = dim_x) for the prediction step,
      * ``sigma_upd_set``   (dim = dim_x) for the update step.
  - Q and R are injected in an **additive** fashion (non-augmented UKF).
  - ``_fx`` and ``_hx`` encapsulate the state equation and the
    observation equation respectively; both are vectorised over
    the sigma-point axis (batch axis 0).
  - At the end of each cycle, a block (Zkp1_predict, Pkp1_predict)
    of dimension (dim_xy × dim_xy) is assembled in order to reuse
    :meth:`PKF._nextUpdating` without modification.
"""

from __future__ import annotations

import contextlib
from collections.abc import Generator

import numpy as np

from prg.classes.pkf import PKF
from prg.classes.sigma_points_set import SigmaPointsSet
from prg.utils.exceptions import (
    FilterError,
    InvertibilityError,
    NumericalError,
    ParamError,
)

__all__ = ["NonLinear_UKF"]


class NonLinear_UKF(PKF):
    """
    Nonlinear Unscented Kalman Filter (UKF) with additive noise.

    Extends :class:`PKF` by implementing the standard UKF cycle:

    1. **Prediction** — sigma-points on the current state ``(x, P_xx)``,
       propagation through :meth:`_fx`, predicted covariance augmented by Q.
    2. **Update** — sigma-points on the predicted state ``(x_pred, P_xx_pred)``,
       propagation through :meth:`_hx`, innovation covariance augmented by R,
       gain computation via :meth:`PKF._nextUpdating`.

    Parameters
    ----------
    param : ParamLinear | ParamNonLinear
        Model parameters. Must expose:

        * ``mQ`` — process noise covariance, shape ``(dim_xy, dim_xy)``
        * ``f(x, t, dt)`` — vectorised state transition function;
        * ``h(x, u, dt)`` — vectorised observation function.

    sigmaSet : str
        Key of the sigma-point set in ``SigmaPointsSet.registry``.
    sKey : int, optional
        Random seed for reproducibility.
    verbose : int, optional
        Verbosity level (default 0).

    Raises
    ------
    ParamError
        If ``sigmaSet`` is not a known key in the registry.
    """

    def __init__(
        self,
        param,
        sigmaSet: str,
        sKey: int | None = None,
        verbose: int = 0,
    ) -> None:

        super().__init__(param, sKey, verbose)

        try:
            cls = SigmaPointsSet.registry[sigmaSet]
        except KeyError as e:
            raise ParamError(
                f"Unknown sigma-point set: {sigmaSet!r}. "
                f"Available: {list(SigmaPointsSet.registry.keys())}."
            ) from e

        if self.param.pairwiseModel:
            raise FilterError("UKF does not support pairwise models.")

        # Sigma-point set for the prediction step (state space dim_x)
        self.sigma_pred_set = cls(dim=self.dim_x, param=self.param)

        # Sigma-point set for the update step (state space dim_x)
        self.sigma_upd_set = cls(dim=self.dim_x, param=self.param)

        # Extract Q_x, R and M once — avoids slicing inside the loop.
        #
        # For linear pairwise models the noise enters as  z' = A z + B v,
        # v ~ N(0, mQ), with B = [[B_xx, 0], [B_yx, B_yy]].
        # The UKF reformulates the model as
        #   x' = f(x) + w,  w = B_xx v^x ~ N(0, Q_x)
        #   y  = H x  + e,  e = B_yy v^y ~ N(0, R)
        # where v^x and v^y may be correlated: M = Cov(w, e) = B_xx mQ_xy B_yy^T.
        #
        # Using B @ mQ @ B^T to extract R is WRONG: the [dim_x:, dim_x:] block
        # equals  B_yx mQ_xx B_yx^T + ... + B_yy mQ_yy B_yy^T, which contains
        # a H·Q_x·H^T term that is already counted in H·P_pred·H^T → double
        # counting inflates R and biases the Kalman gain.
        #
        # Correct extraction for linear models:
        #   Q_x = B_xx @ mQ_xx @ B_xx^T   (unchanged — B @ mQ @ B^T [0:, 0:] is the same)
        #   R   = B_yy @ mQ_yy @ B_yy^T   (pure obs-noise variance, no H·Q·H^T term)
        #   M   = B_xx @ mQ_xy @ B_yy^T   (process/obs cross-covariance)
        #
        # For nonlinear models B = I implicitly, so mQ is the effective covariance
        # and noise channels are independent (M = 0).
        if hasattr(self.param, "B"):
            _B   = self.param.B
            _mQ  = self.param.mQ
            _Bxx = _B[: self.dim_x, : self.dim_x]   # process-noise input  (dim_x, dim_x)
            _Byy = _B[self.dim_x :, self.dim_x :]   # obs-noise input      (dim_y, dim_y)
            self._Q_x: np.ndarray = _Bxx @ _mQ[: self.dim_x, : self.dim_x] @ _Bxx.T
            self._R:   np.ndarray = _Byy @ _mQ[self.dim_x :, self.dim_x :] @ _Byy.T
            self._M:   np.ndarray | None = (
                _Bxx @ _mQ[: self.dim_x, self.dim_x :] @ _Byy.T
            )
        else:
            _mQ = self.param.mQ
            self._Q_x = _mQ[: self.dim_x, : self.dim_x]
            self._R   = _mQ[self.dim_x :, self.dim_x :]
            self._M   = None

        # For linear models with pairwise A matrix, _hx gives h(x) = A_yx @ x
        # where A_yx = H_true @ F (classic) or H_true @ A_aug (augmented).
        # The standard UKF update step applies h to sigma-points representing
        # x_{k+1|k}, so it needs H_true @ x_{k+1}, not H_true @ F @ x_{k+1}.
        # H_true is recovered analytically: H_true = A_yx @ inv(F)
        # where F = A[:dim_x, :dim_x].  This identity holds for both classic
        # (A_yx = H @ F) and augmented (A_yx = H_aug @ A_pairwise) models.
        # For nonlinear models (no A attribute), param.h is used directly.
        self._H_obs: np.ndarray | None = None
        if hasattr(self.param, "A"):
            _F_blk = self.param.A[: self.dim_x, : self.dim_x]  # (dim_x, dim_x)
            _A_yx  = self.param.A[self.dim_x :, : self.dim_x]  # (dim_y, dim_x)
            with contextlib.suppress(np.linalg.LinAlgError):
                # singular F — leave self._H_obs as None, fallback to param.h
                self._H_obs = _A_yx @ np.linalg.inv(_F_blk)   # (dim_y, dim_x)

    # ------------------------------------------------------------------
    # Main filter loop
    # ------------------------------------------------------------------

    def process_filter(
        self,
        N: int | None = None,
        data_generator: Generator[tuple[int, np.ndarray, np.ndarray], None, None] | None = None,
    ) -> Generator[tuple[int, np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
        """
        Runs the UKF filter as a generator.

        Parameters
        ----------
        N : int, optional
            Maximum number of time steps. If ``None``, runs until the data
            generator is exhausted.
        data_generator : Generator, optional
            External data generator. If ``None``, the internal generator
            is used.

        Yields
        ------
        k : int
            Current time index.
        x_true : np.ndarray or None
            Ground truth at time ``k``.
        y_observed : np.ndarray
            Observation at time ``k``.
        X_predict : np.ndarray
            Prior estimate, shape ``(dim_x, 1)``.
        X_update : np.ndarray
            Posterior estimate, shape ``(dim_x, 1)``.

        Raises
        ------
        ParamError
            If ``N`` is not a strictly positive integer or ``None``.
        InvertibilityError
            If the innovation matrix ``Skp1`` is not invertible.
        NumericalError
            If the predicted covariance ``P_xx_pred`` is not valid.
        FilterError
            If an unexpected error occurs during the update.
        """

        self._validate_N(N)
        self.history.clear()

        generator = (
            data_generator if data_generator is not None else self._data_generation()
        )

        # --- First estimate (Gaussian conditioning on y_0) ------------------
        step = self._firstEstimate(generator)
        if step.xkp1 is None:
            self.ground_truth = False

        yield step.k, step.xkp1, step.ykp1, step.Xkp1_predict, step.Xkp1_update

        # Pre-allocation of the augmented covariance block reused at each step
        Pkp1_predict = self.zeros_dim_xy_xy.copy()

        # Zero noise vectors — allocated once at the first step
        zeros_x: np.ndarray | None = None
        zeros_y: np.ndarray | None = None

        # --- Main loop ----------------------------------------------------
        while N is None or step.k < N:

            # ================================================================
            # PREDICTION STEP
            # ================================================================

            # Sigma-points on (x_k, P_xx_k) — dimension dim_x
            sigma_pred_list = self.sigma_pred_set._sigma_point(
                step.Xkp1_update, step.PXXkp1_update
            )
            sigma_pred = np.array(sigma_pred_list)  # (n_sigma, dim_x, 1)
            n_sigma = sigma_pred.shape[0]

            if zeros_x is None:
                zeros_x = np.zeros((n_sigma, self.dim_x, 1))

            # Vectorised propagation through f  →  f(σ_i)
            sigma_f = self.param.f(sigma_pred, zeros_x, self.dt)  # (n_sigma, dim_x, 1)

            # Predicted mean  x_pred = Σ Wm_i · f(σ_i)
            x_pred: np.ndarray = np.sum(
                self.sigma_pred_set.Wm[:, None, None] * sigma_f, axis=0
            )  # (dim_x, 1)

            # Predicted covariance  P_xx_pred = Σ Wc_i · δf_i δf_iᵀ  +  Q
            diffs_f = sigma_f - x_pred  # (n_sigma, dim_x, 1)
            P_xx_pred: np.ndarray = (
                np.einsum("i,ijk,ilk->jl", self.sigma_pred_set.Wc, diffs_f, diffs_f)
                + self._Q_x
            )  # (dim_x, dim_x)
            # Force exact symmetry — protects the downstream Cholesky.
            P_xx_pred = 0.5 * (P_xx_pred + P_xx_pred.T)

            # Validation — raises NumericalError if invalid
            self._check_covariance(P_xx_pred, step.k, name="P_xx_pred")

            # ================================================================
            # UPDATE STEP (sigma-points on the predicted state)
            # ================================================================

            # Sigma-points on (x_pred, P_xx_pred) — dimension dim_x
            sigma_upd_list = self.sigma_upd_set._sigma_point(x_pred, P_xx_pred)
            sigma_upd = np.array(sigma_upd_list)  # (n_sigma, dim_x, 1)

            # Zero auxiliary term for _hx — allocated once
            if zeros_y is None:
                zeros_y = np.zeros((n_sigma, self.dim_y, 1))

            # Vectorised propagation through h  →  h(σ_i)
            # Linear models: apply H_true directly (avoids the pairwise A_yx @ x
            # bias where A_yx = H_true @ F introduces an extra factor F).
            # Nonlinear models: call param.h as usual.
            if self._H_obs is not None:
                sigma_h = np.einsum("ij,njk->nik", self._H_obs, sigma_upd)
            else:
                sigma_h = self.param.h(sigma_upd, zeros_y, self.dt)  # (n_sigma, dim_y, 1)

            # Predicted observation  y_pred = Σ Wm_i · h(σ_i)
            y_pred: np.ndarray = np.sum(
                self.sigma_upd_set.Wm[:, None, None] * sigma_h, axis=0
            )  # (dim_y, 1)

            # Innovation covariance  P_yy = Σ Wc_i · δh_i δh_iᵀ  +  R
            diffs_h = sigma_h - y_pred  # (n_sigma, dim_y, 1)
            P_yy: np.ndarray = (
                np.einsum("i,ijk,ilk->jl", self.sigma_upd_set.Wc, diffs_h, diffs_h)
                + self._R
            )  # (dim_y, dim_y)
            # Force exact symmetry — Skp1 = P_yy is Cholesky-factored downstream.
            P_yy = 0.5 * (P_yy + P_yy.T)

            # Cross-covariance  P_xy = Σ Wc_i · δx_i δh_iᵀ
            diffs_x = sigma_upd - x_pred  # (n_sigma, dim_x, 1)
            P_xy: np.ndarray = np.einsum(
                "i,ijk,ilk->jl", self.sigma_upd_set.Wc, diffs_x, diffs_h
            )  # (dim_x, dim_y)

            # Correction for correlated process/observation noise (linear models).
            # The unscented cross-covariance P_xy misses M = Cov(w, e) and P_yy
            # misses H·M + M^T·H^T (see derivation in __init__).
            if self._M is not None and self._H_obs is not None:
                P_xy += self._M
                _HM   = self._H_obs @ self._M   # (dim_y, dim_y)
                P_yy += _HM + _HM.T

            # ================================================================
            # Assembly of the augmented block expected by _nextUpdating:
            #
            #   Zkp1_predict = [ x_pred ]   shape (dim_xy, 1)
            #                  [ y_pred ]
            #
            #   Pkp1_predict = [ P_xx_pred  P_xy ]   shape (dim_xy, dim_xy)
            #                  [ P_xy.T     P_yy ]
            # ================================================================
            Zkp1_predict: np.ndarray = np.concatenate(
                [x_pred, y_pred], axis=0
            )  # (dim_xy, 1)

            Pkp1_predict[: self.dim_x, : self.dim_x] = P_xx_pred
            Pkp1_predict[: self.dim_x, self.dim_x :] = P_xy
            Pkp1_predict[self.dim_x :, : self.dim_x] = P_xy.T
            Pkp1_predict[self.dim_x :, self.dim_x :] = P_yy

            # Consume the next observation
            try:
                new_k, new_xkp1, new_ykp1 = next(generator)
            except StopIteration:
                return  # generator exhausted — normal stop, not an error

            # Kalman update — custom exceptions propagate naturally
            try:
                step = self._nextUpdating(
                    new_k, new_xkp1, new_ykp1, Zkp1_predict, Pkp1_predict
                )
            except (InvertibilityError, NumericalError):
                raise
            except Exception as e:
                raise FilterError(
                    f"Step {new_k}: unexpected error during update step."
                ) from e

            yield step.k, step.xkp1, step.ykp1, step.Xkp1_predict, step.Xkp1_update
