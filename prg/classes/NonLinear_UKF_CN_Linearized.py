#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
####################################################################
Unscented Kalman Filter (UKF) — nonlinear, additive noise
####################################################################

Differences compared to the UPKF:
  - No pairwise structure: the state is not augmented with the observation.
  - Two independent sigma-point sets:
      * ``sigma_pred_set``  (dim = dim_x) for the prediction step,
      * ``sigma_upd_set``   (dim = dim_x) for the update step.
  - Q and R are injected additively (unaugmented UKF).
  - The cross-correlation M = E[t·uᵀ] is taken into account in
    P_xy and P_yy (first-order correction via the Jacobian H).
  - ``_fx`` and ``_hx`` encapsulate the state equation and the
    observation equation respectively; both are vectorised over
    the sigma-point axis (batch axis 0).
  - At the end of each cycle, a block (Zkp1_predict, Pkp1_predict)
    of dimension (dim_xy × dim_xy) is assembled in order to reuse
    :meth:`PKF._nextUpdating` without modification.
"""

from __future__ import annotations
from typing import Generator, Optional

import numpy as np
from scipy.linalg import LinAlgError

from prg.classes.PKF import PKF
from prg.classes.SigmaPointsSet import SigmaPointsSet
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
       cross-correlation correction M via the Jacobian H,
       gain computation via :meth:`PKF._nextUpdating`.

    Parameters
    ----------
    param : ParamLinear | ParamNonLinear
        Model parameters. Must expose:

        * ``mQ`` — full noise covariance matrix, shape ``(dim_xy, dim_xy)``:

          .. code-block:: text

              mQ = [ Q   M  ]
                   [ M^T R  ]

          with ``Q = E[t·tᵀ]``, ``R = E[u·uᵀ]``, ``M = E[t·uᵀ]``.

        * ``f(x, t, dt)`` — vectorised transition function;
        * ``h(x, u, dt)`` — vectorised observation function;
        * ``jacobiens_g(z, noise_z, dt)`` — Jacobians of g, from which dh/dx is extracted.

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
    FilterError
        If ``param.pairwiseModel`` is ``True``.
    """

    def __init__(
        self,
        param,
        sigmaSet: str,
        sKey: Optional[int] = None,
        verbose: int = 0,
    ) -> None:

        super().__init__(param, sKey, verbose)

        try:
            cls = SigmaPointsSet.registry[sigmaSet]
        except KeyError:
            raise ParamError(
                f"Jeu de sigma-points inconnu : {sigmaSet!r}. "
                f"Disponibles : {list(SigmaPointsSet.registry.keys())}."
            )

        if self.param.pairwiseModel:
            raise FilterError(f"Failed to process a pairwise model with UKF.")

        # Sigma-point set for the prediction step (state space dim_x)
        self.sigma_pred_set = cls(dim=self.dim_x, param=self.param)

        # Sigma-point set for the update step (state space dim_x)
        self.sigma_upd_set = cls(dim=self.dim_x, param=self.param)

        # Extract mQ blocks once — avoids slicing inside the loop.
        #
        #   mQ = [ Q_x   M  ]
        #        [ M^T   R  ]
        #
        self._Q_x: np.ndarray = self.param.mQ[: self.dim_x, : self.dim_x]
        self._R: np.ndarray = self.param.mQ[self.dim_x :, self.dim_x :]
        self._M: np.ndarray = self.param.mQ[: self.dim_x, self.dim_x :]

    # ------------------------------------------------------------------
    # Main filter loop
    # ------------------------------------------------------------------

    def process_filter(
        self,
        N: Optional[int] = None,
        data_generator: Optional[
            Generator[tuple[int, np.ndarray, np.ndarray], None, None]
        ] = None,
    ) -> Generator[tuple[int, np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
        """
        Runs the UKF filter as a generator.

        Parameters
        ----------
        N : int, optional
            Maximum number of time steps. If ``None``, runs until the data generator is exhausted.
        data_generator : Generator, optional
            External data generator. If ``None``, the internal generator is used.

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
        zeros_x: Optional[np.ndarray] = None
        zeros_y: Optional[np.ndarray] = None

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

            # Vectorised propagation through f → f(σ_i)
            sigma_f = self.param.f(sigma_pred, zeros_x, self.dt)  # (n_sigma, dim_x, 1)

            # Predicted mean x_pred = Σ Wm_i · f(σ_i)
            x_pred: np.ndarray = np.sum(
                self.sigma_pred_set.Wm[:, None, None] * sigma_f, axis=0
            )  # (dim_x, 1)

            # Predicted covariance P_xx_pred = Σ Wc_i · δf_i δf_iᵀ + Q
            diffs_f = sigma_f - x_pred  # (n_sigma, dim_x, 1)
            P_xx_pred: np.ndarray = (
                np.einsum("i,ijk,ilk->jl", self.sigma_pred_set.Wc, diffs_f, diffs_f)
                + self._Q_x
            )  # (dim_x, dim_x)

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

            # Vectorised propagation through h — zero noise (additive: R added to P_yy)
            sigma_h = self.param.h(sigma_upd, zeros_y, self.dt)  # (n_sigma, dim_y, 1)

            # Predicted observation y_pred = Σ Wm_i · h(σ_i)
            y_pred: np.ndarray = np.sum(
                self.sigma_upd_set.Wm[:, None, None] * sigma_h, axis=0
            )  # (dim_y, 1)

            # Jacobian H = dh/dx evaluated at x_pred — required for the M correction.
            # jacobiens_g expects (z, noise_z, dt) with z of shape (dim_xy, 1).
            # Bn[dim_x:, :dim_x] = dh/dx (cf. _jacobiens_g in base_model_fxhx).
            z_pred_aug = np.concatenate(
                [x_pred, np.zeros((self.dim_y, 1))], axis=0
            )  # (dim_xy, 1)
            _, Bn = self.param.jacobiens_g(
                z_pred_aug, np.zeros((self.dim_xy, 1)), self.dt
            )
            H_pred: np.ndarray = Bn[self.dim_x :, : self.dim_x]  # (dim_y, dim_x)

            # Innovation covariance
            #   P_yy = Σ Wc_i · δh_i δh_iᵀ  +  R  +  H·M + Mᵀ·Hᵀ
            # The term H·M + Mᵀ·Hᵀ (symmetric) ensures P_yy is consistent with P_xy
            # when E[t·uᵀ] = M ≠ 0 (chain rule on h∘f).
            # When M = 0, the term vanishes → identical behaviour to the uncorrelated case.
            diffs_h = sigma_h - y_pred  # (n_sigma, dim_y, 1)
            P_yy: np.ndarray = (
                np.einsum("i,ijk,ilk->jl", self.sigma_upd_set.Wc, diffs_h, diffs_h)
                + self._R
                + H_pred @ self._M
                + self._M.T @ H_pred.T
            )  # (dim_y, dim_y)

            # Cross-covariance
            #   P_xy = Σ Wc_i · δx_i δh_iᵀ  +  M
            diffs_x = sigma_upd - x_pred  # (n_sigma, dim_x, 1)
            P_xy: np.ndarray = (
                np.einsum("i,ijk,ilk->jl", self.sigma_upd_set.Wc, diffs_x, diffs_h)
                + self._M
            )  # (dim_x, dim_y)

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
