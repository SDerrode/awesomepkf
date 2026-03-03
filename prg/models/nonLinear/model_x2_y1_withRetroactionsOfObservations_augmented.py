#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from prg.models.nonLinear.base_model_nonLinear import BaseModelNonLinear
from prg.models.nonLinear.model_x2_y1_withRetroactionsOfObservations import (
    ModelX2Y1_withRetroactionsOfObservations,
)
from prg.exceptions import NumericalError

__all__ = ["ModelX2Y1_withRetroactionsOfObservations_augmented"]


class ModelX2Y1_withRetroactionsOfObservations_augmented(BaseModelNonLinear):
    """
    Nonlinear model with retro-actions of observations and of states.
    The model includes additive Gaussian process and observation noises.
    ATTENTION : ce modèle a été construit pour être utilisé avec un filtre
                UKF et comparé avec le modèle 'ModelX2Y1_withRetroactionsOfObservations'
                pour un filtre UPKF, cf rapport.
    """

    MODEL_NAME: str = "x2_y1_withRetroactionsOfObservations_augmented"

    def __init__(self) -> None:
        super().__init__(dim_x=3, dim_y=1, model_type="nonlinear", augmented=True)

        # NumericalError remonte naturellement si __init__ du modèle interne échoue
        self.mod = ModelX2Y1_withRetroactionsOfObservations()

        try:
            self.mQ = np.zeros((self.dim_xy, self.dim_xy))
            self.mQ[: self.dim_x, : self.dim_x] = self.mod.mQ

            dim_x = self.mod.dim_x
            dim_y = self.mod.dim_y
            dim_xy = self.mod.dim_xy

            self.mz0 = np.zeros((dim_xy + dim_y, 1))
            self.mz0[0:dim_xy] = self.mod.mz0
            self.mz0[dim_xy : dim_xy + dim_y] = self.mz0[dim_xy - dim_y : dim_xy]

            self.Pz0 = np.zeros((dim_xy + dim_y, dim_xy + dim_y))
            self.Pz0[0:dim_xy, 0:dim_xy] = self.mod.Pz0
            self.Pz0[dim_xy : dim_xy + dim_y, :] = self.Pz0[dim_xy - dim_y : dim_xy, :]
            self.Pz0[:, dim_xy : dim_xy + dim_y] = self.Pz0[:, dim_xy - dim_y : dim_xy]
        except (ValueError, IndexError) as e:
            raise NumericalError(
                f"[{self.MODEL_NAME}] Initialization failed: {e}"
            ) from e

        self.a, self.b, self.c, self.d, self.e, self.f = (
            self.mod.a,
            self.mod.b,
            self.mod.c,
            self.mod.d,
            self.mod.e,
            self.mod.f,
        )

    # ------------------------------------------------------------------
    def _fx(self, x, t, dt):
        """
        Nonlinear state function with retro-action on observation.
        """
        try:
            ax = self.mod._gx(
                x[: self.dim_x - self.dim_y],
                x[self.dim_x - self.dim_y :],
                t[: self.dim_x - self.dim_y],
                t[self.dim_x - self.dim_y :],
                dt,
            )
            ay = self.mod._gy(
                x[: self.dim_x - self.dim_y],
                x[self.dim_x - self.dim_y :],
                t[: self.dim_x - self.dim_y],
                t[self.dim_x - self.dim_y :],
                dt,
            )
            return np.block([[ax], [ay]])
        except NumericalError:
            raise
        except (ValueError, IndexError) as e:
            raise NumericalError(
                f"[{self.MODEL_NAME}] _fx: error at x={x}, t={t}: {e}"
            ) from e

    # ------------------------------------------------------------------
    def _hx(self, x, u, dt):
        """
        Nonlinear observation function with retro-action on previous observation.
        Le bruit $u$ est nul dans cette formulation.
        """
        try:
            return x[-1].reshape(-1, 1)
        except (IndexError, ValueError) as e:
            raise NumericalError(f"[{self.MODEL_NAME}] _hx: error at x={x}: {e}") from e

    # ------------------------------------------------------------------
    def _g(self, x, y, t, u, dt):
        """
        Combined state and observation using Wojciech's formulation.
        """
        if __debug__:
            assert x.shape == (3, 1), f"x must be (3,1), got {x.shape}"
            assert y.shape == (1, 1), f"y must be (1,1), got {y.shape}"
            assert t.shape == (3, 1), f"t must be (3,1), got {t.shape}"
            assert u.shape == (1, 1), f"u must be (1,1), got {u.shape}"
            assert isinstance(dt, (float, int)), "dt must be a float"

        try:
            fx_val = self._fx(x, t, dt)
            hx_val = self._hx(fx_val, u, dt)
            return np.vstack((fx_val, hx_val))
        except NumericalError:
            raise
        except ValueError as e:
            raise NumericalError(
                f"[{self.MODEL_NAME}] _g: shape mismatch during vstack: {e}"
            ) from e

    # ------------------------------------------------------------------
    def _jacobiens_g(self, x, y, t, u, dt):
        """
        Jacobians of combined state and observation function.
        """
        if __debug__:
            assert x.shape == (3, 1), f"x must be (3,1), got {x.shape}"
            assert y.shape == (1, 1), f"y must be (1,1), got {y.shape}"
            assert t.shape == (3, 1), f"t must be (3,1), got {t.shape}"
            assert u.shape == (1, 1), f"u must be (1,1), got {u.shape}"
            assert isinstance(dt, (float, int)), "dt must be a float"

        try:
            with np.errstate(all="raise"):
                x1, x2, x3 = x.flatten()

                An = np.array(
                    [
                        [self.a, self.b, self.c * (1.0 - np.tanh(x3) ** 2), 0.0],
                        [0, self.d, self.e * np.cos(x3), 0.0],
                        [2 * x1, 0.0, self.f, 0.0],
                        [2 * x1, 0.0, self.f, 0.0],
                    ]
                )
                Bn = np.array(
                    [
                        [1.0, 0.0, 0.0, 0.0],
                        [0.0, 1.0, 0.0, 0.0],
                        [0.0, 0.0, 1.0, 0.0],
                        [0.0, 0.0, 1.0, 0.0],
                    ]
                )

            return An, Bn

        except FloatingPointError as e:
            raise NumericalError(
                f"[{self.MODEL_NAME}] _jacobiens_g: floating point error at x={x}: {e}"
            ) from e
        except (IndexError, ValueError) as e:
            raise NumericalError(
                f"[{self.MODEL_NAME}] _jacobiens_g: array construction error: {e}"
            ) from e
