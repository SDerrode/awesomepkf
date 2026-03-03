#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np

from prg.models.nonLinear.base_model_nonLinear import BaseModelNonLinear
from prg.models.Generate_MatrixCov import generate_block_matrix

__all__ = ["ModelX2Y1_withRetroactionsOfObservations"]


class ModelX2Y1_withRetroactionsOfObservations(BaseModelNonLinear):
    """
    Nonlinear model with retro-actions of observations and of states.
    The model includes additive Gaussian process and observation noises.
    """

    MODEL_NAME: str = "x2_y1_withRetroactionsOfObservations"

    def __init__(self) -> None:
        super().__init__(dim_x=2, dim_y=1, model_type="nonlinear")

        self.a, self.b, self.c, self.d, self.e, self.f = 1.0, 0.8, 0.05, 0.9, 0.30, 0.6

        # self.mQ = np.diag([1e-1, 1e-1, 5e-1])
        # self.mz0 = np.zeros((self.dim_xy, 1))
        # self.Pz0 = np.eye(self.dim_xy)
        self.mQ = generate_block_matrix(
            self._randMatrices.rng, self.dim_x, self.dim_y, 0.03
        )
        self.mz0 = self._randMatrices.rng.standard_normal((self.dim_xy, 1))
        self.Pz0 = generate_block_matrix(
            self._randMatrices.rng, self.dim_x, self.dim_y, 0.05
        )

    # ------------------------------------------------------------------
    def _gx(self, x, y, t, u, dt):
        """
        Nonlinear state function with retro-action on observation.
        """
        x1, x2 = x.flatten()
        y1 = y.flatten()[0]
        t1, t2 = t.flatten()

        return np.array(
            [
                [self.a * x1 + self.b * x2 + self.c * np.tanh(y1) + t1],
                [self.d * x2 + self.e * np.sin(y1) + t2],
            ]
        )

    # ------------------------------------------------------------------
    def _gy(self, x, y, t, u, dt):
        """
        Nonlinear observation function with retro-action on previous observation.
        """

        return np.array([[x[0, 0] ** 2 + self.f * y[0, 0] + u[0, 0]]])

    # ------------------------------------------------------------------
    def _g(self, x, y, t, u, dt):
        """
        Combined state and observation using Wojciech’s formulation.
        """
        if __debug__:
            assert x.shape == (2, 1), f"x must be (2,1), got {x.shape}"
            assert y.shape == (1, 1), f"y must be (1,1), got {y.shape}"
            assert t.shape == (2, 1), f"t must be (2,1), got {t.shape}"
            assert u.shape == (1, 1), f"u must be (1,1), got {u.shape}"
            assert isinstance(dt, (float, int)), "dt must be a float"

        gx_val = self._gx(x, y, t, u, dt)
        gy_val = self._gy(x, y, t, u, dt)
        return np.vstack((gx_val, gy_val))

    # ------------------------------------------------------------------
    def _jacobiens_g(self, x, y, t, u, dt):
        """
        Jacobians of combined state and observation function.
        """
        if __debug__:
            assert x.shape == (2, 1), f"x must be (2,1), got {x.shape}"
            assert y.shape == (1, 1), f"y must be (1,1), got {y.shape}"
            assert t.shape == (2, 1), f"t must be (2,1), got {t.shape}"
            assert u.shape == (1, 1), f"u must be (1,1), got {u.shape}"
            assert isinstance(dt, (float, int)), "dt must be a float"

        x1 = x.flatten()[0]
        y1 = y.flatten()[0]

        An = np.array(
            [
                [self.a, self.b, self.c * (1.0 - np.tanh(y1) ** 2)],
                [0.0, self.d, self.e * np.cos(y1)],
                [2.0 * x1, 0.0, self.f],
            ]
        )

        Bn = np.eye(self.dim_xy)

        return An, Bn
