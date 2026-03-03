#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from .base_model_nonLinear import BaseModelNonLinear

from others.geneMatriceCov import generate_block_matrix


class ModelX2Y2_withRetroactions(BaseModelNonLinear):
    """
    Nonlinear model with retro-actions of observations and of states.
    The model includes additive Gaussian process and observation noises.
    """

    MODEL_NAME: str = "x2_y2_withRetroactions"

    def __init__(self) -> None:
        super().__init__(dim_x=2, dim_y=2, model_type="nonlinear")

        Q = np.array([[0.08, 0.01], [0.01, 0.05]])
        R = np.array([[0.1, 0.0], [0.0, 0.05]])
        M = np.array([[0.01, 0.0], [0.0, 0.01]])
        self.mQ = np.block([[Q, M], [M.T, R]]) / 2.0
        self.mz0 = np.zeros((self.dim_xy, 1))
        self.Pz0 = np.eye(self.dim_xy) / 20.0

        # self.mQ = generate_block_matrix(
        #     self._randMatrices.rng, self.dim_x, self.dim_y, 0.001
        # )
        # self.mz0 = self._randMatrices.rng.standard_normal((self.dim_xy, 1))
        # self.Pz0 = generate_block_matrix(
        #     self._randMatrices.rng, self.dim_x, self.dim_y, 0.001
        # )

    # ------------------------------------------------------------------
    def _gx(self, x, y, t, u, dt):
        """
        Nonlinear state function with retro-action on observation.
        """
        x1, x2 = x.flatten()
        y1 = y.flatten()[0]
        t1, t2 = t.flatten()

        return np.array(
            [[x1 + 0.1 * x2 * np.tanh(y1) + t1], [0.9 * x2 + 0.1 * np.sin(x1) + t2]]
        )

    # ------------------------------------------------------------------
    def _gy(self, x, y, t, u, dt):
        """
        Nonlinear observation function with retro-action on previous observation.
        """
        x1, x2 = x.flatten()
        y1, y2 = y.flatten()
        u1, u2 = u.flatten()

        # return np.array([[x1**2 - 0.3 * y2 + u1], [x2 + 0.3 * y1 + u2]])
        return np.array([[x1 - 0.3 * y2 + u1], [x2 + 0.3 * y1 + u2]])

    # ------------------------------------------------------------------
    def _g(self, x, y, t, u, dt):
        """
        Combined state and observation using Wojciech’s formulation.
        """
        if __debug__:
            assert x.shape == (2, 1), f"x must be (2,1), got {x.shape}"
            assert y.shape == (2, 1), f"y must be (2,1), got {y.shape}"
            assert t.shape == (2, 1), f"t must be (2,1), got {t.shape}"
            assert u.shape == (2, 1), f"u must be (2,1), got {u.shape}"
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
            assert y.shape == (2, 1), f"y must be (2,1), got {y.shape}"
            assert t.shape == (2, 1), f"t must be (2,1), got {t.shape}"
            assert u.shape == (2, 1), f"u must be (2,1), got {u.shape}"
            assert isinstance(dt, (float, int)), "dt must be a float"

        x1, x2 = x.flatten()
        y1 = y.flatten()[0]

        An = np.array(
            [
                [1.0, 0.1 * np.tanh(y1), 0.1 * x2 * (1.0 - np.tanh(y1)), 0.0],
                [0.1 * np.cos(x1), 0.9, 0.0, 0.0],
                [1.0, 0.0, 0.0, -0.3],
                [0.0, 1.0, 0.3, 0.0],
            ]
        )

        Bn = np.eye(self.dim_xy)

        return An, Bn
