#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from .base_model_nonLinear import BaseModelNonLinear

from others.geneMatriceCov import generate_block_matrix


class ModelSinus(BaseModelNonLinear):
    """
    Nonlinear model with 1D state and 1D observation:

    State dynamics:
        f(x) = 0.8 * x + 0.3 * sin(x)

    Measurement equation:
        h(x) = x^2

    Includes additive Gaussian process and observation noise.
    """

    MODEL_NAME: str = "x1_y1_sinus"

    def __init__(self) -> None:
        super().__init__(dim_x=1, dim_y=1, model_type="nonlinear")

        # Covariance and initial state
        # self.mQ = np.diag([1e-3, 1e-3])
        # self.mz0 = (
        #     np.zeros((self.dim_xy, 1)) + 1.0
        # )  # le +1 est important pour lancer le filtre
        # self.Pz0 = np.eye(self.dim_xy)
        self.mQ = generate_block_matrix(
            self._randMatrices.rng, self.dim_x, self.dim_y, 0.15
        )
        self.mz0 = self._randMatrices.rng.standard_normal((self.dim_xy, 1))
        self.Pz0 = generate_block_matrix(
            self._randMatrices.rng, self.dim_x, self.dim_y, 0.15
        )

    # ------------------------------------------------------------------
    def _fx(self, x: np.ndarray, t: np.ndarray, dt: float) -> np.ndarray:
        """
        State transition function with additive process noise.

        Args:
            x : np.ndarray, shape (1,1) - current state
            t : np.ndarray, shape (1,1) - process noise
            dt: float - timestep

        Returns:
            np.ndarray, shape (1,1) - next state
        """
        return 0.8 * x + 0.3 * np.sin(x) + t

    # ------------------------------------------------------------------
    def _hx(self, x: np.ndarray, u: np.ndarray, dt: float) -> np.ndarray:
        """
        Measurement function with additive observation noise.

        Args:
            x : np.ndarray, shape (1,1) - predicted state
            u : np.ndarray, shape (1,1) - observation noise
            dt: float - timestep

        Returns:
            np.ndarray, shape (1,1) - measurement
        """
        return x**2 + u

    # ------------------------------------------------------------------
    def _g(
        self, x: np.ndarray, y: np.ndarray, t: np.ndarray, u: np.ndarray, dt: float
    ) -> np.ndarray:
        """
        Combine state and observation using Wojciech's formulation.

        Returns:
            np.ndarray, shape (2,1) - stacked state + observation
        """
        if __debug__:
            assert x.shape == (1, 1)
            assert y.shape == (1, 1)
            assert t.shape == (1, 1)
            assert u.shape == (1, 1)

        fx_val = self._fx(x, t, dt)
        hx_val = self._hx(fx_val, u, dt)
        return np.vstack((fx_val, hx_val))

    # ------------------------------------------------------------------
    def _jacobiens_g(
        self, x: np.ndarray, y: np.ndarray, t: np.ndarray, u: np.ndarray, dt: float
    ):
        """
        Compute Jacobians of g w.r.t state and noise.

        Returns:
            Tuple[np.ndarray, np.ndarray] : (dg/dz, dg/dnoise)
        """
        if __debug__:
            assert x.shape == (1, 1)
            assert y.shape == (1, 1)
            assert t.shape == (1, 1)
            assert u.shape == (1, 1)

        x1 = x.flatten()[0]
        t1 = t.flatten()[0]

        # State plus noise
        A = 0.8 * x1 + 0.3 * np.sin(x1) + t1

        # Jacobians exactly as in original code
        An = np.array(
            [[0.8 + 0.3 * np.cos(x1), 0.0], [2.0 * A * (0.8 + 0.3 * np.cos(x1)), 0.0]]
        )
        Bn = np.array([[1.0, 0.0], [2.0 * A, 1.0]])

        return An, Bn
