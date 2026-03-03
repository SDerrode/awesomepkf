#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np

from prg.models.nonLinear.base_model_nonLinear import BaseModelNonLinear
from prg.models.Generate_MatrixCov import generate_block_matrix


__all__ = ["ModelCubique"]


class ModelCubique(BaseModelNonLinear):
    """
    Cubic nonlinear model with 1D state and 1D observation.

    System dynamics:
        f(x) = 0.9*x - 0.2*x^3

    Measurement equation:
        h(x) = x

    Includes additive Gaussian process and observation noise.
    """

    MODEL_NAME: str = "x1_y1_cubique"

    def __init__(self) -> None:
        super().__init__(dim_x=1, dim_y=1, model_type="nonlinear")

        # Covariance and initial state
        # self.mQ = np.diag([1e-4, 1e-4])
        # self.mz0 = np.zeros((self.dim_xy, 1))
        # self.Pz0 = np.eye(self.dim_xy)
        self.mQ = generate_block_matrix(
            self._randMatrices.rng, self.dim_x, self.dim_y, 0.30
        )
        self.mz0 = self._randMatrices.rng.standard_normal((self.dim_xy, 1))
        self.Pz0 = generate_block_matrix(
            self._randMatrices.rng, self.dim_x, self.dim_y, 0.30
        )

    # ------------------------------------------------------------------
    def _fx(self, x: np.ndarray, t: np.ndarray, dt: float) -> np.ndarray:
        """
        State transition function with cubic nonlinearity.

        Args:
            x : np.ndarray, shape (1,1) - current state
            t : np.ndarray, shape (1,1) - process noise
            dt: float - timestep

        Returns:
            np.ndarray, shape (1,1) - next state
        """

        return 0.9 * x - 0.2 * x**3 + t

    # ------------------------------------------------------------------
    def _hx(self, x: np.ndarray, u: np.ndarray, dt: float) -> np.ndarray:
        """
        Linear observation function with additive noise.

        Args:
            x : np.ndarray, shape (1,1) - predicted state
            u : np.ndarray, shape (1,1) - observation noise
            dt: float - timestep

        Returns:
            np.ndarray, shape (1,1) - measurement
        """

        return x + u

    # ------------------------------------------------------------------
    def _g(
        self, x: np.ndarray, y: np.ndarray, t: np.ndarray, u: np.ndarray, dt: float
    ) -> np.ndarray:
        """
        Combine state and observation using Wojciech's formulation.

        Args:
            x : np.ndarray, shape (1,1) - current state
            y : np.ndarray, shape (1,1) - current observation
            t : np.ndarray, shape (1,1) - process noise
            u : np.ndarray, shape (1,1) - observation noise
            dt: float - timestep

        Returns:
            np.ndarray, shape (2,1) - combined state + observation
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

        An = np.array(
            [[0.9 - 0.6 * x[0, 0] ** 2, 0.0], [0.9 - 0.6 * x[0, 0] ** 2, 0.0]]
        )
        Bn = np.array([[1.0, 0.0], [1.0, 1.0]])

        return An, Bn
