#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np

from prg.models.nonLinear.base_model_nonLinear import BaseModelNonLinear
from prg.models.Generate_MatrixCov import generate_block_matrix

__all__ = ["ModelX2Y1"]


class ModelX2Y1(BaseModelNonLinear):
    """
    Nonlinear model with:
      - 2-dimensional state vector x = [x1, x2]
      - 1-dimensional observation y

    System dynamics (nonlinear):
        f(x, t) = [
            x1 + 0.05 * x2 + 0.5 * sin(0.1 * x2) + t1,
            0.9 * x2 + 0.2 * cos(0.3 * x1) + t2
        ]

    Measurement equation:
        h(x, u) = sqrt(x1^2 + x2^2) + u

    The model includes additive Gaussian process and observation noise.
    """

    MODEL_NAME: str = "x2_y1"

    def __init__(self) -> None:
        super().__init__(dim_x=2, dim_y=1, model_type="nonlinear")

        # Covariance and initial state
        # self.mQ = np.diag([1e-4, 1e-4, 1e-4])
        # self.mz0 = np.zeros((self.dim_xy, 1))
        # self.Pz0 = np.eye(self.dim_xy)
        self.mQ = generate_block_matrix(
            self._randMatrices.rng, self.dim_x, self.dim_y, 0.05
        )
        self.mz0 = self._randMatrices.rng.standard_normal((self.dim_xy, 1))
        self.Pz0 = generate_block_matrix(
            self._randMatrices.rng, self.dim_x, self.dim_y, 0.05
        )

    # ------------------------------------------------------------------
    def _fx(self, x: np.ndarray, t: np.ndarray, dt: float) -> np.ndarray:
        """
        State transition function with process noise.

        Args:
            x : np.ndarray, shape (2,1) - current state vector
            t : np.ndarray, shape (2,1) - process noise
            dt: float - timestep

        Returns:
            np.ndarray, shape (2,1) - next state
        """
        x1, x2 = x.flatten()
        t1, t2 = t.flatten()

        return np.array(
            [
                [x1 + 0.05 * x2 + 0.5 * np.sin(0.1 * x2) + t1],
                [0.9 * x2 + 0.2 * np.cos(0.3 * x1) + t2],
            ]
        )

    # ------------------------------------------------------------------
    def _hx(self, x: np.ndarray, u: np.ndarray, dt: float) -> np.ndarray:
        """
        Measurement function with observation noise.

        Args:
            x : np.ndarray, shape (2,1) - predicted state
            u : np.ndarray, shape (1,1) - measurement noise
            dt: float - timestep

        Returns:
            np.ndarray, shape (1,1) - measurement
        """

        return np.array([[np.sqrt(x[0, 0] ** 2 + x[1, 0] ** 2) + u[0, 0]]])

    # ------------------------------------------------------------------
    def _g(
        self, x: np.ndarray, y: np.ndarray, t: np.ndarray, u: np.ndarray, dt: float
    ) -> np.ndarray:
        """
        Combined state + observation function for filtering.

        Args:
            x : state vector (dim_x,1)
            y : observation vector (dim_y,1)
            t : process noise (dim_x,1)
            u : observation noise (dim_y,1)
            dt: timestep

        Returns:
            np.ndarray, shape (dim_xy,1) - combined state+observation
        """
        if __debug__:
            assert x.shape == (2, 1)
            assert y.shape == (1, 1)
            assert t.shape == (2, 1)
            assert u.shape == (1, 1)
            assert isinstance(dt, (float, int))

        fx_val = self._fx(x, t, dt)
        hx_val = self._hx(fx_val, u, dt)
        return np.vstack((fx_val, hx_val))

    # ------------------------------------------------------------------
    def _jacobiens_g(
        self, x: np.ndarray, y: np.ndarray, t: np.ndarray, u: np.ndarray, dt: float
    ):
        """
        Computes Jacobians of the combined function g w.r.t state and noise.

        Returns:
            Tuple[np.ndarray, np.ndarray] : (dg/dz, dg/dnoise)
        """
        if __debug__:
            assert x.shape == (2, 1)
            assert y.shape == (1, 1)
            assert t.shape == (2, 1)
            assert u.shape == (1, 1)
            assert isinstance(dt, (float, int))

        x1, x2 = x.flatten()
        t1, t2 = t.flatten()

        # Predicted state
        A_val = x1 + 0.05 * x2 + 0.5 * np.sin(0.1 * x2) + t1
        B_val = 0.9 * x2 + 0.2 * np.cos(0.3 * x1) + t2
        r_val = np.sqrt(A_val**2 + B_val**2)

        # Jacobian w.r.t state (2x2)
        An = np.array(
            [
                [1.0, 0.05 * (1 + np.cos(0.1 * x2)), 0.0],
                [-0.06 * np.sin(0.3 * x1), 0.9, 0.0],
                [
                    (A_val - 0.06 * np.sin(0.3 * x1)) / r_val,
                    (0.05 * A_val * (1 + np.cos(0.1 * x2)) + 0.9 * B_val) / r_val,
                    0.0,
                ],
            ]
        )

        # Jacobian w.r.t noise (2x3)
        Bn = np.array(
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [A_val / r_val, B_val / r_val, 1.0]]
        )

        return An, Bn
