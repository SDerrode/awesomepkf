#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np

from prg.models.nonLinear.base_model_nonLinear import BaseModelNonLinear
from prg.models.Generate_MatrixCov import generate_block_matrix
from prg.utils.numerics import EPS_REL
from prg.exceptions import NumericalError

__all__ = ["ModelExtSaturant"]


class ModelExtSaturant(BaseModelNonLinear):
    """
    Nonlinear model with saturation in state dynamics and logarithmic observation:

    System dynamics:
        f(x) = 0.5 * x + 2 * (1 - exp(-0.1 * x))

    Measurement equation:
        h(x) = log(1 + |x|)

    Includes additive Gaussian process and observation noise.
    """

    MODEL_NAME: str = "x1_y1_ext_saturant"

    def __init__(self) -> None:
        super().__init__(dim_x=1, dim_y=1, model_type="nonlinear")

        try:
            self.mQ = generate_block_matrix(
                self._randMatrices.rng, self.dim_x, self.dim_y, 0.05
            )
            self.mz0 = self._randMatrices.rng.standard_normal((self.dim_xy, 1))
            self.Pz0 = generate_block_matrix(
                self._randMatrices.rng, self.dim_x, self.dim_y, 0.03
            )
        except (ValueError, np.exceptions.AxisError) as e:
            raise NumericalError(
                f"[{self.MODEL_NAME}] Initialization failed: {e}"
            ) from e

    # ------------------------------------------------------------------
    def _fx(self, x: np.ndarray, t: np.ndarray, dt: float) -> np.ndarray:
        """
        State transition function with additive noise and saturation.

        Args:
            x : np.ndarray, shape (1,1) - current state
            t : np.ndarray, shape (1,1) - process noise
            dt: float - timestep

        Returns:
            np.ndarray, shape (1,1) - next state
        """
        try:
            with np.errstate(all="raise"):
                return 0.5 * x + 2.0 * (1.0 - np.exp(-0.1 * x)) + t
        except FloatingPointError as e:
            raise NumericalError(
                f"[{self.MODEL_NAME}] _fx: floating point error at x={x}, t={t}: {e}"
            ) from e

    # ------------------------------------------------------------------
    def _hx(self, x: np.ndarray, u: np.ndarray, dt: float) -> np.ndarray:
        """
        Measurement function with logarithmic observation and additive noise.

        Args:
            x : np.ndarray, shape (1,1) - predicted state
            u : np.ndarray, shape (1,1) - observation noise
            dt: float - timestep

        Returns:
            np.ndarray, shape (1,1) - measurement
        """
        try:
            with np.errstate(all="raise"):
                return np.log(1.0 + np.maximum(np.abs(x), EPS_REL)) + u
        except FloatingPointError as e:
            raise NumericalError(
                f"[{self.MODEL_NAME}] _hx: floating point error at x={x}, u={u}: {e}"
            ) from e

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

        try:
            with np.errstate(all="raise"):
                x1 = x.flatten()[0]
                t1 = t.flatten()[0]

                # État intermédiaire
                A = 0.5 * x1 + 2.0 * (1.0 - np.exp(-0.1 * x1)) + t1

                An = np.array(
                    [
                        [0.5 + 0.2 * np.exp(-0.1 * x1), 0.0],
                        [
                            np.sign(A)
                            * (0.5 + 0.2 * np.exp(-0.1 * x1))
                            / (1.0 + np.abs(A)),
                            0.0,
                        ],
                    ]
                )
                Bn = np.array([[1.0, 0.0], [np.sign(A) / (1.0 + np.abs(A)), 1.0]])

            return An, Bn

        except FloatingPointError as e:
            raise NumericalError(
                f"[{self.MODEL_NAME}] _jacobiens_g: floating point error at x={x}, t={t}: {e}"
            ) from e
        except (IndexError, ValueError) as e:
            raise NumericalError(
                f"[{self.MODEL_NAME}] _jacobiens_g: array construction error: {e}"
            ) from e
