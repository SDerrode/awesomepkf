#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from prg.models.nonLinear.base_model_nonLinear import BaseModelNonLinear
from prg.models.Generate_MatrixCov import generate_block_matrix
from prg.exceptions import NumericalError

__all__ = ["ModelX1Y1_withRetroactions"]


class ModelX1Y1_withRetroactions(BaseModelNonLinear):
    MODEL_NAME: str = "x1_y1_withRetroactions"

    def __init__(self) -> None:
        super().__init__(dim_x=1, dim_y=1, model_type="nonlinear")
        try:
            self.mQ = generate_block_matrix(
                self._randMatrices.rng, self.dim_x, self.dim_y, 0.05
            )
            self.mz0 = self._randMatrices.rng.standard_normal((self.dim_xy, 1))
            self.Pz0 = generate_block_matrix(
                self._randMatrices.rng, self.dim_x, self.dim_y, 0.05
            )
        except (ValueError, np.exceptions.AxisError) as e:
            raise NumericalError(
                f"[{self.MODEL_NAME}] Initialization failed: {e}"
            ) from e
        self.a, self.b, self.c, self.d = 0.50, 30, 0.40, 40

    def _gx(self, x, y, t, u, dt):
        try:
            with np.errstate(all="raise"):
                return self.a * x + self.b * np.tanh(y) + t
        except FloatingPointError as e:
            raise NumericalError(
                f"[{self.MODEL_NAME}] _gx: floating point error at x={x}, y={y}: {e}"
            ) from e

    def _gy(self, x, y, t, u, dt):
        try:
            with np.errstate(all="raise"):
                return self.c * y + self.d * np.sin(x) + u
        except FloatingPointError as e:
            raise NumericalError(
                f"[{self.MODEL_NAME}] _gy: floating point error at x={x}, y={y}: {e}"
            ) from e

    def _g(self, x, y, t, u, dt):
        if __debug__:
            assert x.shape == (1, 1)
            assert y.shape == (1, 1)
            assert t.shape == (1, 1)
            assert u.shape == (1, 1)
            assert isinstance(dt, (float, int))
        try:
            gx_val = self._gx(x, y, t, u, dt)
            gy_val = self._gy(x, y, t, u, dt)
            return np.vstack((gx_val, gy_val))
        except NumericalError:
            raise
        except ValueError as e:
            raise NumericalError(
                f"[{self.MODEL_NAME}] _g: shape mismatch during vstack: {e}"
            ) from e

    def _jacobiens_g(self, x, y, t, u, dt):
        if __debug__:
            assert x.shape == (1, 1)
            assert y.shape == (1, 1)
            assert t.shape == (1, 1)
            assert u.shape == (1, 1)
            assert isinstance(dt, (float, int))
        try:
            with np.errstate(all="raise"):
                An = np.array(
                    [
                        [self.a, self.b / np.cosh(y[0, 0]) ** 2],
                        [self.d * np.cos(x[0, 0]), self.c],
                    ]
                )
                Bn = np.eye(self.dim_xy)
            return An, Bn
        except FloatingPointError as e:
            raise NumericalError(
                f"[{self.MODEL_NAME}] _jacobiens_g: floating point error at x={x}, y={y}: {e}"
            ) from e
        except (IndexError, ValueError) as e:
            raise NumericalError(
                f"[{self.MODEL_NAME}] _jacobiens_g: array construction error: {e}"
            ) from e
