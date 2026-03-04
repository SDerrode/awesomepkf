#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from prg.models.nonLinear.base_model_nonLinear import BaseModelNonLinear
from prg.models.Generate_MatrixCov import generate_block_matrix
from prg.exceptions import NumericalError

__all__ = ["ModelSinus"]


class ModelSinus(BaseModelNonLinear):
    MODEL_NAME: str = "x1_y1_sinus"

    def __init__(self) -> None:
        super().__init__(dim_x=1, dim_y=1, model_type="nonlinear")
        try:
            self.mQ = generate_block_matrix(
                self._randMatrices.rng, self.dim_x, self.dim_y, 0.15
            )
            self.mz0 = np.zeros((self.dim_xy, 1)) + 0.3
            self.Pz0 = generate_block_matrix(
                self._randMatrices.rng, self.dim_x, self.dim_y, 0.05
            )
        except (ValueError, np.exceptions.AxisError) as e:
            raise NumericalError(
                f"[{self.MODEL_NAME}] Initialization failed: {e}"
            ) from e

    def _fx(self, x, t, dt):
        try:
            with np.errstate(all="raise"):
                return 0.8 * x + 0.3 * np.sin(x) + t
        except FloatingPointError as e:
            raise NumericalError(
                f"[{self.MODEL_NAME}] _fx: floating point error at x={x}, t={t}: {e}"
            ) from e

    def _hx(self, x, u, dt):
        try:
            with np.errstate(all="raise"):
                return np.sin(x) + u
        except FloatingPointError as e:
            raise NumericalError(
                f"[{self.MODEL_NAME}] _hx: floating point error at x={x}, u={u}: {e}"
            ) from e

    def _g(self, x, y, t, u, dt):
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

    def _jacobiens_g(self, x, y, t, u, dt):
        if __debug__:
            assert x.shape == (1, 1)
            assert y.shape == (1, 1)
            assert t.shape == (1, 1)
            assert u.shape == (1, 1)
        try:
            with np.errstate(all="raise"):
                x1 = x.flatten()[0]
                t1 = t.flatten()[0]
                A = 0.8 * x1 + 0.3 * np.sin(x1) + t1
                An = np.array(
                    [
                        [0.8 + 0.3 * np.cos(x1), 0.0],
                        [np.cos(A) * (0.8 + 0.3 * np.cos(x1)), 0.0],
                    ]
                )
                Bn = np.array([[1.0, 0.0], [np.cos(A), 1.0]])
            return An, Bn
        except FloatingPointError as e:
            raise NumericalError(
                f"[{self.MODEL_NAME}] _jacobiens_g: floating point error at x={x}, t={t}: {e}"
            ) from e
        except (IndexError, ValueError) as e:
            raise NumericalError(
                f"[{self.MODEL_NAME}] _jacobiens_g: array construction error: {e}"
            ) from e
