#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import sympy as sp

from prg.models.nonLinear.base_model_gxgy import BaseModelGxGy
from prg.models.Generate_MatrixCov import generate_block_matrix
from prg.exceptions import NumericalError

__all__ = ["ModelX1Y1_withRetroactions"]


class ModelX1Y1_withRetroactions(BaseModelGxGy):
    """
    Nonlinear model with full retroaction (dim_x=1, dim_y=1).

    State transition (depends on both x and y) :
        gx(x, y, t) = a*x + b*tanh(y) + t

    Observation (depends on both x and y) :
        gy(x, y, u) = c*y + d*sin(x) + u

    Jacobians (directs, pas de chain rule) :

        An = dg/dz = [ a           b/cosh²(y) ]
                     [ d*cos(x)    c           ]

        Bn = dg/dn = I_2
    """

    MODEL_NAME: str = "x1_y1_withRetroactions"

    def __init__(self):
        # ← paramètres AVANT super().__init__() qui appelle _build_symbolic_model()
        self.a = 0.50
        self.b = 30
        self.c = 0.40
        self.d = 40

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

    # ------------------------------------------------------------------
    def symbolic_model(self, sx, sy, st, su):
        x, y, t, u = sx[0], sy[0], st[0], su[0]

        sgx = sp.Matrix([self.a * x + self.b * sp.tanh(y) + t])
        sgy = sp.Matrix([self.c * y + self.d * sp.sin(x) + u])

        return sgx, sgy

    # # ------------------------------------------------------------------
    # def _gx(self, x, y, t, u, dt):
    #     if __debug__:
    #         if x.ndim == 2:
    #             assert all(a.shape == (1, 1) for a in (x, y, t))
    #         else:
    #             assert all(a.ndim == 3 and a.shape[1:] == (1, 1) for a in (x, y, t))
    #             assert x.shape[0] == y.shape[0] == t.shape[0]
    #     try:
    #         with np.errstate(all="raise"):
    #             return self.a * x + self.b * np.tanh(y) + t
    #     except FloatingPointError as e:
    #         raise NumericalError(
    #             f"[{self.MODEL_NAME}] _gx: floating point error at x={x}, y={y}: {e}"
    #         ) from e

    # # ------------------------------------------------------------------
    # def _gy(self, x, y, t, u, dt):
    #     if __debug__:
    #         if x.ndim == 2:
    #             assert all(a.shape == (1, 1) for a in (x, y, u))
    #         else:
    #             assert all(a.ndim == 3 and a.shape[1:] == (1, 1) for a in (x, y, u))
    #             assert x.shape[0] == y.shape[0] == u.shape[0]
    #     try:
    #         with np.errstate(all="raise"):
    #             return self.c * y + self.d * np.sin(x) + u
    #     except FloatingPointError as e:
    #         raise NumericalError(
    #             f"[{self.MODEL_NAME}] _gy: floating point error at x={x}, y={y}: {e}"
    #         ) from e

    # # ------------------------------------------------------------------
    # def _jacobiens_g(self, x, y, t, u, dt):
    #     if __debug__:
    #         if x.ndim == 2:
    #             assert all(a.shape == (1, 1) for a in (x, y, t, u))
    #         else:
    #             assert all(a.ndim == 3 and a.shape[1:] == (1, 1) for a in (x, y, t, u))
    #             assert x.shape[0] == y.shape[0] == t.shape[0] == u.shape[0]
    #         assert isinstance(dt, (float, int))
    #     try:
    #         with np.errstate(all="raise"):
    #             if x.ndim == 2:
    #                 An = np.array(
    #                     [
    #                         [self.a, self.b / np.cosh(y[0, 0]) ** 2],
    #                         [self.d * np.cos(x[0, 0]), self.c],
    #                     ]
    #                 )
    #                 Bn = np.eye(self.dim_xy)
    #             else:
    #                 N = x.shape[0]
    #                 An = np.zeros((N, 2, 2))
    #                 An[:, 0, 0] = self.a
    #                 An[:, 0, 1] = self.b / np.cosh(y[:, 0, 0]) ** 2
    #                 An[:, 1, 0] = self.d * np.cos(x[:, 0, 0])
    #                 An[:, 1, 1] = self.c
    #                 Bn = np.tile(np.eye(self.dim_xy), (N, 1, 1))
    #         return An, Bn
    #     except FloatingPointError as e:
    #         raise NumericalError(
    #             f"[{self.MODEL_NAME}] _jacobiens_g: floating point error at x={x}, y={y}: {e}"
    #         ) from e
    #     except (IndexError, ValueError) as e:
    #         raise NumericalError(
    #             f"[{self.MODEL_NAME}] _jacobiens_g: array construction error: {e}"
    #         ) from e
