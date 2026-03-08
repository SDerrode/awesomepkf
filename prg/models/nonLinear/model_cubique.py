#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import sympy as sp

from prg.models.nonLinear.base_model_fxhx import BaseModelFxHx
from prg.models.Generate_MatrixCov import generate_block_matrix
from prg.exceptions import NumericalError

__all__ = ["ModelCubique"]


class ModelCubique(BaseModelFxHx):
    MODEL_NAME: str = "x1_y1_cubique"

    def __init__(self):
        super().__init__(dim_x=1, dim_y=1, model_type="nonlinear")
        try:
            self.mQ = generate_block_matrix(
                self._randMatrices.rng, self.dim_x, self.dim_y, 0.30
            )
            self.mz0 = self._randMatrices.rng.standard_normal((self.dim_xy, 1))
            self.Pz0 = generate_block_matrix(
                self._randMatrices.rng, self.dim_x, self.dim_y, 0.30
            )
        except (ValueError, np.exceptions.AxisError) as e:
            raise NumericalError(
                f"[{self.MODEL_NAME}] Initialization failed: {e}"
            ) from e

    # ------------------------------------------------------------------
    def symbolic_model(self, x, t, u):

        try:
            fx = 0.9 * x - 0.6 * x**3 + t
            hx = x + u

            return fx, hx

        except Exception as e:
            import traceback

            print("[ERREUR] dans symbolic_model:")
            print(traceback.format_exc())
            return None, None

    # ------------------------------------------------------------------
    # def _fx(self, x, t, dt):
    #     if __debug__:
    #         if x.ndim == 2:
    #             assert all(a.shape == (1, 1) for a in (x, t))
    #         else:
    #             assert all(a.ndim == 3 and a.shape[1:] == (1, 1) for a in (x, t))
    #             assert x.shape[0] == t.shape[0]
    #     try:
    #         with np.errstate(all="raise"):
    #             return 0.9 * x - 0.6 * x**3 + t
    #     except FloatingPointError as e:
    #         raise NumericalError(
    #             f"[{self.MODEL_NAME}] _fx: floating point error at x={x}, t={t}: {e}"
    #         ) from e

    # # ------------------------------------------------------------------
    # def _hx(self, x, u, dt):
    #     if __debug__:
    #         if x.ndim == 2:
    #             assert all(a.shape == (1, 1) for a in (x, u))
    #         else:
    #             assert all(a.ndim == 3 and a.shape[1:] == (1, 1) for a in (x, u))
    #             assert x.shape[0] == u.shape[0]
    #     try:
    #         with np.errstate(all="raise"):
    #             return x + u
    #     except FloatingPointError as e:
    #         raise NumericalError(
    #             f"[{self.MODEL_NAME}] _hx: floating point error at x={x}, u={u}: {e}"
    #         ) from e

    # ------------------------------------------------------------------
    # def _jacobiens_g(self, x, y, t, u, dt):
    #     if __debug__:
    #         if x.ndim == 2:
    #             assert x.shape == (1, 1)
    #         else:
    #             assert x.ndim == 3 and x.shape[1:] == (1, 1)
    #     try:
    #         with np.errstate(all="raise"):
    #             if x.ndim == 2:
    #                 dfdx = 0.9 - 1.8 * x[0, 0] ** 2
    #                 An = np.array([[dfdx, 0.0], [dfdx, 0.0]])
    #                 Bn = np.array([[1.0, 0.0], [1.0, 1.0]])
    #             else:
    #                 N = x.shape[0]
    #                 dfdx = 0.9 - 1.8 * x[:, 0, 0] ** 2
    #                 An = np.zeros((N, 2, 2))
    #                 An[:, 0, 0] = dfdx
    #                 An[:, 1, 0] = dfdx
    #                 Bn = np.tile(np.array([[1.0, 0.0], [1.0, 1.0]]), (N, 1, 1))
    #         return An, Bn
    #     except FloatingPointError as e:
    #         raise NumericalError(
    #             f"[{self.MODEL_NAME}] _jacobiens_g: floating point error at x={x}: {e}"
    #         ) from e
    #     except (IndexError, ValueError) as e:
    #         raise NumericalError(
    #             f"[{self.MODEL_NAME}] _jacobiens_g: array construction error: {e}"
    #         ) from e
