#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import sympy as sp

from prg.models.nonLinear.base_model_fxhx import BaseModelFxHx
from prg.models.Generate_MatrixCov import generate_block_matrix
from prg.exceptions import NumericalError

__all__ = ["ModelX2Y1"]


class ModelX2Y1(BaseModelFxHx):
    """
    Nonlinear model with:
      - 2-dimensional state vector x = [x1, x2]
      - 1-dimensional observation y

    System dynamics:
        f(x, t) = [
            (1 - KAPPA)*x1 + 0.05*x2 + 0.5*sin(0.1*x2) + t1,
            0.9*x2 + 0.2*cos(0.3*x1) + t2
        ]

    Measurement equation:
        h(x, u) = sqrt(x1^2 + x2^2) + u
    """

    MODEL_NAME: str = "x2_y1"
    KAPPA: float = 0.10
    R_MIN: float = 1e-8

    def __init__(self):
        super().__init__(dim_x=2, dim_y=1, model_type="nonlinear")
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
    def symbolic_model(self, sx, st, su):
        x1, x2 = sx[0], sx[1]
        t1, t2 = st[0], st[1]
        u = su[0]

        sfx = sp.Matrix(
            [
                (1 - self.KAPPA) * x1
                + sp.Rational(1, 20) * x2
                + sp.Rational(1, 2) * sp.sin(sp.Rational(1, 10) * x2)
                + t1,
                sp.Rational(9, 10) * x2
                + sp.Rational(1, 5) * sp.cos(sp.Rational(3, 10) * x1)
                + t2,
            ]
        )
        shx = sp.Matrix([sp.sqrt(x1**2 + x2**2) + u])

        return sfx, shx

    # ------------------------------------------------------------------
    def _eval_H(self, x):
        """
        Surcharge pour protéger la singularité de dh/dx = [x1, x2] / sqrt(x1²+x2²)
        à l'origine, en clampant le dénominateur à R_MIN.
        """
        if x.ndim == 2:
            x1, x2 = x[0, 0], x[1, 0]
            r = float(np.maximum(np.sqrt(x1**2 + x2**2), self.R_MIN))
            return np.array([[x1 / r, x2 / r]], dtype=float)  # (1, 2)
        else:
            x1 = x[:, 0, 0]
            x2 = x[:, 1, 0]
            r = np.maximum(np.sqrt(x1**2 + x2**2), self.R_MIN)
            N = x.shape[0]
            out = np.empty((N, self.dim_y, self.dim_x))
            out[:, 0, 0] = x1 / r
            out[:, 0, 1] = x2 / r
            return out

    # # ------------------------------------------------------------------
    # def _fx(self, x, t, dt):
    #     if __debug__:
    #         if x.ndim == 2:
    #             assert all(a.shape == (self.dim_x, 1) for a in (x, t))
    #         else:
    #             assert all(a.ndim == 3 and a.shape[1:] == (self.dim_x, 1) for a in (x, t))
    #             assert x.shape[0] == t.shape[0]
    #     try:
    #         with np.errstate(all="raise"):
    #             if x.ndim == 2:
    #                 x1, x2 = x[0, 0], x[1, 0]
    #                 t1, t2 = t[0, 0], t[1, 0]
    #                 return np.array([
    #                     [(1.0 - self.KAPPA) * x1 + 0.05 * x2 + 0.5 * np.sin(0.1 * x2) + t1],
    #                     [0.9 * x2 + 0.2 * np.cos(0.3 * x1) + t2],
    #                 ])
    #             else:
    #                 x1, x2 = x[:, 0, 0], x[:, 1, 0]
    #                 t1, t2 = t[:, 0, 0], t[:, 1, 0]
    #                 out = np.empty_like(x)
    #                 out[:, 0, 0] = (1.0 - self.KAPPA) * x1 + 0.05 * x2 + 0.5 * np.sin(0.1 * x2) + t1
    #                 out[:, 1, 0] = 0.9 * x2 + 0.2 * np.cos(0.3 * x1) + t2
    #                 return out
    #     except FloatingPointError as e:
    #         raise NumericalError(f"[{self.MODEL_NAME}] _fx: floating point error at x={x}, t={t}: {e}") from e
    #     except (IndexError, ValueError) as e:
    #         raise NumericalError(f"[{self.MODEL_NAME}] _fx: array access error at x={x}, t={t}: {e}") from e

    # # ------------------------------------------------------------------
    # def _hx(self, x, u, dt):
    #     if __debug__:
    #         if x.ndim == 2:
    #             assert x.shape == (self.dim_x, 1)
    #             assert u.shape == (self.dim_y, 1)
    #         else:
    #             assert x.ndim == 3 and x.shape[1:] == (self.dim_x, 1)
    #             assert u.ndim == 3 and u.shape[1:] == (self.dim_y, 1)
    #             assert x.shape[0] == u.shape[0]
    #     try:
    #         with np.errstate(all="raise"):
    #             if x.ndim == 2:
    #                 return np.array([[np.sqrt(x[0, 0]**2 + x[1, 0]**2) + u[0, 0]]])
    #             else:
    #                 out = np.empty((x.shape[0], self.dim_y, 1))
    #                 out[:, 0, 0] = np.sqrt(x[:, 0, 0]**2 + x[:, 1, 0]**2) + u[:, 0, 0]
    #                 return out
    #     except FloatingPointError as e:
    #         raise NumericalError(f"[{self.MODEL_NAME}] _hx: floating point error at x={x}, u={u}: {e}") from e
    #     except (IndexError, ValueError) as e:
    #         raise NumericalError(f"[{self.MODEL_NAME}] _hx: array access error at x={x}, u={u}: {e}") from e

    # # ------------------------------------------------------------------
    # def _jacobiens_g(self, x, y, t, u, dt):
    #     if __debug__:
    #         assert isinstance(dt, (float, int))
    #         if x.ndim == 2:
    #             assert all(a.shape == (self.dim_x, 1) for a in (x, t))
    #             assert all(a.shape == (self.dim_y, 1) for a in (y, u))
    #         else:
    #             assert all(a.ndim == 3 and a.shape[1:] == (self.dim_x, 1) for a in (x, t))
    #             assert all(a.ndim == 3 and a.shape[1:] == (self.dim_y, 1) for a in (y, u))
    #             assert x.shape[0] == y.shape[0] == t.shape[0] == u.shape[0]
    #     try:
    #         with np.errstate(all="raise"):
    #             if x.ndim == 2:
    #                 x1, x2 = x[0, 0], x[1, 0]
    #                 t1, t2 = t[0, 0], t[1, 0]
    #                 A = (1.0 - self.KAPPA) * x1 + 0.05 * x2 + 0.5 * np.sin(0.1 * x2) + t1
    #                 B = 0.9 * x2 + 0.2 * np.cos(0.3 * x1) + t2
    #                 r = float(np.maximum(np.sqrt(A**2 + B**2), self.R_MIN))
    #                 An = np.array([
    #                     [1.0 - self.KAPPA,          0.05 * (1 + np.cos(0.1 * x2)),                        0.0],
    #                     [-0.06 * np.sin(0.3 * x1),  0.9,                                                   0.0],
    #                     [(A - 0.06 * np.sin(0.3 * x1)) / r, (0.05 * A * (1 + np.cos(0.1 * x2)) + 0.9 * B) / r, 0.0],
    #                 ])
    #                 Bn = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [A / r, B / r, 1.0]])
    #             else:
    #                 x1, x2 = x[:, 0, 0], x[:, 1, 0]
    #                 t1, t2 = t[:, 0, 0], t[:, 1, 0]
    #                 N = x.shape[0]
    #                 A = (1.0 - self.KAPPA) * x1 + 0.05 * x2 + 0.5 * np.sin(0.1 * x2) + t1
    #                 B = 0.9 * x2 + 0.2 * np.cos(0.3 * x1) + t2
    #                 r = np.maximum(np.sqrt(A**2 + B**2), self.R_MIN)
    #                 An = np.zeros((N, 3, 3))
    #                 An[:, 0, 0] = 1.0 - self.KAPPA
    #                 An[:, 0, 1] = 0.05 * (1 + np.cos(0.1 * x2))
    #                 An[:, 1, 0] = -0.06 * np.sin(0.3 * x1)
    #                 An[:, 1, 1] = 0.9
    #                 An[:, 2, 0] = (A - 0.06 * np.sin(0.3 * x1)) / r
    #                 An[:, 2, 1] = (0.05 * A * (1 + np.cos(0.1 * x2)) + 0.9 * B) / r
    #                 Bn = np.zeros((N, 3, 3))
    #                 Bn[:, 0, 0] = 1.0
    #                 Bn[:, 1, 1] = 1.0
    #                 Bn[:, 2, 0] = A / r
    #                 Bn[:, 2, 1] = B / r
    #                 Bn[:, 2, 2] = 1.0
    #         return An, Bn
    #     except FloatingPointError as e:
    #         raise NumericalError(f"[{self.MODEL_NAME}] _jacobiens_g: floating point error at x={x}, t={t}: {e}") from e
    #     except (IndexError, ValueError) as e:
    #         raise NumericalError(f"[{self.MODEL_NAME}] _jacobiens_g: array construction error: {e}") from e
