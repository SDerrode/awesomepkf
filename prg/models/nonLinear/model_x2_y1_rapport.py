#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import sympy as sp

from prg.models.nonLinear.base_model_fxhx import BaseModelFxHx
from prg.models.Generate_MatrixCov import generate_block_matrix
from prg.exceptions import NumericalError

__all__ = ["ModelX2Y1Rapport"]


class ModelX2Y1Rapport(BaseModelFxHx):
    """
    Nonlinear model with:
      - 2-dimensional state vector x = [x1, x2]
      - 1-dimensional observation y

    System dynamics:
        f(x) = [
            (1 - kappa_m)*x1 + dt_model*x2 + t1,
            x2 - dt_model*(alpha*sin(x1) + beta*x2) + t2
        ]

    Measurement equation:
        h(x) = x1^2/(1 + x1^2) + gamma*sin(x2) + u
    """

    MODEL_NAME: str = "x2_y1_rapport"

    def __init__(self):
        # ← paramètres AVANT super().__init__() qui appelle _build_symbolic_model()
        self.alpham = 0.5
        self.betam = 0.5
        self.gammam = 0.5
        self.kappa_m = 0.15
        self.dt_model = 0.1

        super().__init__(dim_x=2, dim_y=1, model_type="nonlinear")

        try:
            self.mQ = generate_block_matrix(
                self._randMatrices.rng, self.dim_x, self.dim_y, 0.1
            )
            self.mz0 = self._randMatrices.rng.standard_normal((self.dim_xy, 1))
            self.Pz0 = generate_block_matrix(
                self._randMatrices.rng, self.dim_x, self.dim_y, 0.05
            )
        except (ValueError, np.exceptions.AxisError) as e:
            raise NumericalError(
                f"[{self.MODEL_NAME}] Initialization failed: {e}"
            ) from e

    def symbolic_model(self, sx, st, su):
        x1, x2 = sx[0], sx[1]
        t1, t2 = st[0], st[1]
        u = su[0]

        sfx1 = (1 - self.kappa_m) * x1 + self.dt_model * x2 + t1
        sfx2 = x2 - self.dt_model * (self.alpham * sp.sin(x1) + self.betam * x2) + t2
        shx = x1**2 / (1 + x1**2) + self.gammam * sp.sin(x2) + u

        return sp.Matrix([sfx1, sfx2]), sp.Matrix([shx])

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
    #                 return np.array([
    #                     [(1.0 - self.kappa_m) * x[0, 0] + self.dt_model * x[1, 0] + t[0, 0]],
    #                     [x[1, 0] - self.dt_model * (self.alpham * np.sin(x[0, 0]) + self.betam * x[1, 0]) + t[1, 0]],
    #                 ])
    #             else:
    #                 out = np.empty_like(x)
    #                 out[:, 0, 0] = (1.0 - self.kappa_m) * x[:, 0, 0] + self.dt_model * x[:, 1, 0] + t[:, 0, 0]
    #                 out[:, 1, 0] = x[:, 1, 0] - self.dt_model * (self.alpham * np.sin(x[:, 0, 0]) + self.betam * x[:, 1, 0]) + t[:, 1, 0]
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
    #                 return np.array([[x[0, 0]**2 / (1.0 + x[0, 0]**2) + self.gammam * np.sin(x[1, 0]) + u[0, 0]]])
    #             else:
    #                 out = np.empty((x.shape[0], self.dim_y, 1))
    #                 out[:, 0, 0] = x[:, 0, 0]**2 / (1.0 + x[:, 0, 0]**2) + self.gammam * np.sin(x[:, 1, 0]) + u[:, 0, 0]
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
    #                 A = x1 + dt * x2 + t1
    #                 B = x2 - dt * (self.alpham * np.sin(x1) + self.betam * x2) + t2
    #                 Z = 2 * A / (1.0 + A**2)**2
    #                 W = self.gammam * np.cos(B)
    #                 An = np.array([
    #                     [1.0 - self.kappa_m,                           self.dt_model,                    0.0],
    #                     [-self.alpham * self.dt_model * np.cos(x1),  1.0 - self.betam * self.dt_model, 0.0],
    #                     [Z - self.alpham * dt * np.cos(x1) * W,      Z * dt + (1.0 - self.betam * dt) * W, 0.0],
    #                 ])
    #                 Bn = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [Z, W, 1.0]])
    #             else:
    #                 N  = x.shape[0]
    #                 x1, x2 = x[:, 0, 0], x[:, 1, 0]
    #                 t1, t2 = t[:, 0, 0], t[:, 1, 0]
    #                 A = x1 + dt * x2 + t1
    #                 B = x2 - dt * (self.alpham * np.sin(x1) + self.betam * x2) + t2
    #                 Z = 2 * A / (1.0 + A**2)**2
    #                 W = self.gammam * np.cos(B)
    #                 An = np.zeros((N, 3, 3))
    #                 An[:, 0, 0] = 1.0 - self.kappa_m
    #                 An[:, 0, 1] = self.dt_model
    #                 An[:, 1, 0] = -self.alpham * self.dt_model * np.cos(x1)
    #                 An[:, 1, 1] = 1.0 - self.betam * self.dt_model
    #                 An[:, 2, 0] = Z - self.alpham * dt * np.cos(x1) * W
    #                 An[:, 2, 1] = Z * dt + (1.0 - self.betam * dt) * W
    #                 Bn = np.zeros((N, 3, 3))
    #                 Bn[:, 0, 0] = 1.0
    #                 Bn[:, 1, 1] = 1.0
    #                 Bn[:, 2, 0] = Z
    #                 Bn[:, 2, 1] = W
    #                 Bn[:, 2, 2] = 1.0
    #         return An, Bn
    #     except FloatingPointError as e:
    #         raise NumericalError(f"[{self.MODEL_NAME}] _jacobiens_g: floating point error at x={x}, t={t}: {e}") from e
    #     except (IndexError, ValueError) as e:
    #         raise NumericalError(f"[{self.MODEL_NAME}] _jacobiens_g: array construction error: {e}") from e
