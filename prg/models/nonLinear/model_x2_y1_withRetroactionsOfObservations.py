#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from prg.models.nonLinear.base_model_gxgy import BaseModelGxGy
from prg.models.Generate_MatrixCov import generate_block_matrix
from prg.exceptions import NumericalError

__all__ = ["ModelX2Y1_withRetroactionsOfObservations"]


class ModelX2Y1_withRetroactionsOfObservations(BaseModelGxGy):
    MODEL_NAME: str = "x2_y1_withRetroactionsOfObservations"

    def __init__(self):
        super().__init__(dim_x=2, dim_y=1, model_type="nonlinear")
        self.a, self.b, self.c, self.d, self.e, self.f = 1.0, 0.8, 0.05, 0.9, 0.30, 0.6
        self.a = 0.95
        self.b = 0.10
        try:
            Q = np.array([[0.03, 0.0], [0.0, 0.03]])
            R = np.array([[0.03]])
            M = np.zeros((self.dim_x, self.dim_y))
            self.mQ  = np.block([[Q, M], [M.T, R]])
            self.mz0 = self._randMatrices.rng.standard_normal((self.dim_xy, 1))
            self.Pz0 = generate_block_matrix(self._randMatrices.rng, self.dim_x, self.dim_y, 0.05)
        except (ValueError, np.exceptions.AxisError) as e:
            raise NumericalError(f"[{self.MODEL_NAME}] Initialization failed: {e}") from e

    # ------------------------------------------------------------------
    def _gx(self, x, y, t, u, dt):
        if __debug__:
            if x.ndim == 2:
                assert all(a.shape == (self.dim_x, 1) for a in (x, t))
                assert y.shape == (self.dim_y, 1)
            else:
                assert all(a.ndim == 3 and a.shape[1:] == (self.dim_x, 1) for a in (x, t))
                assert y.ndim == 3 and y.shape[1:] == (self.dim_y, 1)
                assert x.shape[0] == y.shape[0] == t.shape[0]
        try:
            with np.errstate(all="raise"):
                if x.ndim == 2:
                    x1, x2 = x[0, 0], x[1, 0]
                    y1 = y[0, 0]
                    t1, t2 = t[0, 0], t[1, 0]
                    return np.array([
                        [self.a * x1 + self.b * x2 + self.c * np.tanh(y1) + t1],
                        [self.d * x2 + self.e * np.sin(y1) + t2],
                    ])
                else:
                    x1, x2 = x[:, 0, 0], x[:, 1, 0]
                    y1 = y[:, 0, 0]
                    t1, t2 = t[:, 0, 0], t[:, 1, 0]
                    out = np.empty_like(x)
                    out[:, 0, 0] = self.a * x1 + self.b * x2 + self.c * np.tanh(y1) + t1
                    out[:, 1, 0] = self.d * x2 + self.e * np.sin(y1) + t2
                    return out
        except FloatingPointError as e:
            raise NumericalError(f"[{self.MODEL_NAME}] _gx: floating point error at x={x}, y={y}, t={t}: {e}") from e
        except (IndexError, ValueError) as e:
            raise NumericalError(f"[{self.MODEL_NAME}] _gx: array access error at x={x}, y={y}, t={t}: {e}") from e

    # ------------------------------------------------------------------
    def _gy(self, x, y, t, u, dt):
        if __debug__:
            if x.ndim == 2:
                assert x.shape == (self.dim_x, 1)
                assert all(a.shape == (self.dim_y, 1) for a in (y, u))
            else:
                assert x.ndim == 3 and x.shape[1:] == (self.dim_x, 1)
                assert all(a.ndim == 3 and a.shape[1:] == (self.dim_y, 1) for a in (y, u))
                assert x.shape[0] == y.shape[0] == u.shape[0]
        try:
            with np.errstate(all="raise"):
                if x.ndim == 2:
                    x1 = x[0, 0]
                    return np.array([[x1**2 / (1.0 + x1**2) + self.f * y[0, 0] + u[0, 0]]])
                else:
                    x1 = x[:, 0, 0]
                    out = np.empty((x.shape[0], self.dim_y, 1))
                    out[:, 0, 0] = x1**2 / (1.0 + x1**2) + self.f * y[:, 0, 0] + u[:, 0, 0]
                    return out
        except FloatingPointError as e:
            raise NumericalError(f"[{self.MODEL_NAME}] _gy: floating point error at x={x}, y={y}, u={u}: {e}") from e
        except (IndexError, ValueError) as e:
            raise NumericalError(f"[{self.MODEL_NAME}] _gy: array access error at x={x}, y={y}, u={u}: {e}") from e

    # ------------------------------------------------------------------
    def _jacobiens_g(self, x, y, t, u, dt):
        if __debug__:
            assert isinstance(dt, (float, int))
            if x.ndim == 2:
                assert all(a.shape == (self.dim_x, 1) for a in (x, t))
                assert all(a.shape == (self.dim_y, 1) for a in (y, u))
            else:
                assert all(a.ndim == 3 and a.shape[1:] == (self.dim_x, 1) for a in (x, t))
                assert all(a.ndim == 3 and a.shape[1:] == (self.dim_y, 1) for a in (y, u))
                assert x.shape[0] == y.shape[0] == t.shape[0] == u.shape[0]
        try:
            with np.errstate(all="raise"):
                if x.ndim == 2:
                    x1 = x[0, 0]
                    y1 = y[0, 0]
                    An = np.array([
                        [self.a, self.b, self.c * (1.0 - np.tanh(y1)**2)],
                        [0.0,   self.d, self.e * np.cos(y1)              ],
                        [2.0 * x1 / (1.0 + x1**2)**2, 0.0, self.f       ],
                    ])
                    Bn = np.eye(self.dim_xy)
                else:
                    N  = x.shape[0]
                    x1 = x[:, 0, 0]
                    y1 = y[:, 0, 0]
                    An = np.zeros((N, 3, 3))
                    An[:, 0, 0] = self.a
                    An[:, 0, 1] = self.b
                    An[:, 0, 2] = self.c * (1.0 - np.tanh(y1)**2)
                    An[:, 1, 1] = self.d
                    An[:, 1, 2] = self.e * np.cos(y1)
                    An[:, 2, 0] = 2.0 * x1 / (1.0 + x1**2)**2
                    An[:, 2, 2] = self.f
                    Bn = np.tile(np.eye(self.dim_xy), (N, 1, 1))
            return An, Bn
        except FloatingPointError as e:
            raise NumericalError(f"[{self.MODEL_NAME}] _jacobiens_g: floating point error at x={x}, y={y}: {e}") from e
        except (IndexError, ValueError) as e:
            raise NumericalError(f"[{self.MODEL_NAME}] _jacobiens_g: array construction error: {e}") from e
