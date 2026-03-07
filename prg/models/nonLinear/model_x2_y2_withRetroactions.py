#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np

from prg.models.nonLinear.base_model_nonLinear import BaseModelNonLinear
from prg.exceptions import NumericalError

__all__ = ["ModelX2Y2_withRetroactions"]


class ModelX2Y2_withRetroactions(BaseModelNonLinear):
    """
    Nonlinear model with retro-actions of observations and of states.
    The model includes additive Gaussian process and observation noises.
    """

    MODEL_NAME: str = "x2_y2_withRetroactions"

    # Terme de rappel sur x1 pour éviter la dérive (intégrateur pur sinon).
    # kappa=0.10 donne un rayon spectral de 0.9998 (limite), kappa=0.15
    # donne 0.978 — marge confortable même quand tanh(y1) → 1.
    KAPPA: float = 0.15

    def __init__(self) -> None:
        super().__init__(dim_x=2, dim_y=2, model_type="nonlinear")

        try:
            Q = np.array([[0.08, 0.01], [0.01, 0.05]])
            R = np.array([[0.1, 0.0], [0.0, 0.05]])
            M = np.array([[0.01, 0.0], [0.0, 0.01]])
            self.mQ = np.block([[Q, M], [M.T, R]])
            self.mz0 = np.zeros((self.dim_xy, 1))
            self.Pz0 = np.eye(self.dim_xy) / 20.0
        except (ValueError, np.exceptions.AxisError) as e:
            raise NumericalError(
                f"[{self.MODEL_NAME}] Initialization failed: {e}"
            ) from e

    # ------------------------------------------------------------------
    def _gx(self, x, y, t, u, dt):
        if __debug__:
            if x.ndim == 2:
                assert all(a.shape == (self.dim_x, 1) for a in (x, t))
                assert y.shape == (self.dim_y, 1)
            else:
                assert all(
                    a.ndim == 3 and a.shape[1:] == (self.dim_x, 1) for a in (x, t)
                )
                assert y.ndim == 3 and y.shape[1:] == (self.dim_y, 1)
                assert x.shape[0] == y.shape[0] == t.shape[0]

        try:
            with np.errstate(all="raise"):
                if x.ndim == 2:
                    x1, x2 = x[0, 0], x[1, 0]
                    y1 = y[0, 0]
                    t1, t2 = t[0, 0], t[1, 0]
                    return np.array(
                        [
                            [(1.0 - self.KAPPA) * x1 + 0.1 * x2 * np.tanh(y1) + t1],
                            [0.9 * x2 + 0.1 * np.sin(x1) + t2],
                        ]
                    )
                else:
                    x1, x2 = x[:, 0, 0], x[:, 1, 0]
                    y1 = y[:, 0, 0]
                    t1, t2 = t[:, 0, 0], t[:, 1, 0]
                    out = np.empty_like(x)
                    out[:, 0, 0] = (1.0 - self.KAPPA) * x1 + 0.1 * x2 * np.tanh(y1) + t1
                    out[:, 1, 0] = 0.9 * x2 + 0.1 * np.sin(x1) + t2
                    return out

        except FloatingPointError as e:
            raise NumericalError(
                f"[{self.MODEL_NAME}] _gx: floating point error at x={x}, y={y}, t={t}: {e}"
            ) from e
        except (IndexError, ValueError) as e:
            raise NumericalError(
                f"[{self.MODEL_NAME}] _gx: array access error at x={x}, y={y}, t={t}: {e}"
            ) from e

    # ------------------------------------------------------------------
    def _gy(self, x, y, t, u, dt):
        if __debug__:
            if x.ndim == 2:
                assert x.shape == (self.dim_x, 1)
                assert all(a.shape == (self.dim_y, 1) for a in (y, u))
            else:
                assert x.ndim == 3 and x.shape[1:] == (self.dim_x, 1)
                assert all(
                    a.ndim == 3 and a.shape[1:] == (self.dim_y, 1) for a in (y, u)
                )
                assert x.shape[0] == y.shape[0] == u.shape[0]

        try:
            with np.errstate(all="raise"):
                if x.ndim == 2:
                    x1, x2 = x[0, 0], x[1, 0]
                    y1, y2 = y[0, 0], y[1, 0]
                    u1, u2 = u[0, 0], u[1, 0]
                    return np.array([[x1 - 0.3 * y2 + u1], [x2 + 0.3 * y1 + u2]])
                else:
                    x1, x2 = x[:, 0, 0], x[:, 1, 0]
                    y1, y2 = y[:, 0, 0], y[:, 1, 0]
                    u1, u2 = u[:, 0, 0], u[:, 1, 0]
                    out = np.empty_like(y)
                    out[:, 0, 0] = x1 - 0.3 * y2 + u1
                    out[:, 1, 0] = x2 + 0.3 * y1 + u2
                    return out

        except FloatingPointError as e:
            raise NumericalError(
                f"[{self.MODEL_NAME}] _gy: floating point error at x={x}, y={y}, u={u}: {e}"
            ) from e
        except (IndexError, ValueError) as e:
            raise NumericalError(
                f"[{self.MODEL_NAME}] _gy: array access error at x={x}, y={y}, u={u}: {e}"
            ) from e

    # ------------------------------------------------------------------
    def _g(self, x, y, t, u, dt):
        if __debug__:
            assert isinstance(dt, (float, int))
            if x.ndim == 2:
                assert all(a.shape == (self.dim_x, 1) for a in (x, t))
                assert all(a.shape == (self.dim_y, 1) for a in (y, u))
            else:
                assert all(
                    a.ndim == 3 and a.shape[1:] == (self.dim_x, 1) for a in (x, t)
                )
                assert all(
                    a.ndim == 3 and a.shape[1:] == (self.dim_y, 1) for a in (y, u)
                )
                assert x.shape[0] == y.shape[0] == t.shape[0] == u.shape[0]

        try:
            gx_val = self._gx(x, y, t, u, dt)
            gy_val = self._gy(x, y, t, u, dt)
            if x.ndim == 2:
                return np.vstack((gx_val, gy_val))
            else:
                return np.concatenate((gx_val, gy_val), axis=1)
        except NumericalError:
            raise
        except ValueError as e:
            raise NumericalError(
                f"[{self.MODEL_NAME}] _g: shape mismatch during stack: {e}"
            ) from e

    # ------------------------------------------------------------------
    def _jacobiens_g(self, x, y, t, u, dt):
        if __debug__:
            assert isinstance(dt, (float, int))
            if x.ndim == 2:
                assert all(a.shape == (self.dim_x, 1) for a in (x, t))
                assert all(a.shape == (self.dim_y, 1) for a in (y, u))
            else:
                assert all(
                    a.ndim == 3 and a.shape[1:] == (self.dim_x, 1) for a in (x, t)
                )
                assert all(
                    a.ndim == 3 and a.shape[1:] == (self.dim_y, 1) for a in (y, u)
                )
                assert x.shape[0] == y.shape[0] == t.shape[0] == u.shape[0]

        try:
            with np.errstate(all="raise"):
                if x.ndim == 2:
                    x1, x2 = x[0, 0], x[1, 0]
                    y1 = y[0, 0]

                    An = np.array(
                        [
                            [
                                1.0 - self.KAPPA,
                                0.1 * np.tanh(y1),
                                0.1 * x2 * (1.0 - np.tanh(y1) ** 2),
                                0.0,
                            ],
                            [0.1 * np.cos(x1), 0.9, 0.0, 0.0],
                            [1.0, 0.0, 0.0, -0.3],
                            [0.0, 1.0, 0.3, 0.0],
                        ]
                    )
                    Bn = np.eye(self.dim_xy)

                else:
                    N = x.shape[0]
                    x1, x2 = x[:, 0, 0], x[:, 1, 0]
                    y1 = y[:, 0, 0]

                    An = np.zeros((N, 4, 4))
                    An[:, 0, 0] = 1.0 - self.KAPPA
                    An[:, 0, 1] = 0.1 * np.tanh(y1)
                    An[:, 0, 2] = 0.1 * x2 * (1.0 - np.tanh(y1) ** 2)
                    An[:, 1, 0] = 0.1 * np.cos(x1)
                    An[:, 1, 1] = 0.9
                    An[:, 2, 0] = 1.0
                    An[:, 2, 3] = -0.3
                    An[:, 3, 1] = 1.0
                    An[:, 3, 2] = 0.3

                    Bn = np.tile(np.eye(self.dim_xy), (N, 1, 1))

            return An, Bn

        except FloatingPointError as e:
            raise NumericalError(
                f"[{self.MODEL_NAME}] _jacobiens_g: floating point error at x={x}, y={y}: {e}"
            ) from e
        except (IndexError, ValueError) as e:
            raise NumericalError(
                f"[{self.MODEL_NAME}] _jacobiens_g: array construction error: {e}"
            ) from e
