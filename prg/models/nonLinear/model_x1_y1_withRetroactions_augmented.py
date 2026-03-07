#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from prg.models.nonLinear.base_model_nonLinear import BaseModelNonLinear
from prg.models.nonLinear.model_x1_y1_withRetroactions import ModelX1Y1_withRetroactions
from prg.exceptions import NumericalError

__all__ = ["ModelX1Y1_withRetroactions_augmented"]


class ModelX1Y1_withRetroactions_augmented(BaseModelNonLinear):
    MODEL_NAME: str = "x1_y1_withRetroactions_augmented"

    def __init__(self) -> None:
        super().__init__(dim_x=2, dim_y=1, model_type="nonlinear", augmented=True)
        # NumericalError peut remonter depuis __init__ du modèle non augmenté — on laisse propager
        self.mod = ModelX1Y1_withRetroactions()

        try:
            dim_x, dim_y, dim_xy = self.mod.dim_x, self.mod.dim_y, self.mod.dim_xy
            self.mQ = np.zeros((self.dim_xy, self.dim_xy))
            self.mQ[: self.dim_x, : self.dim_x] = self.mod.mQ

            self.mz0 = np.zeros((dim_xy + dim_y, 1))
            self.mz0[0:dim_xy] = self.mod.mz0
            self.mz0[dim_xy : dim_xy + dim_y] = self.mz0[dim_xy - dim_y : dim_xy]

            self.Pz0 = np.zeros((dim_xy + dim_y, dim_xy + dim_y))
            self.Pz0[0:dim_xy, 0:dim_xy] = self.mod.Pz0
            self.Pz0[dim_xy : dim_xy + dim_y, :] = self.Pz0[dim_xy - dim_y : dim_xy, :]
            self.Pz0[:, dim_xy : dim_xy + dim_y] = self.Pz0[:, dim_xy - dim_y : dim_xy]
        except (ValueError, IndexError) as e:
            raise NumericalError(
                f"[{self.MODEL_NAME}] Initialization failed: {e}"
            ) from e

        self.a, self.b, self.c, self.d = self.mod.a, self.mod.b, self.mod.c, self.mod.d

    # ------------------------------------------------------------------
    def _fx(self, x, t, dt):
        if __debug__:
            if x.ndim == 2:
                assert x.shape == (self.dim_x, 1)
                assert t.shape == (self.dim_x, 1)
            else:
                assert x.ndim == 3 and x.shape[1:] == (self.dim_x, 1)
                assert t.ndim == 3 and t.shape[1:] == (self.dim_x, 1)
                assert x.shape[0] == t.shape[0]

        try:
            split = self.dim_x - self.dim_y
            axis = 1 if x.ndim == 3 else 0

            xA, xB = x[:split] if x.ndim == 2 else x[:, :split], (
                x[split:] if x.ndim == 2 else x[:, split:]
            )
            tA, tB = t[:split] if t.ndim == 2 else t[:, :split], (
                t[split:] if t.ndim == 2 else t[:, split:]
            )

            ax = self.mod._gx(xA, xB, tA, tB, dt)
            ay = self.mod._gy(xA, xB, tA, tB, dt)

            if x.ndim == 2:
                return np.block([[ax], [ay]])
            else:
                return np.concatenate((ax, ay), axis=1)

        except NumericalError:
            raise
        except (ValueError, IndexError) as e:
            raise NumericalError(f"[{self.MODEL_NAME}] _fx: error: {e}") from e

    # ------------------------------------------------------------------
    def _hx(self, x, u, dt):
        if __debug__:
            if x.ndim == 2:
                assert x.shape == (self.dim_x, 1)
            else:
                assert x.ndim == 3 and x.shape[1:] == (self.dim_x, 1)

        try:
            if x.ndim == 2:
                return x[-1].reshape(-1, 1)
            else:
                return x[:, -1:, :]  # (N, 1, 1)
        except (IndexError, ValueError) as e:
            raise NumericalError(f"[{self.MODEL_NAME}] _hx: error at x={x}: {e}") from e

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
            fx_val = self._fx(x, t, dt)
            hx_val = self._hx(fx_val, u, dt)
            if x.ndim == 2:
                return np.vstack((fx_val, hx_val))
            else:
                return np.concatenate((fx_val, hx_val), axis=1)
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

                    An = np.array(
                        [
                            [self.a, self.b * (1.0 - np.tanh(x2) ** 2), 0.0],
                            [self.d * np.cos(x1), self.c, 0.0],
                            [self.d * np.cos(x1), self.c, 0.0],
                        ]
                    )
                    Bn = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 1.0, 0.0]])
                else:
                    N = x.shape[0]
                    x1 = x[:, 0, 0]  # (N,)
                    x2 = x[:, 1, 0]  # (N,)

                    An = np.zeros((N, 3, 3))
                    An[:, 0, 0] = self.a
                    An[:, 0, 1] = self.b * (1.0 - np.tanh(x2) ** 2)
                    An[:, 1, 0] = self.d * np.cos(x1)
                    An[:, 1, 1] = self.c
                    An[:, 2, 0] = self.d * np.cos(x1)
                    An[:, 2, 1] = self.c

                    Bn = np.tile(
                        np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 1.0, 0.0]]),
                        (N, 1, 1),
                    )

            return An, Bn

        except FloatingPointError as e:
            raise NumericalError(
                f"[{self.MODEL_NAME}] _jacobiens_g: floating point error at x={x}: {e}"
            ) from e
        except (IndexError, ValueError) as e:
            raise NumericalError(
                f"[{self.MODEL_NAME}] _jacobiens_g: array construction error: {e}"
            ) from e
