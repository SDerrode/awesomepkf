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

    def _fx(self, x, t, dt):
        try:
            ax = self.mod._gx(
                x[: self.dim_x - self.dim_y],
                x[self.dim_x - self.dim_y :],
                t[: self.dim_x - self.dim_y],
                t[self.dim_x - self.dim_y :],
                dt,
            )
            ay = self.mod._gy(
                x[: self.dim_x - self.dim_y],
                x[self.dim_x - self.dim_y :],
                t[: self.dim_x - self.dim_y],
                t[self.dim_x - self.dim_y :],
                dt,
            )
            return np.block([[ax], [ay]])
        except NumericalError:
            raise
        except (ValueError, IndexError) as e:
            raise NumericalError(f"[{self.MODEL_NAME}] _fx: error: {e}") from e

    def _hx(self, x, u, dt):
        try:
            return x[-1].reshape(-1, 1)
        except (IndexError, ValueError) as e:
            raise NumericalError(f"[{self.MODEL_NAME}] _hx: error at x={x}: {e}") from e

    def _g(self, x, y, t, u, dt):
        if __debug__:
            assert x.shape == (2, 1), f"x must be (2,1), got {x.shape}"
            assert y.shape == (1, 1), f"y must be (1,1), got {y.shape}"
            assert t.shape == (2, 1), f"t must be (2,1), got {t.shape}"
            assert u.shape == (1, 1), f"u must be (1,1), got {u.shape}"
            assert isinstance(dt, (float, int)), "dt must be a float"
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
            assert x.shape == (2, 1), f"x must be (2,1), got {x.shape}"
            assert y.shape == (1, 1), f"y must be (1,1), got {y.shape}"
            assert t.shape == (2, 1), f"t must be (2,1), got {t.shape}"
            assert u.shape == (1, 1), f"u must be (1,1), got {u.shape}"
            assert isinstance(dt, (float, int)), "dt must be a float"
        try:
            with np.errstate(all="raise"):
                x1, x2 = x.flatten()
                An = np.array(
                    [
                        [self.a, self.b * (1.0 - np.tanh(x2) ** 2.0), 0.0],
                        [self.d * np.cos(x1), self.c, 0.0],
                        [self.d * np.cos(x1), self.c, 0.0],
                    ]
                )
                Bn = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 1.0, 0.0]])
            return An, Bn
        except FloatingPointError as e:
            raise NumericalError(
                f"[{self.MODEL_NAME}] _jacobiens_g: floating point error at x={x}: {e}"
            ) from e
        except (IndexError, ValueError) as e:
            raise NumericalError(
                f"[{self.MODEL_NAME}] _jacobiens_g: array construction error: {e}"
            ) from e
