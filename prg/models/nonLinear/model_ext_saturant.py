#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from prg.models.nonLinear.base_model_fxhx import BaseModelFxHx
from prg.models.Generate_MatrixCov import generate_block_matrix
from prg.utils.numerics import EPS_REL
from prg.exceptions import NumericalError

__all__ = ["ModelExtSaturant"]


class ModelExtSaturant(BaseModelFxHx):
    MODEL_NAME: str = "x1_y1_ext_saturant"

    def __init__(self):
        super().__init__(dim_x=1, dim_y=1, model_type="nonlinear")
        try:
            self.mQ  = generate_block_matrix(self._randMatrices.rng, self.dim_x, self.dim_y, 0.05)
            self.mz0 = self._randMatrices.rng.standard_normal((self.dim_xy, 1))
            self.Pz0 = generate_block_matrix(self._randMatrices.rng, self.dim_x, self.dim_y, 0.03)
        except (ValueError, np.exceptions.AxisError) as e:
            raise NumericalError(f"[{self.MODEL_NAME}] Initialization failed: {e}") from e

    # ------------------------------------------------------------------
    def _fx(self, x, t, dt):
        if __debug__:
            if x.ndim == 2:
                assert all(a.shape == (1, 1) for a in (x, t))
            else:
                assert all(a.ndim == 3 and a.shape[1:] == (1, 1) for a in (x, t))
                assert x.shape[0] == t.shape[0]
        try:
            with np.errstate(all="raise"):
                return 0.5 * x + 2.0 * (1.0 - np.exp(-0.1 * x)) + t
        except FloatingPointError as e:
            raise NumericalError(f"[{self.MODEL_NAME}] _fx: floating point error at x={x}, t={t}: {e}") from e

    # ------------------------------------------------------------------
    def _hx(self, x, u, dt):
        if __debug__:
            if x.ndim == 2:
                assert all(a.shape == (1, 1) for a in (x, u))
            else:
                assert all(a.ndim == 3 and a.shape[1:] == (1, 1) for a in (x, u))
                assert x.shape[0] == u.shape[0]
        try:
            with np.errstate(all="raise"):
                return np.log(1.0 + np.maximum(np.abs(x), EPS_REL)) + u
        except FloatingPointError as e:
            raise NumericalError(f"[{self.MODEL_NAME}] _hx: floating point error at x={x}, u={u}: {e}") from e

    # ------------------------------------------------------------------
    def _jacobiens_g(self, x, y, t, u, dt):
        if __debug__:
            if x.ndim == 2:
                assert all(a.shape == (1, 1) for a in (x, t))
            else:
                assert all(a.ndim == 3 and a.shape[1:] == (1, 1) for a in (x, t))
                assert x.shape[0] == t.shape[0]
        try:
            with np.errstate(all="raise"):
                x0 = x[0, 0]    if x.ndim == 2 else x[:, 0, 0]
                t0 = t[0, 0]    if t.ndim == 2 else t[:, 0, 0]
                A  = 0.5 * x0 + 2.0 * (1.0 - np.exp(-0.1 * x0)) + t0
                dA = 0.5 + 0.2 * np.exp(-0.1 * x0)
                sA = np.sign(A) / (1.0 + np.abs(A))
                if x.ndim == 2:
                    An = np.array([[dA, 0.0], [sA * dA, 0.0]])
                    Bn = np.array([[1.0, 0.0], [sA, 1.0]])
                else:
                    N  = x.shape[0]
                    An = np.zeros((N, 2, 2))
                    An[:, 0, 0] = dA
                    An[:, 1, 0] = sA * dA
                    Bn = np.tile(np.array([[1.0, 0.0], [0.0, 1.0]]), (N, 1, 1))
                    Bn[:, 1, 0] = sA
            return An, Bn
        except FloatingPointError as e:
            raise NumericalError(f"[{self.MODEL_NAME}] _jacobiens_g: floating point error at x={x}, t={t}: {e}") from e
        except (IndexError, ValueError) as e:
            raise NumericalError(f"[{self.MODEL_NAME}] _jacobiens_g: array construction error: {e}") from e
