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
                return 0.05 * x + 2.0 * np.sin(x) + t
        except FloatingPointError as e:
            raise NumericalError(
                f"[{self.MODEL_NAME}] _fx: floating point error at x={x}, t={t}: {e}"
            ) from e

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
                return 1.5 * np.sin(x) + u
        except FloatingPointError as e:
            raise NumericalError(
                f"[{self.MODEL_NAME}] _hx: floating point error at x={x}, u={u}: {e}"
            ) from e

    # ------------------------------------------------------------------
    def _g(self, x, y, t, u, dt):
        if __debug__:
            if x.ndim == 2:
                assert all(a.shape == (1, 1) for a in (x, y, t, u))
            else:
                assert all(a.ndim == 3 and a.shape[1:] == (1, 1) for a in (x, y, t, u))
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
            if x.ndim == 2:
                assert all(a.shape == (1, 1) for a in (x, t))
            else:
                assert all(a.ndim == 3 and a.shape[1:] == (1, 1) for a in (x, t))
                assert x.shape[0] == t.shape[0]

        try:
            with np.errstate(all="raise"):
                x1 = x[0, 0] if x.ndim == 2 else x[:, 0, 0]
                t1 = t[0, 0] if t.ndim == 2 else t[:, 0, 0]

                A = 0.05 * x1 + 2.0 * np.sin(x1) + t1
                dfdx = 0.05 + 2.0 * np.cos(x1)
                c = 1.5 * np.cos(A)

                if x.ndim == 2:
                    An = np.array([[dfdx, 0.0], [c * dfdx, 0.0]])
                    Bn = np.array([[1.0, 0.0], [c, 1.0]])
                else:
                    N = x.shape[0]
                    An = np.zeros((N, 2, 2))
                    An[:, 0, 0] = dfdx
                    An[:, 1, 0] = c * dfdx

                    Bn = np.tile(np.array([[1.0, 0.0], [0.0, 1.0]]), (N, 1, 1))
                    Bn[:, 1, 0] = c

            return An, Bn

        except FloatingPointError as e:
            raise NumericalError(
                f"[{self.MODEL_NAME}] _jacobiens_g: floating point error at x={x}, t={t}: {e}"
            ) from e
        except (IndexError, ValueError) as e:
            raise NumericalError(
                f"[{self.MODEL_NAME}] _jacobiens_g: array construction error: {e}"
            ) from e
