#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np

from prg.models.nonLinear.base_model_nonLinear import BaseModelNonLinear
from prg.models.Generate_MatrixCov import generate_block_matrix
from prg.exceptions import NumericalError

__all__ = ["ModelX2Y1"]


class ModelX2Y1(BaseModelNonLinear):
    """
    Nonlinear model with:
      - 2-dimensional state vector x = [x1, x2]
      - 1-dimensional observation y

    System dynamics (nonlinear):
        f(x, t) = [
            x1 + 0.05 * x2 + 0.5 * sin(0.1 * x2) + t1,
            0.9 * x2 + 0.2 * cos(0.3 * x1) + t2
        ]

    Measurement equation:
        h(x, u) = sqrt(x1^2 + x2^2) + u

    The model includes additive Gaussian process and observation noise.
    """

    MODEL_NAME: str = "x2_y1"

    # Terme de rappel sur x1 pour éviter la dérive (intégrateur pur sinon).
    # Avec kappa=0.10 : max||A||_2 ≈ 0.980 < 1 sur tout l'espace d'état.
    # (kappa=0.02 insuffisant : ||A||_2 > 1 pour |x1| > 5)
    KAPPA: float = 0.10

    # Garde-fou pour la division par r dans _jacobiens_g.
    R_MIN: float = 1e-8

    def __init__(self) -> None:
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
    def _fx(self, x, t, dt):
        if __debug__:
            if x.ndim == 2:
                assert all(a.shape == (self.dim_x, 1) for a in (x, t))
            else:
                assert all(
                    a.ndim == 3 and a.shape[1:] == (self.dim_x, 1) for a in (x, t)
                )
                assert x.shape[0] == t.shape[0]

        try:
            with np.errstate(all="raise"):
                if x.ndim == 2:
                    x1, x2 = x[0, 0], x[1, 0]
                    t1, t2 = t[0, 0], t[1, 0]
                    return np.array(
                        [
                            [
                                (1.0 - self.KAPPA) * x1
                                + 0.05 * x2
                                + 0.5 * np.sin(0.1 * x2)
                                + t1
                            ],
                            [0.9 * x2 + 0.2 * np.cos(0.3 * x1) + t2],
                        ]
                    )
                else:
                    x1, x2 = x[:, 0, 0], x[:, 1, 0]
                    t1, t2 = t[:, 0, 0], t[:, 1, 0]
                    out = np.empty_like(x)
                    out[:, 0, 0] = (
                        (1.0 - self.KAPPA) * x1
                        + 0.05 * x2
                        + 0.5 * np.sin(0.1 * x2)
                        + t1
                    )
                    out[:, 1, 0] = 0.9 * x2 + 0.2 * np.cos(0.3 * x1) + t2
                    return out

        except FloatingPointError as e:
            raise NumericalError(
                f"[{self.MODEL_NAME}] _fx: floating point error at x={x}, t={t}: {e}"
            ) from e
        except (IndexError, ValueError) as e:
            raise NumericalError(
                f"[{self.MODEL_NAME}] _fx: array access error at x={x}, t={t}: {e}"
            ) from e

    def _hx(self, x, u, dt):
        if __debug__:
            if x.ndim == 2:
                assert x.shape == (self.dim_x, 1)
                assert u.shape == (self.dim_y, 1)
            else:
                assert x.ndim == 3 and x.shape[1:] == (self.dim_x, 1)
                assert u.ndim == 3 and u.shape[1:] == (self.dim_y, 1)
                assert x.shape[0] == u.shape[0]

        try:
            with np.errstate(all="raise"):
                if x.ndim == 2:
                    return np.array([[np.sqrt(x[0, 0] ** 2 + x[1, 0] ** 2) + u[0, 0]]])
                else:
                    out = np.empty((x.shape[0], self.dim_y, 1))
                    out[:, 0, 0] = (
                        np.sqrt(x[:, 0, 0] ** 2 + x[:, 1, 0] ** 2) + u[:, 0, 0]
                    )
                    return out

        except FloatingPointError as e:
            raise NumericalError(
                f"[{self.MODEL_NAME}] _hx: floating point error at x={x}, u={u}: {e}"
            ) from e
        except (IndexError, ValueError) as e:
            raise NumericalError(
                f"[{self.MODEL_NAME}] _hx: array access error at x={x}, u={u}: {e}"
            ) from e

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
                    t1, t2 = t[0, 0], t[1, 0]

                    A = (
                        (1.0 - self.KAPPA) * x1
                        + 0.05 * x2
                        + 0.5 * np.sin(0.1 * x2)
                        + t1
                    )
                    B = 0.9 * x2 + 0.2 * np.cos(0.3 * x1) + t2
                    r = float(np.maximum(np.sqrt(A**2 + B**2), self.R_MIN))

                    An = np.array(
                        [
                            [1.0 - self.KAPPA, 0.05 * (1 + np.cos(0.1 * x2)), 0.0],
                            [-0.06 * np.sin(0.3 * x1), 0.9, 0.0],
                            [
                                (A - 0.06 * np.sin(0.3 * x1)) / r,
                                (0.05 * A * (1 + np.cos(0.1 * x2)) + 0.9 * B) / r,
                                0.0,
                            ],
                        ]
                    )
                    Bn = np.array(
                        [
                            [1.0, 0.0, 0.0],
                            [0.0, 1.0, 0.0],
                            [A / r, B / r, 1.0],
                        ]
                    )

                else:
                    x1, x2 = x[:, 0, 0], x[:, 1, 0]
                    t1, t2 = t[:, 0, 0], t[:, 1, 0]
                    N = x.shape[0]

                    A = (
                        (1.0 - self.KAPPA) * x1
                        + 0.05 * x2
                        + 0.5 * np.sin(0.1 * x2)
                        + t1
                    )
                    B = 0.9 * x2 + 0.2 * np.cos(0.3 * x1) + t2
                    r = np.maximum(np.sqrt(A**2 + B**2), self.R_MIN)  # (N,)

                    An = np.zeros((N, 3, 3))
                    An[:, 0, 0] = 1.0 - self.KAPPA
                    An[:, 0, 1] = 0.05 * (1 + np.cos(0.1 * x2))
                    An[:, 1, 0] = -0.06 * np.sin(0.3 * x1)
                    An[:, 1, 1] = 0.9
                    An[:, 2, 0] = (A - 0.06 * np.sin(0.3 * x1)) / r
                    An[:, 2, 1] = (0.05 * A * (1 + np.cos(0.1 * x2)) + 0.9 * B) / r

                    Bn = np.zeros((N, 3, 3))
                    Bn[:, 0, 0] = 1.0
                    Bn[:, 1, 1] = 1.0
                    Bn[:, 2, 0] = A / r
                    Bn[:, 2, 1] = B / r
                    Bn[:, 2, 2] = 1.0

            return An, Bn

        except FloatingPointError as e:
            raise NumericalError(
                f"[{self.MODEL_NAME}] _jacobiens_g: floating point error at x={x}, t={t}: {e}"
            ) from e
        except (IndexError, ValueError) as e:
            raise NumericalError(
                f"[{self.MODEL_NAME}] _jacobiens_g: array construction error: {e}"
            ) from e
