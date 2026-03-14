#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import sympy as sp

from prg.models.nonLinear.base_model_fxhx import BaseModelFxHx
from prg.models.Generate_MatrixCov import generate_block_matrix
from prg.utils.exceptions import NumericalError

__all__ = ["Model_x2_y1_classic"]


class Model_x2_y1_classic(BaseModelFxHx):
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

    KAPPA: float = 0.10
    R_MIN: float = 1e-8

    def __init__(self):

        super().__init__(dim_x=2, dim_y=1, model_type="nonlinear")
        try:
            self.mQ, self.mz0, self.Pz0 = self._init_random_params(
                self.dim_x, self.dim_y, 0.50, seed=None
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
