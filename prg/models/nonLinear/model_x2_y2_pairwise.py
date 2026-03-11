#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import sympy as sp

from prg.models.nonLinear.base_model_gxgy import BaseModelGxGy
from prg.utils.exceptions import NumericalError

__all__ = ["Model_x2_y2_pairwise"]


class Model_x2_y2_pairwise(BaseModelGxGy):
    """
    Nonlinear model with full retroaction (dim_x=2, dim_y=2).

    State transition (depends on x and y) :
        gx1 = (1 - KAPPA)*x1 + 0.1*x2*tanh(y1) + t1
        gx2 = 0.9*x2 + 0.1*sin(x1)             + t2

    Observation (depends on x and y) :
        gy1 = x1 - 0.3*y2 + u1
        gy2 = x2 + 0.3*y1 + u2

    Jacobians (directs, pas de chain rule) :

        An = dg/dz = [ 1-K   0.1*tanh(y1)   0.1*x2*(1-tanh²(y1))   0    ]
                     [ 0.1*cos(x1)   0.9     0                       0    ]
                     [ 1     0               0                      -0.3  ]
                     [ 0     1               0.3                     0    ]

        Bn = dg/dn = I_4
    """

    MODEL_NAME: str = "Model_x2_y2_pairwise"
    KAPPA: float = 0.15

    def __init__(self):
        # KAPPA est un attribut de classe — pas de collision avec self.kappa (UPKF)
        # super().__init__() peut donc rester en première position
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
    def symbolic_model(self, sx, sy, st, su):
        x1, x2 = sx[0], sx[1]
        y1, y2 = sy[0], sy[1]
        t1, t2 = st[0], st[1]
        u1, u2 = su[0], su[1]

        sgx = sp.Matrix(
            [
                (1 - self.KAPPA) * x1 + sp.Rational(1, 10) * x2 * sp.tanh(y1) + t1,
                sp.Rational(9, 10) * x2 + sp.Rational(1, 10) * sp.sin(x1) + t2,
            ]
        )
        sgy = sp.Matrix(
            [
                x1 - sp.Rational(3, 10) * y2 + u1,
                x2 + sp.Rational(3, 10) * y1 + u2,
            ]
        )

        return sgx, sgy
