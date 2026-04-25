#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import sympy as sp

from prg.models.nonLinear.base_model_gxgy import BaseModelGxGy
from prg.utils.generate_matrix_cov import generate_block_matrix
from prg.utils.exceptions import NumericalError

__all__ = ["Model_x1_y1_pairwise"]


class Model_x1_y1_pairwise(BaseModelGxGy):
    """
    Nonlinear model with full retroaction (dim_x=1, dim_y=1).

    State transition (depends on both x and y) :
        gx(x, y, t) = a*x + b*tanh(y) + t

    Observation (depends on both x and y) :
        gy(x, y, u) = c*y + d*sin(x) + u

    Jacobians (directs, pas de chain rule) :

        An = dg/dz = [ a           b/cosh²(y) ]
                     [ d*cos(x)    c           ]

        Bn = dg/dn = I_2
    """

    def __init__(self):
        # paramètres AVANT super().__init__()
        self.a = 0.50
        self.b = 3
        self.c = 0.40
        self.d = 2
        # self.a = 0.99
        # self.b = 1.2
        # self.c = 0.9
        # self.d = 1.5

        super().__init__(dim_x=1, dim_y=1, model_type="nonlinear")
        try:
            self.mQ, self.mz0, self.Pz0 = self._init_random_params(
                self.dim_x, self.dim_y, 0.50, seed=None
            )

        except (ValueError, np.exceptions.AxisError) as e:
            raise NumericalError(
                f"[{self.MODEL_NAME}] Initialization failed: {e}"
            ) from e

    # ------------------------------------------------------------------
    def symbolic_model(self, sx, sy, st, su):
        x, y, t, u = sx[0], sy[0], st[0], su[0]

        sgx = sp.Matrix([self.a * x + self.b * sp.tanh(y) + t])
        sgy = sp.Matrix([self.c * y + self.d * sp.sin(x / 20) + u])

        return sgx, sgy
