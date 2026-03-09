#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import sympy as sp

from prg.models.nonLinear.base_model_gxgy import BaseModelGxGy
from prg.models.Generate_MatrixCov import generate_block_matrix
from prg.utils.exceptions import NumericalError

__all__ = ["ModelX2Y1_Retroactions"]


class ModelX2Y1_Retroactions(BaseModelGxGy):
    """
    Nonlinear model with full retroaction (dim_x=2, dim_y=1).

    State transition (depends on x and y) :
        gx1 = a*x1 + b*x2 + c*tanh(y1) + t1
        gx2 = d*x2 + e*sin(y1)          + t2

    Observation (depends on x and y) :
        gy  = x1²/(1 + x1²) + f*y1 + u

    Jacobians (directs, pas de chain rule) :

        An = dg/dz = [ a    b    c*(1 - tanh²(y1))   ]
                     [ 0    d    e*cos(y1)             ]
                     [ 2*x1/(1+x1²)²   0    f         ]

        Bn = dg/dn = I_3
    """

    MODEL_NAME: str = "x2_y1_Retroactions"

    def __init__(self):
        # ← paramètres AVANT super().__init__() qui appelle _build_symbolic_model()
        self.a = 0.95
        self.b = 0.10
        self.c = 0.05
        self.d = 0.9
        self.e = 0.30
        self.f = 0.6

        super().__init__(dim_x=2, dim_y=1, model_type="nonlinear")

        try:
            Q = np.array([[0.03, 0.0], [0.0, 0.03]])
            R = np.array([[0.03]])
            M = np.zeros((self.dim_x, self.dim_y))
            self.mQ = np.block([[Q, M], [M.T, R]])
            self.mz0 = self._randMatrices.rng.standard_normal((self.dim_xy, 1))
            self.Pz0 = generate_block_matrix(
                self._randMatrices.rng, self.dim_x, self.dim_y, 0.05
            )
        except (ValueError, np.exceptions.AxisError) as e:
            raise NumericalError(
                f"[{self.MODEL_NAME}] Initialization failed: {e}"
            ) from e

    # ------------------------------------------------------------------
    def symbolic_model(self, sx, sy, st, su):
        x1, x2 = sx[0], sx[1]
        y1 = sy[0]
        t1, t2 = st[0], st[1]
        u = su[0]

        sgx = sp.Matrix(
            [
                self.a * x1 + self.b * x2 + self.c * sp.tanh(y1) + t1,
                self.d * x2 + self.e * sp.sin(y1) + t2,
            ]
        )
        sgy = sp.Matrix(
            [
                x1**2 / (1 + x1**2) + self.f * y1 + u,
            ]
        )

        return sgx, sgy
