#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import sympy as sp

from prg.models.nonLinear.base_model_gxgy import BaseModelGxGy
from prg.utils.exceptions import NumericalError

__all__ = ["Model_x1_y1_multiplicative"]


class Model_x1_y1_multiplicative(BaseModelGxGy):
    """
    Pairwise nonlinear model with MULTIPLICATIVE observation noise (dim_x=1, dim_y=1).

    State transition (additive noise):
        gx(x, y, t) = a*x + b*tanh(y) + t

    Observation (multiplicative noise in v^y):
        gy(x, y, u) = c*y + d*sin(x/20) + (1 + x^2) * u

    The multiplicative factor (1+x^2) makes Bn non-constant:
        Bn = dg/dn = [ 1          0         ]
                     [ 0     (1 + x^2)      ]

    This model is designed to reveal the difference between UPKF and UKF with
    augmented state: the UKF linearises Bn at the estimated state (EKF-style
    correction to P^yy), whereas the UPKF propagates the observation noise
    through g^y via sigma points, capturing the nonlinear x-dependence exactly.
    """

    def __init__(self, q_y: float = 0.10):
        """
        Parameters
        ----------
        q_y : float
            Variance of the observation noise v^y (before the (1+x^2) factor).
            The effective observation noise variance is (1+x^2)^2 * q_y.
        """
        self.a = 0.50
        self.b = 3.0
        self.c = 0.40
        self.d = 2.0
        self.q_y = float(q_y)

        super().__init__(dim_x=1, dim_y=1, model_type="nonlinear")

        try:
            q_x = 0.04
            self.mQ  = np.array([[q_x, 0.0],
                                  [0.0, self.q_y]])
            self.mz0 = np.zeros((self.dim_xy, 1))
            self.Pz0 = np.eye(self.dim_xy) * 0.1

        except (ValueError, np.exceptions.AxisError) as e:
            raise NumericalError(
                f"[{self.MODEL_NAME}] Initialization failed: {e}"
            ) from e

    def symbolic_model(self, sx, sy, st, su):
        x, y, t, u = sx[0], sy[0], st[0], su[0]

        sgx = sp.Matrix([self.a * x + self.b * sp.tanh(y) + t])
        sgy = sp.Matrix([self.c * y + self.d * sp.sin(x / 20) + (1 + x**2) * u])

        return sgx, sgy
