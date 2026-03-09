#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import sympy as sp

from prg.models.nonLinear.base_model_fxhx import BaseModelFxHx
from prg.models.Generate_MatrixCov import generate_block_matrix
from prg.utils.exceptions import NumericalError

__all__ = ["ModelX2Y1Rapport"]


class ModelX2Y1Rapport(BaseModelFxHx):
    """
    Nonlinear model with:
      - 2-dimensional state vector x = [x1, x2]
      - 1-dimensional observation y

    System dynamics:
        f(x) = [
            (1 - kappa_m)*x1 + dt_model*x2 + t1,
            x2 - dt_model*(alpha*sin(x1) + beta*x2) + t2
        ]

    Measurement equation:
        h(x) = x1^2/(1 + x1^2) + gamma*sin(x2) + u
    """

    MODEL_NAME: str = "x2_y1_rapport"

    def __init__(self):
        # ← paramètres AVANT super().__init__() qui appelle _build_symbolic_model()
        self.alpham = 0.5
        self.betam = 0.5
        self.gammam = 0.5
        self.kappa_m = 0.15
        self.dt_model = 0.1

        super().__init__(dim_x=2, dim_y=1, model_type="nonlinear")

        try:
            self.mQ = generate_block_matrix(
                self._randMatrices.rng, self.dim_x, self.dim_y, 0.1
            )
            self.mz0 = self._randMatrices.rng.standard_normal((self.dim_xy, 1))
            self.Pz0 = generate_block_matrix(
                self._randMatrices.rng, self.dim_x, self.dim_y, 0.05
            )
        except (ValueError, np.exceptions.AxisError) as e:
            raise NumericalError(
                f"[{self.MODEL_NAME}] Initialization failed: {e}"
            ) from e

    def symbolic_model(self, sx, st, su):
        x1, x2 = sx[0], sx[1]
        t1, t2 = st[0], st[1]
        u = su[0]

        sfx1 = (1 - self.kappa_m) * x1 + self.dt_model * x2 + t1
        sfx2 = x2 - self.dt_model * (self.alpham * sp.sin(x1) + self.betam * x2) + t2
        shx = x1**2 / (1 + x1**2) + self.gammam * sp.sin(x2) + u

        return sp.Matrix([sfx1, sfx2]), sp.Matrix([shx])
