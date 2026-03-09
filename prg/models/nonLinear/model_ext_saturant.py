#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import sympy as sp

from prg.models.nonLinear.base_model_fxhx import BaseModelFxHx
from prg.models.Generate_MatrixCov import generate_block_matrix
from prg.utils.numerics import EPS_REL
from prg.utils.exceptions import NumericalError

__all__ = ["ModelExtSaturant"]


class ModelExtSaturant(BaseModelFxHx):
    MODEL_NAME: str = "x1_y1_ext_saturant"

    def __init__(self):

        super().__init__(dim_x=1, dim_y=1, model_type="nonlinear")

        try:
            self.mQ = generate_block_matrix(
                self._randMatrices.rng, self.dim_x, self.dim_y, 0.05
            )
            self.mz0 = self._randMatrices.rng.standard_normal((self.dim_xy, 1))
            self.Pz0 = generate_block_matrix(
                self._randMatrices.rng, self.dim_x, self.dim_y, 0.03
            )
        except (ValueError, np.exceptions.AxisError) as e:
            raise NumericalError(
                f"[{self.MODEL_NAME}] Initialization failed: {e}"
            ) from e

    # ------------------------------------------------------------------
    def symbolic_model(self, sx, st, su):
        x, t, u = sx[0], st[0], su[0]
        sfx = 0.5 * x + 2.0 * (1 - sp.exp(-0.1 * x)) + t
        shx = sp.log(1 + sp.Max(sp.Abs(x), EPS_REL)) + u
        return sp.Matrix([[sfx]]), sp.Matrix([[shx]])
