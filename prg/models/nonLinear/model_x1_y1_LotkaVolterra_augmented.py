#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import sympy as sp

from prg.models.nonLinear.base_model_fxhx import BaseModelFxHx
from prg.models.nonLinear.model_x1_y1_LotkaVolterra_pairwise import (
    Model_x1_y1_LotkaVolterra_pairwise,
)
from prg.utils.exceptions import NumericalError

__all__ = ["Model_x1_y1_LotkaVolterra_augmented"]


class Model_x1_y1_LotkaVolterra_augmented(BaseModelFxHx):
    """
    Version augmentee de Model_x1_y1_LotkaVolterra_pairwise (BaseModelFxHx).

    Etat augmente : x_aug = [xA, xB]
        xA = population de proies    (etat original x)
        xB = population de predateurs (observation precedente y)
    dim_x = 2, dim_y = 1, augmented = True.

    Dynamique  : f(x_aug) = [gx(xA, xB), gy(xA, xB)]
    Observation: h(x_aug) = xB  (predateurs)
    """

    def __init__(self):
        self.mod = Model_x1_y1_LotkaVolterra_pairwise()
        dim_x = self.mod.dim_x  # 1
        dim_y = self.mod.dim_y  # 1
        dim_xy = self.mod.dim_xy  # 2

        super().__init__(
            dim_x=dim_x + dim_y,  # 2
            dim_y=dim_y,  # 1
            model_type="nonlinear",
            augmented=True,
        )

        try:
            self.mQ = np.zeros((dim_xy + dim_y, dim_xy + dim_y))
            self.mQ[0:dim_xy, 0:dim_xy] = self.mod.mQ

            self.mz0 = np.zeros((dim_xy + dim_y, 1))
            self.mz0[0:dim_xy] = self.mod.mz0
            self.mz0[dim_xy : dim_xy + dim_y] = self.mz0[dim_xy - dim_y : dim_xy]

            self.Pz0 = np.zeros((dim_xy + dim_y, dim_xy + dim_y))
            self.Pz0[0:dim_xy, 0:dim_xy] = self.mod.Pz0
            self.Pz0[dim_xy : dim_xy + dim_y, :] = self.Pz0[dim_xy - dim_y : dim_xy, :]
            self.Pz0[:, dim_xy : dim_xy + dim_y] = self.Pz0[:, dim_xy - dim_y : dim_xy]

        except (ValueError, IndexError) as e:
            raise NumericalError(
                f"[{self.MODEL_NAME}] Initialization failed: {e}"
            ) from e

    # ------------------------------------------------------------------
    def symbolic_model(self, sx, st, su):
        """
        sx : sp.Matrix(2, 1) -> [xA, xB]  (proies, predateurs)
        st : sp.Matrix(2, 1) -> [t0, t1]  (bruits process)
        su : sp.Matrix(1, 1) -> [u0]      (bruit observation, non utilise dans h)
        """
        mx0 = sp.Symbol("x0", real=True)
        my0 = sp.Symbol("y0", real=True)
        mt0 = sp.Symbol("t0", real=True)
        mu0 = sp.Symbol("u0", real=True)

        subs_state = {mx0: sx[0], my0: sx[1]}

        sfx = sp.Matrix(
            [
                self.mod._sgx.subs({**subs_state, mt0: st[0]})[0],  # gx(xA, xB)
                self.mod._sgy.subs({**subs_state, mu0: st[1]})[0],  # gy(xA, xB)
            ]
        )

        # h(x_aug) = xB = predateurs
        shx = sp.Matrix([[sx[1]]])

        return sfx, shx
