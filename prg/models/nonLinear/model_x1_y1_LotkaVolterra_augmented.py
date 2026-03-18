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

    Dynamique  : schema d Euler explicite (polynomiale) :
        gx = (1 + ALPHA*DT)*xA - BETA*DT*xA*xB + vx
        gy = (1 - GAMMA*DT)*xB + DELTA*DT*xA*xB + vy
        L integrateur symplectique du modele pairwise contient exp(DELTA*xA)
        qui deborde des que xA > ~10 (x_eq = 0.47 pour C1). Le schema
        d Euler est polynomial — pas d overflow possible — et convient pour
        le filtre PKF qui corrige l etat a chaque pas avec les observations.
    Observation: h(x_aug) = xB  (predateurs)
    """

    def __init__(self):
        self.mod = Model_x1_y1_LotkaVolterra_pairwise()
        dim_x  = self.mod.dim_x   # 1
        dim_y  = self.mod.dim_y   # 1
        dim_xy = self.mod.dim_xy  # 2

        super().__init__(
            dim_x=dim_x + dim_y,   # 2
            dim_y=dim_y,           # 1
            model_type="nonlinear",
            augmented=True,
        )

        try:
            x_eq = self.mod.GAMMA / self.mod.DELTA
            y_eq = self.mod.ALPHA / self.mod.BETA

            # Convertit le bruit log-normal du modele pairwise en bruit additif
            # pour le schema d Euler : var_additif = equilibre^2 * sigma2_log.
            # Le Jacobien d Euler a |lambda|^2 = 1 + alpha*gamma*DT^2 > 1 :
            # un Q trop petit laisse la covariance croitre et devenir indefinie.
            # Q[1,1] = y_eq^2 * mQ[1,1] ≈ 26.7^2 * 0.01 ≈ 7 domine la croissance.
            var_xA = x_eq**2 * self.mod.mQ[0, 0]
            var_xB = y_eq**2 * self.mod.mQ[1, 1]

            self.mQ = np.zeros((dim_xy + dim_y, dim_xy + dim_y))
            self.mQ[0, 0] = var_xA
            self.mQ[1, 1] = var_xB

            self.mz0 = np.zeros((dim_xy + dim_y, 1))
            self.mz0[0:dim_xy] = self.mod.mz0
            self.mz0[dim_xy : dim_xy + dim_y] = self.mz0[dim_xy - dim_y : dim_xy]

            self.Pz0 = np.zeros((dim_xy + dim_y, dim_xy + dim_y))
            self.Pz0[0, 0] = var_xA
            self.Pz0[1, 1] = var_xB
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
        st : sp.Matrix(2, 1) -> [t0, t1]  (bruits process additifs)
        su : sp.Matrix(1, 1) -> [u0]      (bruit observation, non utilise dans h)

        Schema d Euler explicite : dynamique polynomiale, pas d overflow.
        Le Jacobien d(sfx)/d(sx) ne depend pas de st -> _eval_A fonctionne.
        """
        xA, xB = sx[0], sx[1]
        A  = self.mod.ALPHA
        B  = self.mod.BETA
        G  = self.mod.GAMMA
        D  = self.mod.DELTA
        DT = self.mod.DT

        sfx = sp.Matrix([
            (1 + A * DT) * xA - B * DT * xA * xB + st[0],
            (1 - G * DT) * xB + D * DT * xA * xB + st[1],
        ])

        shx = sp.Matrix([[xB]])

        return sfx, shx
