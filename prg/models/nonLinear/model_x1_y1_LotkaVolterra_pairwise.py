#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import sympy as sp

from prg.models.nonLinear.base_model_gxgy import BaseModelGxGy
from prg.utils.exceptions import NumericalError

__all__ = ["Model_x1_y1_LotkaVolterra_pairwise"]


class Model_x1_y1_LotkaVolterra_pairwise(BaseModelGxGy):
    """
    Modele proie-predateur de Lotka-Volterra discretise a l ordre 1
    (Euler explicite), forme pairwise scalaire (dim_x=1, dim_y=1).

    x = population de proies    (scalaire)
    y = population de predateurs (scalaire)

    Parametres :
        ALPHA = 0.5   taux de croissance des proies
        BETA  = 0.1   taux de predation
        GAMMA = 0.4   taux de mortalite des predateurs
        DELTA = 0.05  efficacite de conversion predation -> croissance
        DT    = 0.3   pas de discretisation temporelle

    Equation de transition gx(x, y, vx) :
        gx = (1 + ALPHA*DT)*x - BETA*DT*x*y + vx

    Equation d observation gy(x, y, vy) :
        gy = (1 - GAMMA*DT)*y + DELTA*DT*x*y + vy

    Point d equilibre non trivial : (x*, y*) = (GAMMA/DELTA, ALPHA/BETA) = (8, 5)

    Les jacobiens An = dg/dz et Bn = dg/dn sont calcules automatiquement
    par SymPy dans BaseModelGxGy.
    """

    ALPHA: float = 0.5
    BETA:  float = 0.1
    GAMMA: float = 0.4
    DELTA: float = 0.05
    DT:    float = 0.3

    def __init__(self):
        # Les attributs de classe sont accessibles avant super().__init__()
        # car super() appelle _build_symbolic_model() -> symbolic_model()
        super().__init__(dim_x=1, dim_y=1, model_type="nonlinear")

        # Point d equilibre non trivial
        x_eq = self.GAMMA / self.DELTA   # 8.0
        y_eq = self.ALPHA / self.BETA    # 5.0

        try:
            self.mQ, self.mz0, self.Pz0 = self._init_random_params(
                self.dim_x, self.dim_y, 0.10, seed=None
            )
            # Initialisation autour du point d equilibre
            self.mz0 = np.array([[x_eq], [y_eq]])
        except (ValueError, np.exceptions.AxisError) as e:
            raise NumericalError(
                f"[{self.MODEL_NAME}] Initialization failed: {e}"
            ) from e

    # ------------------------------------------------------------------
    def symbolic_model(self, sx, sy, st, su):
        x, y, t, u = sx[0], sy[0], st[0], su[0]

        sgx = sp.Matrix([
            (1 + self.ALPHA * self.DT) * x - self.BETA * self.DT * x * y + t
        ])
        sgy = sp.Matrix([
            (1 - self.GAMMA * self.DT) * y + self.DELTA * self.DT * x * y + u
        ])

        return sgx, sgy
