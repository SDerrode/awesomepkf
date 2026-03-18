#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import sympy as sp

from prg.models.nonLinear.base_model_gxgy import BaseModelGxGy
from prg.utils.exceptions import NumericalError

__all__ = ["Model_x1_y1_LotkaVolterra_pairwise"]


class Model_x1_y1_LotkaVolterra_pairwise(BaseModelGxGy):
    """
    Modele proie-predateur de Lotka-Volterra, forme pairwise scalaire
    (dim_x=1, dim_y=1).

    x = population de proies    (scalaire)
    y = population de predateurs (scalaire)

    Equations de transition (integrateur symplectique de Suris) :
        y_det  = y * exp((DELTA*x - GAMMA)*DT)           [y sans bruit]
        gx     = x * exp((ALPHA - BETA*y_det)*DT + vx)   [x utilise y_det]
        gy     = y_det * exp(vy)                          [bruit multiplicatif]

    L integrateur symplectique (Suris / Volterra-preserving) met a jour y
    en premier puis utilise ce y dans la mise a jour de x. Le determinant
    du Jacobien de la partie deterministe vaut 1 (volume-preservant) :
    les trajectoires restent sur les courbes fermees du systeme continu.
    Toute methode explicite (Euler, exponentielle) a des valeurs propres
    1 +/- i*sqrt(alpha*gamma)*DT de module > 1 -- instable.

    Bruit log-normal (dans l exponentielle) : garantit x > 0, y > 0.

    Point d equilibre non trivial : (x*, y*) = (GAMMA/DELTA, ALPHA/BETA)

    Les jacobiens An = dg/dz et Bn = dg/dn sont calcules automatiquement
    par SymPy dans BaseModelGxGy.

    D apres le script estimate_lotka_volterra.py :
         alpha    beta   gamma   delta  sigma2_u  sigma2_v    x_eq     y_eq
    file
    C1.csv 0.27503 0.01030 0.35974 0.76738   0.16179   0.09662 0.46878 26.69027
    C2.csv 0.10867 0.00768 0.22848 0.41950   0.12951   0.13069 0.54466 14.14477
    C4.csv 0.37483 0.01657 0.10005 0.18329   0.05395   0.08138 0.54587 22.62324
    C5.csv 0.13292 0.00784 0.25214 0.56788   0.10016   0.08002 0.44399 16.95028
    C6.csv 0.00312 0.00014 0.02534 0.01175   0.04425   0.15036 2.15718 21.86837
    C8.csv 0.17664 0.02475 0.25907 0.34799   0.20895   0.13382 0.74447  7.13705
    C9.csv 0.14704 0.00608 0.26263 0.27429   0.11398   0.04613 0.95752 24.17644
    """

    # C1
    ALPHA: float = 0.27503
    BETA:  float = 0.01030
    GAMMA: float = 0.35974
    DELTA: float = 0.76738
    # C4
    # ALPHA: float = 0.37483
    # BETA:  float = 0.01657
    # GAMMA: float = 0.10005
    # DELTA: float = 0.18329
    # parametres classiques (equilibre x*=8, y*=5)
    # ALPHA: float = 0.5
    # BETA:  float = 0.1
    # GAMMA: float = 0.4
    # DELTA: float = 0.05

    DT: float = 1.0

    def __init__(self):
        # Les attributs de classe sont accessibles avant super().__init__()
        # car super() appelle _build_symbolic_model() -> symbolic_model()
        super().__init__(dim_x=1, dim_y=1, model_type="nonlinear")

        # Point d equilibre non trivial
        x_eq = self.GAMMA / self.DELTA
        y_eq = self.ALPHA / self.BETA

        try:
            self.mQ, self.mz0, self.Pz0 = self._init_random_params(
                self.dim_x, self.dim_y, 0.10, seed=None
            )
            # Initialisation autour du point d equilibre
            self.mz0 = np.array([[x_eq], [y_eq]])
            # Variance du bruit (faible pour simulation stable)
            self.mQ = np.diag([0.01, 0.01])

        except (ValueError, np.exceptions.AxisError) as e:
            raise NumericalError(
                f"[{self.MODEL_NAME}] Initialization failed: {e}"
            ) from e

    # ------------------------------------------------------------------
    def symbolic_model(self, sx, sy, st, su):
        x, y, t, u = sx[0], sy[0], st[0], su[0]

        # Integrateur symplectique de Suris : y mis a jour en premier,
        # puis x utilise ce y_det (volume-preservant, trajectoires bornees)
        y_det = y * sp.exp((self.DELTA * x - self.GAMMA) * self.DT)

        sgx = sp.Matrix([x * sp.exp((self.ALPHA - self.BETA * y_det) * self.DT + t)])
        sgy = sp.Matrix([y_det * sp.exp(u)])

        return sgx, sgy
