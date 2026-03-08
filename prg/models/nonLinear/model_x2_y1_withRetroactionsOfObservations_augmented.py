#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import sympy as sp


import numpy as np
from prg.models.nonLinear.base_model_fxhx import BaseModelFxHx
from prg.models.nonLinear.model_x2_y1_withRetroactionsOfObservations import (
    ModelX2Y1_withRetroactionsOfObservations,
)
from prg.exceptions import NumericalError


__all__ = ["ModelX2Y1_withRetroactionsOfObservations_augmented"]


class ModelX2Y1_withRetroactionsOfObservations_augmented(BaseModelFxHx):
    """
    Version augmentée de ModelX2Y1_withRetroactionsOfObservations (BaseModelFxHx).

    État augmenté : x_aug = [x1, x2, x3] où x3 = observation précédente (y_prev).
    dim_x = 3, dim_y = 1, augmented = True.  split = dim_x - dim_y = 2.

    Transition d'état (fx) :
        fx[0] = a*x1 + b*x2 + c*tanh(x3) + t1   ← gx1 du modèle de base
        fx[1] = d*x2 + e*sin(x3)          + t2   ← gx2 du modèle de base
        fx[2] = x1²/(1+x1²) + f*x3        + t3   ← gy  du modèle de base (t3 ≡ u)

    Observation (hx) :
        hx = x3                                   ← dernière composante

    Jacobiennes (calculées par la classe de base via chain rule) :
        A  = dfx/dx (3×3)
           = [ a    b    c*(1-tanh²(x3))       ]
             [ 0    d    e*cos(x3)              ]
             [ 2x1/(1+x1²)²   0    f            ]

        H  = dhx/dx (évalué en fx) = [0, 0, 1]

        An = [[A,      0],      (4×4)
              [H@A,    0]]

        Bn = [[I_3,    0],      (4×4)
              [H,      0]]
    """

    MODEL_NAME: str = "x2_y1_withRetroactionsOfObservations_augmented"

    def __init__(self):
        # ← self.mod et ses paramètres AVANT super() :
        #   symbolic_model() est appelé depuis _build_symbolic_model()
        #   à l'intérieur de super().__init__().
        self.mod = ModelX2Y1_withRetroactionsOfObservations()
        self.a, self.b, self.c, self.d, self.e, self.f = (
            self.mod.a,
            self.mod.b,
            self.mod.c,
            self.mod.d,
            self.mod.e,
            self.mod.f,
        )

        super().__init__(dim_x=3, dim_y=1, model_type="nonlinear", augmented=True)

        try:
            self.mQ = np.zeros((self.dim_xy, self.dim_xy))
            self.mQ[: self.dim_x, : self.dim_x] = self.mod.mQ

            dim_x, dim_y, dim_xy = self.mod.dim_x, self.mod.dim_y, self.mod.dim_xy
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
        sx : sp.Matrix(3, 1) → [x1, x2, x3]  (état augmenté, x3 = y_prev)
        st : sp.Matrix(3, 1) → [t1, t2, t3]  (bruits d'état, t3 ≡ u du modèle de base)
        su : sp.Matrix(1, 1) → [u]            (bruit d'observation, non utilisé ici)
        """
        x1, x2, x3 = sx[0], sx[1], sx[2]
        t1, t2, t3 = st[0], st[1], st[2]

        sfx = sp.Matrix(
            [
                self.a * x1 + self.b * x2 + self.c * sp.tanh(x3) + t1,
                self.d * x2 + self.e * sp.sin(x3) + t2,
                x1**2 / (1 + x1**2) + self.f * x3 + t3,
            ]
        )
        shx = sp.Matrix([x3])  # hx = dernière composante de l'état augmenté

        return sfx, shx
