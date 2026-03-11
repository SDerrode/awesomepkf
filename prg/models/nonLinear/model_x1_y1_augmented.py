#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import sympy as sp

from prg.models.nonLinear.base_model_fxhx import BaseModelFxHx
from prg.models.nonLinear.Model_x1_y1_pairwise import Model_x1_y1_pairwise
from prg.utils.exceptions import NumericalError

__all__ = ["Model_x1_y1_augmented"]


class Model_x1_y1_augmented(BaseModelFxHx):
    """
    Version augmentée de Model_x1_y1_pairwise (BaseModelFxHx).

    État augmenté : x_aug = [xA, xB] où xA = état original, xB = observation précédente.
    dim_x = 2, dim_y = 1, augmented = True.

    Transition d'état (fx) :
        fx[0] = a*x1 + b*tanh(x2) + t1   ← gx du modèle de base
        fx[1] = c*x2 + d*sin(x1)  + t2   ← gy du modèle de base (t2 joue le rôle de u)

    Observation (hx) :
        hx = x2                           ← dernière composante de l'état augmenté

    Jacobiennes (calculées par la classe de base via chain rule) :
        A  = dfx/dx = [[a,           b*(1-tanh²(x2))],
                       [d*cos(x1),   c              ]]

        H  = dhx/dx (évalué en fx) = [0, 1]

        An = [[A,     0],
              [H@A,   0]]

        Bn = [[I,     0],
              [H,     0]]
    """

    MODEL_NAME: str = "Model_x1_y1_augmented"

    def __init__(self):
        # ← self.mod instancié AVANT super() : ses paramètres sont utilisés
        #   dans symbolic_model(), appelé depuis _build_symbolic_model()
        #   à l'intérieur de super().__init__().
        self.mod = Model_x1_y1_pairwise()
        self.a, self.b, self.c, self.d = (
            self.mod.a,
            self.mod.b,
            self.mod.c,
            self.mod.d,
        )

        super().__init__(dim_x=2, dim_y=1, model_type="nonlinear", augmented=True)

        try:
            dim_x, dim_y, dim_xy = self.mod.dim_x, self.mod.dim_y, self.mod.dim_xy
            self.mQ = np.zeros((self.dim_xy, self.dim_xy))
            self.mQ[: self.dim_x, : self.dim_x] = self.mod.mQ

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
        sx : sp.Matrix(2, 1) → [x1, x2]  (état augmenté)
        st : sp.Matrix(2, 1) → [t1, t2]  (bruits d'état)
        su : sp.Matrix(1, 1) → [u]        (bruit d'observation, non utilisé ici)
        """
        x1, x2 = sx[0], sx[1]
        t1, t2 = st[0], st[1]

        sfx = sp.Matrix(
            [
                self.a * x1 + self.b * sp.tanh(x2) + t1,
                self.c * x2 + self.d * sp.sin(x1) + t2,
            ]
        )
        shx = sp.Matrix([x2])  # hx = dernière composante de l'état augmenté

        return sfx, shx
