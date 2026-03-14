#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import sympy as sp

from prg.models.nonLinear.base_model_fxhx import BaseModelFxHx
from prg.models.nonLinear.model_x1_y1_pairwise import Model_x1_y1_pairwise
from prg.utils.exceptions import NumericalError

__all__ = ["Model_x1_y1_augmented"]


class Model_x1_y1_augmented(BaseModelFxHx):
    """
    Version augmentée de Model_x1_y1_pairwise (BaseModelFxHx).

    État augmenté : x_aug = [xA, xB] où xA = état original, xB = observation précédente.
    dim_x = 2, dim_y = 1, augmented = True.

    """

    def __init__(self):

        self.mod = Model_x1_y1_pairwise()
        dim_x = self.mod.dim_x
        dim_y = self.mod.dim_y
        dim_xy = self.mod.dim_xy

        super().__init__(
            dim_x=dim_x + dim_y,
            dim_y=dim_y,
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

    def symbolic_model(self, sx, st, su):
        mx0 = sp.Symbol("x0", real=True)
        my0 = sp.Symbol("y0", real=True)
        mt0 = sp.Symbol("t0", real=True)
        mu0 = sp.Symbol("u0", real=True)

        subs_state = {mx0: sx[0], my0: sx[1]}

        sfx = sp.Matrix(
            [
                self.mod._sgx.subs({**subs_state, mt0: st[0]})[
                    0
                ],  # a*x + b*tanh(y) + t0
                self.mod._sgy.subs({**subs_state, mu0: st[1]})[
                    0
                ],  # c*y + d*sin(x/20) + t1
            ]
        )

        shx = sp.Matrix([[sx[1]]])

        # ── Diagnostic ───────────────────────────────────────────────────
        expected_fx = set(sx.free_symbols) | set(st.free_symbols)
        expected_hx = set(sx.free_symbols) | set(su.free_symbols)
        residual_fx = sfx.free_symbols - expected_fx
        residual_hx = shx.free_symbols - expected_hx

        if residual_fx:
            print(f"[WARN] sfx contient des symboles non attendus : {residual_fx}")
        if residual_hx:
            print(f"[WARN] shx contient des symboles non attendus : {residual_hx}")

        return sfx, shx
