#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import sympy as sp


import numpy as np
from prg.models.nonLinear.base_model_fxhx import BaseModelFxHx
from prg.models.nonLinear.model_x2_y2_pairwise import Model_x2_y2_pairwise
from prg.utils.exceptions import NumericalError


__all__ = ["Model_x2_y2_augmented"]


class Model_x2_y2_augmented(BaseModelFxHx):
    """
    Version augmentée de Model_x2_y2_pairwise (BaseModelFxHx).

    État augmenté : x_aug = [x1, x2, x3] où x3 = observation précédente (y_prev).
    dim_x = 3, dim_y = 1, augmented = True.  split = dim_x - dim_y = 2.
    """

    def __init__(self):

        self.mod = Model_x2_y2_pairwise()

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
        """
        sx : sp.Matrix(4, 1) → [x1, x2, y1, y2]   (état augmenté)
        st : sp.Matrix(4, 1) → [t0, t1, t2, t3]   (bruits process : x1, x2, y1, y2)
        su : sp.Matrix(2, 1) → [u0, u1]            (bruits d'observation)
        """
        mx0 = sp.Symbol("x0", real=True)
        mx1 = sp.Symbol("x1", real=True)
        my0 = sp.Symbol("y0", real=True)
        my1 = sp.Symbol("y1", real=True)
        mt0 = sp.Symbol("t0", real=True)
        mt1 = sp.Symbol("t1", real=True)
        mu0 = sp.Symbol("u0", real=True)
        mu1 = sp.Symbol("u1", real=True)

        subs_state = {mx0: sx[0], mx1: sx[1], my0: sx[2], my1: sx[3]}

        sfx = sp.Matrix(
            [
                self.mod._sgx.subs({**subs_state, mt0: st[0]})[
                    0
                ],  # (1-κ)*x1 + 0.1*x2*tanh(y1) + t0
                self.mod._sgx.subs({**subs_state, mt1: st[1]})[
                    1
                ],  # 0.9*x2 + 0.1*sin(x1) + t1
                self.mod._sgy.subs({**subs_state, mu0: st[2]})[0],  # x1 - 0.3*y2 + t2
                self.mod._sgy.subs({**subs_state, mu1: st[3]})[1],  # x2 + 0.3*y1 + t3
            ]
        )

        # h(x) = [x2, x3] (= y1, y2) : identité pure
        shx = sp.Matrix([[sx[2]], [sx[3]]])

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
