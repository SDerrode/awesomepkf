import logging

import numpy as np
import sympy as sp

from prg.models.nonlinear import ModelFactoryNonLinear
from prg.models.nonlinear.base_model_fxhx import BaseModelFxHx
from prg.utils.exceptions import NumericalError

logger = logging.getLogger(__name__)

__all__ = ["Model_x2_y1_augmented"]


class Model_x2_y1_augmented(BaseModelFxHx):
    """
    Version augmentée de Model_x2_y1_pairwise (BaseModelFxHx).

    État augmenté : x_aug = [x1, x2, x3] où x3 = observation précédente (y_prev).
    dim_x = 3, dim_y = 1, augmented = True.  split = dim_x - dim_y = 2.

    """

    def __init__(self):

        self.mod = ModelFactoryNonLinear.create("model_x2_y1_pairwise")

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

    # ------------------------------------------------------------------
    def symbolic_model(self, sx, st, su):
        """
        sx : sp.Matrix(3, 1) → [x1, x2, y1]   (état augmenté)
        st : sp.Matrix(3, 1) → [t0, t1, t2]   (bruits process : x1, x2, y1)
        su : sp.Matrix(1, 1) → [u0]           (bruit d'observation, non utilisé)
        """
        mx0 = sp.Symbol("x0", real=True)
        mx1 = sp.Symbol("x1", real=True)
        my0 = sp.Symbol("y0", real=True)
        mt0 = sp.Symbol("t0", real=True)
        mt1 = sp.Symbol("t1", real=True)
        mu0 = sp.Symbol("u0", real=True)

        subs_state = {mx0: sx[0], mx1: sx[1], my0: sx[2]}

        sfx = sp.Matrix(
            [
                self.mod._sgx.subs({**subs_state, mt0: st[0]})[
                    0
                ],  # a*x1 + b*x2 + c*tanh(y1) + t0
                self.mod._sgx.subs({**subs_state, mt1: st[1]})[
                    1
                ],  # d*x2 + e*sin(y1) + t1
                self.mod._sgy.subs({**subs_state, mu0: st[2]})[
                    0
                ],  # x1²/(1+x1²) + f*y1 + t2
            ]
        )

        # h(x) = x2 (= y1) : identité pure
        shx = sp.Matrix([[sx[2]]])

        # ── Diagnostic ───────────────────────────────────────────────────
        expected_fx = set(sx.free_symbols) | set(st.free_symbols)
        expected_hx = set(sx.free_symbols) | set(su.free_symbols)
        residual_fx = sfx.free_symbols - expected_fx
        residual_hx = shx.free_symbols - expected_hx

        if residual_fx:
            logger.warning("sfx contains unexpected symbols : %s", residual_fx)
        if residual_hx:
            logger.warning("shx contains unexpected symbols : %s", residual_hx)

        return sfx, shx
