import numpy as np
import sympy as sp

from prg.models.nonLinear.base_model_fxhx import BaseModelFxHx
from prg.utils.exceptions import NumericalError

__all__ = ["Model_x1_y1_Sinus_classic"]


class Model_x1_y1_Sinus_classic(BaseModelFxHx):

    def __init__(self):

        super().__init__(dim_x=1, dim_y=1, model_type="nonlinear")

        try:
            self.mQ, self.mz0, self.Pz0 = self._init_random_params(
                self.dim_x, self.dim_y, 0.50, seed=None
            )

        except (ValueError, np.exceptions.AxisError) as e:
            raise NumericalError(
                f"[{self.MODEL_NAME}] Initialization failed: {e}"
            ) from e

    def symbolic_model(self, sx, st, su):
        x, t, u = sx[0], st[0], su[0]
        sfx = 0.05 * x + 2.0 * sp.sin(x) + t
        shx = 1.5 * sp.sin(x) + u
        return sp.Matrix([[sfx]]), sp.Matrix([[shx]])
