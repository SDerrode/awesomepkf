import numpy as np
import sympy as sp

from prg.models.nonlinear.base_model_gxgy import BaseModelGxGy
from prg.utils.exceptions import NumericalError

__all__ = ["Model_x1_y1_pairwise_param"]


class Model_x1_y1_pairwise_param(BaseModelGxGy):
    """
    Parameterizable pairwise nonlinear model (dim_x=1, dim_y=1).

    State transition (additive noise, depends on y via tanh):
        gx(x, y, t) = a*x + b*tanh(y) + t

    Observation (additive noise, depends on x via sin):
        gy(x, y, u) = c*y + d*sin(x/20) + u

    Parameters
    ----------
    b : float
        Back-action coupling strength (y -> x). b=0 means no coupling.
    q_x : float
        Variance of the state noise.
    q_y : float
        Variance of the observation noise.
    """

    def __init__(self, b: float = 3.0, q_x: float = 0.04, q_y: float = 0.025):
        self.a = 0.50
        self.b = float(b)
        self.c = 0.40
        self.d = 2.0

        super().__init__(dim_x=1, dim_y=1, model_type="nonlinear")

        try:
            self.mQ  = np.array([[q_x, 0.0], [0.0, q_y]])
            self.mz0 = np.zeros((self.dim_xy, 1))
            self.Pz0 = np.eye(self.dim_xy) * 0.1

        except (ValueError, np.exceptions.AxisError) as e:
            raise NumericalError(
                f"[{self.MODEL_NAME}] Initialization failed: {e}"
            ) from e

    def symbolic_model(self, sx, sy, st, su):
        x, y, t, u = sx[0], sy[0], st[0], su[0]

        sgx = sp.Matrix([self.a * x + self.b * sp.tanh(y) + t])
        sgy = sp.Matrix([self.c * y + self.d * sp.sin(x / 20) + u])

        return sgx, sgy
