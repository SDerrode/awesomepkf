import numpy as np
import sympy as sp

from prg.models.nonLinear.base_model_fxhx import BaseModelFxHx
from prg.models.nonLinear.model_x1_y1_multiplicative import Model_x1_y1_multiplicative
from prg.utils.exceptions import NumericalError

__all__ = ["Model_x1_y1_multiplicative_augmented"]


class Model_x1_y1_multiplicative_augmented(BaseModelFxHx):
    """
    Augmented-state version of Model_x1_y1_multiplicative for use with the UKF.

    Augmented state: x_aug = [x, y] in R^2, observation: y_obs = x_aug[1].

    Transition (same dynamics, noise enters via st):
        f_x(x_aug, t) = a*x + b*tanh(y) + t[0]
        f_y(x_aug, t) = c*y + d*sin(x/20) + (1 + x^2) * t[1]   <- multiplicative

    Observation (linear, noiseless):
        h(x_aug) = x_aug[1]   (= y)

    The UKF applied to this model propagates sigma points in R^{2*dim_x} = R^4
    (state [x,y] + noise [t_x, t_y]), using lambda = alpha^2*(4+kappa)-4.

    By contrast, the UPKF applied to Model_x1_y1_multiplicative propagates
    sigma points in R^{2*dim_x+dim_y} = R^3 (only [x, t_x, t_y], since y is
    observed deterministically), using lambda = alpha^2*(3+kappa)-3.
    """

    def __init__(self, q_y: float = 0.10):
        self.mod = Model_x1_y1_multiplicative(q_y=q_y)
        dim_x = self.mod.dim_x   # = 1
        dim_y = self.mod.dim_y   # = 1
        dim_xy = self.mod.dim_xy  # = 2

        super().__init__(
            dim_x=dim_x + dim_y,   # = 2  (augmented state [x, y])
            dim_y=dim_y,            # = 1
            model_type="nonlinear",
            augmented=True,
        )

        try:
            # Build Q for the augmented model:
            #   mQ[0:dim_xy, 0:dim_xy] = Q from pairwise model
            #   extra dim_y rows/cols = 0 (no additional noise)
            extra = dim_y
            total = dim_xy + extra
            self.mQ = np.zeros((total, total))
            self.mQ[0:dim_xy, 0:dim_xy] = self.mod.mQ

            self.mz0 = np.zeros((total, 1))
            self.mz0[0:dim_xy] = self.mod.mz0
            self.mz0[dim_xy : dim_xy + dim_y] = self.mod.mz0[dim_xy - dim_y : dim_xy]

            self.Pz0 = np.zeros((total, total))
            self.Pz0[0:dim_xy, 0:dim_xy] = self.mod.Pz0
            self.Pz0[dim_xy : dim_xy + dim_y, :] = self.Pz0[dim_xy - dim_y : dim_xy, :]
            self.Pz0[:, dim_xy : dim_xy + dim_y] = self.Pz0[:, dim_xy - dim_y : dim_xy]

        except (ValueError, IndexError) as e:
            raise NumericalError(
                f"[{self.MODEL_NAME}] Initialization failed: {e}"
            ) from e

    def symbolic_model(self, sx, st, su):
        # sx[0] = x, sx[1] = y (augmented state)
        # st[0] = t_x (state noise), st[1] = t_y (obs noise, multiplicative)
        x, y = sx[0], sx[1]
        t_x, t_y = st[0], st[1]

        sfx = sp.Matrix([
            self.mod.a * x + self.mod.b * sp.tanh(y) + t_x,
            self.mod.c * y + self.mod.d * sp.sin(x / 20) + (1 + x**2) * t_y,
        ])

        shx = sp.Matrix([[sx[1]]])   # observation = y (second augmented-state component)

        return sfx, shx
