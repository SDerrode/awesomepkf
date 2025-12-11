import numpy as np
from typing import Callable
from .base_model_nonLinear import BaseModelNonLinear

# A few utils functions that are used several times
from others.utils import check_consistency

class ModelX1Y1_withRetroactions(BaseModelNonLinear):
    """
    Nonlinear model with retro-actions of observations and of states.
    The model includes additive Gaussian process and observation noises.
    """

    MODEL_NAME: str = "x1_y1_withRetroactions"

    def __init__(self) -> None:
        super().__init__(dim_x=1, dim_y=1, model_type="nonlinear")

        self.mQ   = np.diag([0.05, 0.05]) #[1E-1, 1E-1])
        self.z00  = np.zeros((self.dim_xy, 1))
        self.Pz00 = np.eye(self.dim_xy)
        
        self.a, self.b, self.c, self.d = 1., 0.2, 0.3, 1.

        if __debug__:
            check_consistency(mQ=self.mQ, Pz00=self.Pz00)

    # ------------------------------------------------------------------
    def _gx(self, x, y, t, u, dt):
        """
        Nonlinear state function with retro-action of observations on state.
        """
        x1 = x.flatten()[0]
        y1 = y.flatten()[0]
        t1 = t.flatten()[0]
        return np.array([
              self.a * x1 + self.b * np.tanh(y1) + t1
        ]).reshape(-1, 1)

    # ------------------------------------------------------------------
    def _gy(self, x, y, t, u, dt):
        """
        Nonlinear state function with retro-action of states on observation.
        """

        x1 = x.flatten()[0]
        y1 = y.flatten()[0]
        u1 = u.flatten()[0]
        return np.array([
            self.c * y1 + self.d * np.sin(x1) + u1
        ]).reshape(-1, 1)

    # ------------------------------------------------------------------
    def _g(self, x, y, t, u, dt):
        """
        Combined state and observation using Wojciech’s formulation.
        """
        if __debug__:
            assert x.shape == (1, 1), f"x must be (1, 1), got {x.shape}"
            assert y.shape == (1, 1), f"y must be (1, 1), got {y.shape}"
            assert t.shape == (1, 1), f"t must be (1, 1), got {t.shape}"
            assert u.shape == (1, 1), f"u must be (1, 1), got {u.shape}"
            assert isinstance(dt, (float, int)), "dt must be a float"

        gx_val = self._gx(x, y, t, u, dt)
        gy_val = self._gy(x, y, t, u, dt)
        return np.vstack((gx_val, gy_val))

    # ------------------------------------------------------------------

    def _jacobiens_g(self, x, y, t, u, dt):
        if __debug__:
            assert x.shape == (1, 1), f"x must be (1, 1), got {x.shape}"
            assert y.shape == (1, 1), f"y must be (1, 1), got {y.shape}"
            assert t.shape == (1, 1), f"t must be (1, 1), got {t.shape}"
            assert u.shape == (1, 1), f"u must be (1, 1), got {u.shape}"
            assert isinstance(dt, (float, int)), "dt must be a float"

        x1 = x.flatten()[0]
        y1 = y.flatten()[0]

        An = np.array([
                [self.a,              self.b / np.cosh(y1)**2],
                [self.d * np.cos(x1), self.c                 ]
            ])

        Bn = np.eye(self.dim_xy)
        
        # print(f'An={An}')
        # print(f'Bn={Bn}')
        # exit(1)
        return An, Bn