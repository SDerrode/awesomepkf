import numpy as np
from typing import Callable
from .base_model_nonLinear import BaseModelNonLinear

# A few utils functions that are used several times
from others.utils import check_consistency

class ModelX2Y1_withRetroactionsOfObservations(BaseModelNonLinear):
    """
    Nonlinear model with retro-actions of observations.
    The model includes additive Gaussian process and observation noises.
    """

    MODEL_NAME: str = "x2_y1_withRetroactionsOfObservations"

    def __init__(self) -> None:
        super().__init__(dim_x=2, dim_y=1, model_type="nonlinear")

        self.mQ   = np.diag([1E-4, 1E-4, 1e-4])
        self.z00  = np.zeros((self.dim_xy, 1))
        self.Pz00 = np.eye(self.dim_xy)

        if __debug__:
            check_consistency(mQ=self.mQ, Pz00=self.Pz00)

    # ------------------------------------------------------------------
    def _gx(self, x, y, t, u, dt):
        """
        Nonlinear state function with retro-action on observation.
        """

        x1, x2 = x.flatten()
        y1     = y.flatten()[0]
        t1, t2 = t.flatten()

        return np.array([
             0.5 * x1 + 0.1 * x2 + 0.3 * np.tanh(y1) + t1,
             0.8 * x2            - 0.2 * np.sin(y1)  + t2
        ]).reshape(-1, 1)

    # ------------------------------------------------------------------
    def _gy(self, x, y, t, u, dt):
        """
        Nonlinear observation function with retro-action on previous observation.
        """

        x1, x2 = x.flatten()
        y1     = y.flatten()[0]
        u     = u.flatten()[0]

        return np.array([
            x1**2 + 0.5*y1 + u
        ]).reshape(-1, 1)

    # ------------------------------------------------------------------
    def _g(self, x, y, t, u, dt):
        """
        Combined state and observation using Wojciech’s formulation.
        """
        if __debug__:
            assert x.shape == (2, 1), f"x must be (2,1), got {x.shape}"
            assert y.shape == (1, 1), f"y must be (1,1), got {y.shape}"
            assert t.shape == (2, 1), f"t must be (2,1), got {t.shape}"
            assert u.shape == (1, 1), f"u must be (1,1), got {u.shape}"
            assert isinstance(dt, (float, int)), "dt must be a float"

        gx_val = self._gx(x, y, t, u, dt)
        gy_val = self._gy(x, y, t, u, dt)
        return np.vstack((gx_val, gy_val))

    # ------------------------------------------------------------------

    def _jacobiens_g(self, x, y, t, u, dt):
        if __debug__:
            assert x.shape == (2, 1), f"x must be (2,1), got {x.shape}"
            assert y.shape == (1, 1), f"y must be (1,1), got {y.shape}"
            assert t.shape == (2, 1), f"t must be (2,1), got {t.shape}"
            assert u.shape == (1, 1), f"u must be (1,1), got {u.shape}"
            assert isinstance(dt, (float, int)), "dt must be a float"

        x1, x2   = x.flatten()
        # t1, t2 = t.flatten()
        y1       = y.flatten()[0]
        # u       = u.flatten()[0]

        An = np.array([[0.5,   0.1,  0.3*(1.-np.tanh(y1)**2)],
                       [0.,    0.8, -0.2*np.cos(y1)         ],
                       [2.*x1, 0.,   0.5]                   ])

        Bn = np.eye(self.dim_xy)
        
        # print(f'An={An}')
        # print(f'Bn={Bn}')
        # exit(1)
        return An, Bn