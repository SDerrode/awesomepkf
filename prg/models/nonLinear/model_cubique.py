import numpy as np
from typing import Callable
from .base_model_nonLinear import BaseModelNonLinear

# A few utils functions that are used several times
from others.utils import check_consistency


class ModelCubique(BaseModelNonLinear):
    """
    Cubic nonlinear model with 1D state and 1D observation:
        f(x) = 0.9*x - 0.2*x^3
        h(x) = x
    """
    MODEL_NAME = "x1_y1_cubique"

    def __init__(self):
        super().__init__(dim_x=1, dim_y=1, model_type="nonlinear")
        
        self.mQ   = np.diag([1E-4, 1E-4])
        self.z00  = np.zeros((self.dim_xy, 1))
        self.Pz00 = np.eye(self.dim_xy)

        if __debug__:  # ⚙️ ignoré en mode -O
            check_consistency(mQ=self.mQ, Pz00=self.Pz00)

    # ------------------------------------------------------------------
    def _fx(self, x, t, dt):
        """State transition with cubic nonlinearity and additive noise."""
        x1 = x.flatten()[0]
        t1 = t.flatten()[0]
        return 0.9*x1 - 0.2*x1**3 + t1

    # ------------------------------------------------------------------
    def _hx(self, x, u, dt):
        """Linear observation function with additive noise."""
        x1 = x.flatten()[0]
        u1 = u.flatten()[0]
        return x1 + u1

    # ------------------------------------------------------------------
    def _g(self, x, y, t, u, dt):
        """Combine state and observation using Wojciech’s formulation."""
        if __debug__:
            assert x.shape == (1, 1), f"x doit être (1,1), reçu {x.shape}"
            assert y.shape == (1, 1), f"y doit être (1,1), reçu {y.shape}"
            assert t.shape == (1, 1), f"t doit être (1,1), reçu {t.shape}"
            assert u.shape == (1, 1), f"u doit être (1,1), reçu {u.shape}"

        fx_val = self._fx(x,      t, dt)
        hx_val = self._hx(fx_val, u, dt)
        return np.vstack((fx_val, hx_val))

    # ------------------------------------------------------------------
    def _jacobiens_g(self, x, y, t, u, dt):
        if __debug__:
            assert x.shape == (1, 1), f"x doit être (1,1), reçu {x.shape}"
            assert y.shape == (1, 1), f"y doit être (1,1), reçu {y.shape}"
            assert t.shape == (1, 1), f"t doit être (1,1), reçu {t.shape}"
            assert u.shape == (1, 1), f"u doit être (1,1), reçu {u.shape}"
        x1 = x.flatten()[0]
        t1 = t.flatten()[0]
        y1 = y.flatten()[0]
        #u1 = u.flatten()[0]
        
        An = np.array([[0.9-0.6*x1**2,        0.],
                       [0.9-0.6*x1**2, 0.]])
        Bn = np.array([[1.,   0.],
                       [1., 1.]])
        
        # print(f'An={An}')
        # print(f'Bn={Bn}')
        # exit(1)
        return An, Bn