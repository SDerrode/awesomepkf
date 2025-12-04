import numpy as np
from typing import Callable
from .base_model_nonLinear import BaseModelNonLinear

# A few utils functions that are used several times
from others.utils import check_consistency

class ModelExtSaturant(BaseModelNonLinear):
    """
    Nonlinear model with saturation and logarithmic observation:
        f(x) = 0.5*x + 2*(1 - exp(-0.1*x))
        h(x) = log(1 + |x|)
    """
    MODEL_NAME = "x1_y1_ext_saturant"

    def __init__(self):
        super().__init__(dim_x=1, dim_y=1, model_type="nonlinear")

        # Covariances et initialisation
        self.mQ   = np.diag([1E-4, 1E-4])
        self.z00  = np.zeros((self.dim_xy, 1))
        self.Pz00 = np.eye(self.dim_xy) * 2

        if __debug__:
            check_consistency(mQ=self.mQ, Pz00=self.Pz00)

    # ------------------------------------------------------------------
    def _fx(self, x, t, dt):
        """State transition with additive noise."""
        x1 = x.flatten()[0]
        t1 = t.flatten()[0]
        return 0.5*x1 + 2.* (1. - np.exp(-0.1 * x1)) + t1

    # ------------------------------------------------------------------
    def _hx(self, x, u, dt):
        """Measurement function with additive noise."""
        x1 = x.flatten()[0]
        u1 = u.flatten()[0]
        # utilisation de np.maximum pour éviter log(0)
        return np.log(1. + np.maximum(np.abs(x1), 1e-8)) + u1

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
        
        A = 0.5*x1 + 2.* (1. - np.exp(-0.1 * x1)) + t1
        An = np.array([[0.5+0.2*np.exp(-0.1*x1),                             0.],
                       [np.sign(A)*(0.5+0.2*np.exp(-0.1*x1))/(1.+np.abs(A)), 0.]])
        Bn = np.array([[1.,                        0.],
                       [np.sign(A)/(1.+np.abs(A)), 1.]])
        
        # print(f'An={An}')
        # print(f'Bn={Bn}')
        # exit(1)
        return An, Bn
