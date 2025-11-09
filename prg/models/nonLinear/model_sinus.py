import numpy as np
from typing import Callable
from .base_model import BaseModel

class ModelSinus(BaseModel):
    """
    Nonlinear model with 1D state and 1D observation:
        f(x) = 0.8*x + 0.3*sin(x)
        h(x) = x^2
    """
    name = "x1_y1_sinus"

    def __init__(self):
        super().__init__(dim_x=1, dim_y=1, alpha=0.01, beta=2., kappa=0.)
        
        self.mQ = np.array([
            [1e-3, 0.],
            [0.,   1e-2]
        ])
        
        self.z00  = np.zeros((self.dim_xy, 1)) - 1.0
        self.Pz00 = np.eye(self.dim_xy)
        self.check_consistency()

    def _fx(self, x: np.ndarray, nx: np.ndarray, dt: float) -> np.ndarray:
        """State transition with additive noise."""
        return 0.8 * x + 0.3 * np.sin(x) + nx

    def _hx(self, x: np.ndarray, ny: np.ndarray, dt: float) -> np.ndarray:
        """Measurement function with additive noise."""
        return x ** 2 + ny

    def _g(self, x, y, nx, ny, dt):
        """The model is classical and re-written using Wojciech formulation"""
        fx_val = self._fx(x, nx, dt)
        hx_val = self._hx(fx_val, ny, dt)
        g_val  = np.vstack((fx_val, hx_val))
        return g_val