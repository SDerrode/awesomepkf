import numpy as np
from typing import Callable
from .base_model import BaseModel

class ModelCubique(BaseModel):
    """
    Cubic nonlinear model with 1D state and 1D observation:
        f(x) = 0.9*x - 0.2*x^3
        h(x) = x
    """
    name = "x1_y1_cubique"

    def __init__(self):
        super().__init__(dim_x=1, dim_y=1)
        self.mQ = np.array([[1e-2, 0.],
                            [0.,   1e-1]])
        
        self.z00  = np.zeros((self.dim_xy, 1))
        self.Pz00 = np.eye(self.dim_xy)
        self.check_consistency()
    
    def _fx(self, x: np.ndarray, nx: np.ndarray, dt: float) -> np.ndarray:
        """State transition with cubic nonlinearity and additive noise."""
        return 0.9 * x - 0.2 * x**3 + nx

    def _hx(self, x: np.ndarray, ny: np.ndarray, dt: float) -> np.ndarray:
        """Linear observation function with additive noise."""
        return x + ny
    
    def _g(self, x, y, nx, ny, dt):
        """The model is classical and re-written using Wojciech formulation"""
        fx_val = self._fx(x, nx, dt)
        hx_val = self._hx(fx_val, ny, dt)
        g_val  = np.vstack((fx_val, hx_val))
        return g_val
