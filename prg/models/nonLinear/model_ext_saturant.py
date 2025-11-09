import numpy as np
from typing import Callable
from .base_model import BaseModel

class ModelExtSaturant(BaseModel):
    """
    Nonlinear model with saturation and logarithmic observation:
        f(x) = 0.5*x + 2*(1 - exp(-0.1*x))
        h(x) = log(1 + |x|)
    """
    name = "x1_y1_ext_saturant"

    def __init__(self):
        super().__init__(dim_x=1, dim_y=1)
        self.mQ = np.array([[1e-4, 0.],
                            [0.,   1e-3]])
        
        self.z00  = np.zeros((self.dim_xy, 1)) + 0.5
        self.Pz00 = np.eye(self.dim_xy) * 2
        self.check_consistency()
    
    def _fx(self, x: np.ndarray, nx: np.ndarray, dt: float) -> np.ndarray:
        """State transition with additive noise."""
        return 0.5*x + 2*(1 - np.exp(-0.1 * x)) + nx

    def _hx(self, x: np.ndarray, ny: np.ndarray, dt: float) -> np.ndarray:
        """Measurement function with additive noise."""
        return np.log(1 + np.maximum(np.abs(x), 1e-8)) + ny

    def _g(self, x, y, nx, ny, dt):
        """The model is classical and re-written using Wojciech formulation"""
        fx_val = self._fx(x, nx, dt)
        hx_val = self._hx(fx_val, ny, dt)
        g_val  = np.vstack((fx_val, hx_val))
        return g_val