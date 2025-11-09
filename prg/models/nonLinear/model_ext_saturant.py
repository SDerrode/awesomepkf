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
        
        self.z00 = np.zeros((self.dim_xy, 1)) + 0.5
        self.Pz00 = np.eye(self.dim_xy) * 2

        self.check_consistency()
    
    def fx(self, x: np.ndarray, noise: np.ndarray, dt: float) -> np.ndarray:
        """State transition with additive noise."""
        return 0.5*x + 2*(1 - np.exp(-0.1 * x)) + noise

    def hx(self, x: np.ndarray, noise: np.ndarray, dt: float) -> np.ndarray:
        """Measurement function with additive noise."""
        return np.log(1 + np.maximum(np.abs(x), 1e-8)) + noise
