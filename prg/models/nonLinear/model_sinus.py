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
        
        self.z00 = np.zeros((self.dim_xy, 1)) - 1.0
        self.Pz00 = np.eye(self.dim_xy)

        self.check_consistency()

    def fx(self, x: np.ndarray, noise: np.ndarray, dt: float) -> np.ndarray:
        """State transition with additive noise."""
        return 0.8 * x + 0.3 * np.sin(x) + noise

    def hx(self, x: np.ndarray, noise: np.ndarray, dt: float) -> np.ndarray:
        """Measurement function with additive noise."""
        return x ** 2 + noise
