import numpy as np
from typing import Callable
from .base_model import BaseModel

class ModelGordon(BaseModel):
    """
    Gordon et al. (1993) nonlinear model:
        f(x) = 0.5*x + 25*x/(1 + x^2) + 8*cos(1.2*t)
        h(x) = 0.05*x^2
    """

    name = "x1_y1_gordon"

    def __init__(self):
        super().__init__(dim_x=1, dim_y=1, alpha=0.01, beta=2., kappa=0.)
        
        self.mQ = np.array([
            [1e-2, 0.],
            [0.,   1e-1]
        ])
        
        self.z00 = np.zeros((self.dim_xy, 1))
        self.Pz00 = np.eye(self.dim_xy)

        self.check_consistency()

    def fx(self, x: np.ndarray, noise: np.ndarray, dt: float) -> np.ndarray:
        """State transition with additive noise."""
        return 0.5*x + 25*x / (1 + x**2) + 8*np.cos(1.2 * dt) + noise

    def hx(self, x: np.ndarray, noise: np.ndarray, dt: float) -> np.ndarray:
        """Measurement function with additive noise."""
        return 0.05 * x**2 + noise
