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
        
        self.z00 = np.zeros((self.dim_xy, 1))
        self.Pz00 = np.eye(self.dim_xy)

        self.check_consistency()
    
    def fx(self, x: np.ndarray, noise: np.ndarray, dt: float) -> np.ndarray:
        """State transition with cubic nonlinearity and additive noise."""
        return 0.9 * x - 0.2 * x**3 + noise

    def hx(self, x: np.ndarray, noise: np.ndarray, dt: float) -> np.ndarray:
        """Linear observation function with additive noise."""
        return x + noise
