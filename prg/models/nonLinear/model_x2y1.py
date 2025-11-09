import numpy as np
from typing import Callable
from .base_model import BaseModel

class ModelX2Y1(BaseModel):
    """
    Non linear model with:
      - 2-dimensional state vector x = [x1, x2]
      - 1-dimensional observation y

    System dynamics (nonlinear):
        f(x) = [
            x1 + 0.05 * x2 + 0.5 * sin(0.1 * x2),
            0.9 * x2 + 0.2 * cos(0.3 * x1)
        ]

    Measurement equation:
        h(x) = sqrt(x1^2 + x2^2)

    The model includes additive Gaussian process and observation noise.
    """

    name = "x2_y1"

    def __init__(self):
        super().__init__(dim_x=2, dim_y=1, alpha=1e-3, beta=2., kappa=0.)

        self.mQ = np.array([
            [1e-2, 0.,   0.],
            [0.,   1e-2, 0.],
            [0.,   0.,   1e-1]
        ])

        self.z00  = np.zeros((self.dim_xy, 1))
        self.Pz00 = np.eye(self.dim_xy)
        self.check_consistency()

    def fx(self, x: np.ndarray, noise: np.ndarray, dt: float) -> np.ndarray:
        """State transition function f(x) with process noise."""
        x1, x2 = x.flatten()
        nx1, nx2 = noise.flatten()
        return np.array([
            x1 + 0.05 * x2 + 0.5 * np.sin(0.1 * x2) + nx1,
            0.9 * x2 + 0.2 * np.cos(0.3 * x1) + nx2
        ]).reshape(-1, 1)

    def hx(self, x: np.ndarray, noise: np.ndarray, dt: float) -> np.ndarray:
        """Measurement function h(x) with observation noise."""
        x1, x2 = x.flatten()
        return np.array([
            np.sqrt(x1**2 + x2**2) + noise.flatten()[0]
        ]).reshape(-1, 1)