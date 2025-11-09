import numpy as np
from typing import Callable
from .base_model import BaseModel
import logging

logger = logging.getLogger(__name__)

class ModelX2Y1(BaseModel):
    """
    Nonlinear model with:
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

    En mode optimisé (`python -O`), toutes les vérifications sont désactivées.
    """

    name: str = "x2_y1"

    def __init__(self) -> None:
        super().__init__(dim_x=2, dim_y=1, alpha=1e-3, beta=2.0, kappa=0.0)

        self.mQ: np.ndarray = np.array([
            [1e-2, 0.0, 0.0],
            [0.0, 1e-2, 0.0],
            [0.0, 0.0, 1e-1]
        ])
        self.z00: np.ndarray  = np.zeros((self.dim_xy, 1))
        self.Pz00: np.ndarray = np.eye(self.dim_xy)

        if __debug__:
            self.check_consistency()

    # ------------------------------------------------------------------
    def _fx(self, x: np.ndarray, nx: np.ndarray, dt: float) -> np.ndarray:
        """State transition function f(x) with process noise."""
        if __debug__:
            assert isinstance(x, np.ndarray) and x.shape == (2, 1), f"x must be (2,1), got {x.shape}"
            assert isinstance(nx, np.ndarray) and nx.shape == (2, 1), f"nx must be (2,1), got {nx.shape}"
            assert isinstance(dt, (float, int)), "dt must be a float"

        x1, x2 = x.flatten()
        nx1, nx2 = nx.flatten()
        return np.array([
            x1 + 0.05 * x2 + 0.5 * np.sin(0.1 * x2) + nx1,
            0.9 * x2 + 0.2 * np.cos(0.3 * x1) + nx2
        ]).reshape(-1, 1)

    # ------------------------------------------------------------------
    def _hx(self, x: np.ndarray, ny: np.ndarray, dt: float) -> np.ndarray:
        """Measurement function h(x) with observation noise."""
        if __debug__:
            assert isinstance(x, np.ndarray) and x.shape == (2, 1), f"x must be (2,1), got {x.shape}"
            assert isinstance(ny, np.ndarray) and ny.shape == (1, 1), f"ny must be (1,1), got {ny.shape}"
            assert isinstance(dt, (float, int)), "dt must be a float"

        x1, x2 = x.flatten()
        ny_val = ny.flatten()[0]
        return np.array([np.sqrt(x1**2 + x2**2) + ny_val]).reshape(-1, 1)

    # ------------------------------------------------------------------
    def _g(self, x: np.ndarray, y: np.ndarray, nx: np.ndarray, ny: np.ndarray, dt: float) -> np.ndarray:
        """Combine state and observation using Wojciech’s formulation."""
        if __debug__:
            assert x.shape == (2, 1), f"x must be (2,1), got {x.shape}"
            assert y.shape == (1, 1), f"y must be (1,1), got {y.shape}"
            assert nx.shape == (2, 1), f"nx must be (2,1), got {nx.shape}"
            assert ny.shape == (1, 1), f"ny must be (1,1), got {ny.shape}"
            assert isinstance(dt, (float, int)), "dt must be a float"

        fx_val: np.ndarray = self._fx(x, nx, dt)
        hx_val: np.ndarray = self._hx(fx_val, ny, dt)
        g_val: np.ndarray = np.vstack((fx_val, hx_val))
        return g_val
