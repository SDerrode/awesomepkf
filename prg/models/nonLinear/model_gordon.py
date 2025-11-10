import numpy as np
from typing import Callable, Optional
from .base_model_nonLinear import BaseModelNonLinear
import logging

# A few utils functions that are used several times
from others.Utils import check_consistency

logger = logging.getLogger(__name__)

class ModelGordon(BaseModelNonLinear):
    """
    Gordon et al. (1993) nonlinear model:
        f(x) = 0.5*x + 25*x/(1 + x^2) + 8*cos(1.2*t)
        h(x) = 0.05*x^2

    En mode optimisé (`python -O`), toutes les vérifications sont supprimées.
    """

    MODEL_NAME: str = "x1_y1_gordon"

    def __init__(self) -> None:
        super().__init__(dim_x=1, dim_y=1, alpha=0.01, beta=2.0, kappa=0.0)
        
        self.mQ: np.ndarray = np.array([
            [1e-2, 0.0],
            [0.0,  1e-1]
        ])
        self.z00: np.ndarray = np.zeros((self.dim_xy, 1))
        self.Pz00: np.ndarray = np.eye(self.dim_xy)

        if __debug__:
            check_consistency(mQ=self.mQ, Pz00=self.Pz00)

    # ------------------------------------------------------------------
    def _fx(self, x: np.ndarray, nx: np.ndarray, dt: float) -> np.ndarray:
        """
        State transition with additive noise.
        Args:
            x: (dim_x, 1) state vector
            nx: (dim_x, 1) process noise
            dt: time increment
        Returns:
            New state vector (dim_x, 1)
        """
        if __debug__:
            assert isinstance(x, np.ndarray) and x.shape == (1, 1), f"x doit être (1,1), reçu {x.shape}"
            assert isinstance(nx, np.ndarray) and nx.shape == (1, 1), f"nx doit être (1,1), reçu {nx.shape}"
            assert isinstance(dt, (int, float)), "dt doit être un scalaire numérique"

        return 0.5 * x + 25 * x / (1 + x**2) + 8 * np.cos(1.2 * dt) + nx

    # ------------------------------------------------------------------
    def _hx(self, x: np.ndarray, ny: np.ndarray, dt: float) -> np.ndarray:
        """
        Measurement function with additive noise.
        Args:
            x: (dim_x, 1) state vector
            ny: (dim_y, 1) measurement noise
            dt: time increment
        Returns:
            Observation vector (dim_y, 1)
        """
        if __debug__:
            assert isinstance(x, np.ndarray) and x.shape == (1, 1), f"x doit être (1,1), reçu {x.shape}"
            assert isinstance(ny, np.ndarray) and ny.shape == (1, 1), f"ny doit être (1,1), reçu {ny.shape}"
            assert isinstance(dt, (int, float)), "dt doit être un scalaire numérique"

        return 0.05 * x**2 + ny

    # ------------------------------------------------------------------
    def _g(
        self,
        x: np.ndarray,
        y: np.ndarray,
        nx: np.ndarray,
        ny: np.ndarray,
        dt: float
    ) -> np.ndarray:
        """
        Combine state and observation using Wojciech’s formulation.
        Args:
            x, y, nx, ny: (1, 1) vectors
            dt: time increment
        Returns:
            Combined vector (dim_x + dim_y, 1)
        """
        if __debug__:
            assert x.shape == (1, 1), f"x doit être (1,1), reçu {x.shape}"
            assert y.shape == (1, 1), f"y doit être (1,1), reçu {y.shape}"
            assert nx.shape == (1, 1), f"nx doit être (1,1), reçu {nx.shape}"
            assert ny.shape == (1, 1), f"ny doit être (1,1), reçu {ny.shape}"

        fx_val: np.ndarray = self._fx(x, nx, dt)
        hx_val: np.ndarray = self._hx(fx_val, ny, dt)
        g_val: np.ndarray = np.vstack((fx_val, hx_val))
        return g_val
