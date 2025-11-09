import numpy as np
from typing import Callable
from .base_model import BaseModel
import logging

logger = logging.getLogger(__name__)

class ModelExtSaturant(BaseModel):
    """
    Nonlinear model with saturation and logarithmic observation:
        f(x) = 0.5*x + 2*(1 - exp(-0.1*x))
        h(x) = log(1 + |x|)

    En mode optimisé (`python -O`), toutes les vérifications sont désactivées.
    """
    name = "x1_y1_ext_saturant"

    def __init__(self):
        super().__init__(dim_x=1, dim_y=1)

        # Covariances et initialisation
        self.mQ = np.array([
            [1e-4, 0.0],
            [0.0,  1e-3]
        ])
        self.z00  = np.zeros((self.dim_xy, 1)) + 0.5
        self.Pz00 = np.eye(self.dim_xy) * 2

        if __debug__:
            self.check_consistency()

    # ------------------------------------------------------------------
    def _fx(self, x: np.ndarray, nx: np.ndarray, dt: float) -> np.ndarray:
        """State transition with additive noise."""
        if __debug__:
            assert isinstance(x, np.ndarray) and x.shape == (1, 1), f"x doit être (1,1), reçu {x.shape}"
            assert isinstance(nx, np.ndarray) and nx.shape == (1, 1), f"nx doit être (1,1), reçu {nx.shape}"
            assert isinstance(dt, (int, float)), "dt doit être un scalaire numérique"

        return 0.5 * x + 2 * (1 - np.exp(-0.1 * x)) + nx

    # ------------------------------------------------------------------
    def _hx(self, x: np.ndarray, ny: np.ndarray, dt: float) -> np.ndarray:
        """Measurement function with additive noise."""
        if __debug__:
            assert isinstance(x, np.ndarray) and x.shape == (1, 1), f"x doit être (1,1), reçu {x.shape}"
            assert isinstance(ny, np.ndarray) and ny.shape == (1, 1), f"ny doit être (1,1), reçu {ny.shape}"
            assert isinstance(dt, (int, float)), "dt doit être un scalaire numérique"

        # utilisation de np.maximum pour éviter log(0)
        return np.log(1 + np.maximum(np.abs(x), 1e-8)) + ny

    # ------------------------------------------------------------------
    def _g(
        self,
        x: np.ndarray,
        y: np.ndarray,
        nx: np.ndarray,
        ny: np.ndarray,
        dt: float
    ) -> np.ndarray:
        """Combine state and observation into stacked z_{n+1} vector."""
        if __debug__:
            assert x.shape == (1, 1), f"x doit être (1,1), reçu {x.shape}"
            assert y.shape == (1, 1), f"y doit être (1,1), reçu {y.shape}"
            assert nx.shape == (1, 1), f"nx doit être (1,1), reçu {nx.shape}"
            assert ny.shape == (1, 1), f"ny doit être (1,1), reçu {ny.shape}"

        fx_val = self._fx(x, nx, dt)
        hx_val = self._hx(fx_val, ny, dt)
        g_val  = np.vstack((fx_val, hx_val))
        return g_val
