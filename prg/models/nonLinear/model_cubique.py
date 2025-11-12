import numpy as np
from typing import Callable
from .base_model_nonLinear import BaseModelNonLinear

# A few utils functions that are used several times
from others.utils import check_consistency


class ModelCubique(BaseModelNonLinear):
    """
    Cubic nonlinear model with 1D state and 1D observation:
        f(x) = 0.9*x - 0.2*x^3
        h(x) = x

    En mode optimisé (python -O), les vérifications sont désactivées pour accélérer l'exécution.
    """
    MODEL_NAME = "x1_y1_cubique"

    def __init__(self):
        super().__init__(dim_x=1, dim_y=1, model_type="nonlinear")
        self.mQ = np.array([
            [1e-2, 0.0],
            [0.0, 1e-1]
        ])
        self.z00  = np.zeros((self.dim_xy, 1))
        self.Pz00 = np.eye(self.dim_xy)

        if __debug__:  # ⚙️ ignoré en mode -O
            check_consistency(mQ=self.mQ, Pz00=self.Pz00)


    # ------------------------------------------------------------------
    def _fx(self, x: np.ndarray, nx: np.ndarray, dt: float) -> np.ndarray:
        """State transition with cubic nonlinearity and additive noise."""
        if __debug__:
            assert isinstance(x,  np.ndarray) and x.shape  == (1, 1), f"x doit être (1,1), reçu {x.shape}"
            assert isinstance(nx, np.ndarray) and nx.shape == (1, 1), f"nx doit être (1,1), reçu {nx.shape}"
            assert isinstance(dt, (int, float)), "dt doit être un scalaire numérique"
        return 0.9 * x - 0.2 * x**3 + nx

    # ------------------------------------------------------------------
    def _hx(self, x: np.ndarray, ny: np.ndarray, dt: float) -> np.ndarray:
        """Linear observation function with additive noise."""
        if __debug__:
            assert isinstance(x,  np.ndarray) and x.shape  == (1, 1), f"x doit être (1,1), reçu {x.shape}"
            assert isinstance(ny, np.ndarray) and ny.shape == (1, 1), f"ny doit être (1,1), reçu {ny.shape}"
            assert isinstance(dt, (int, float)), "dt doit être un scalaire numérique"
        return x + ny

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
            assert x.shape  == (1, 1), f"x doit être (1,1), reçu {x.shape}"
            assert y.shape  == (1, 1), f"y doit être (1,1), reçu {y.shape}"
            assert nx.shape == (1, 1), f"nx doit être (1,1), reçu {nx.shape}"
            assert ny.shape == (1, 1), f"ny doit être (1,1), reçu {ny.shape}"

        fx_val = self._fx(x, nx, dt)
        hx_val = self._hx(fx_val, ny, dt)
        g_val  = np.vstack((fx_val, hx_val))
        return g_val
