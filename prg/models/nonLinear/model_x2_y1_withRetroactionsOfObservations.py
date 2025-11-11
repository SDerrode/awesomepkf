import numpy as np
from typing import Callable
from .base_model_nonLinear import BaseModelNonLinear

# A few utils functions that are used several times
from others.Utils import check_consistency

class ModelX2Y1_withRetroactionsOfObservations(BaseModelNonLinear):
    """
    Nonlinear model with retro-actions of observations.
    The model includes additive Gaussian process and observation noises.
    
    En mode optimisé (`python -O`), toutes les vérifications sont désactivées.
    """

    MODEL_NAME: str = "x2_y1_withRetroactionsOfObservations"

    def __init__(self) -> None:
        super().__init__(dim_x=2, dim_y=1, alpha=1e-3, beta=2.0, kappa=0.0, model_type="nonlinear")

        self.mQ: np.ndarray = np.array([
            [1e-2, 0.0, 0.0],
            [0.0, 1e-2, 0.0],
            [0.0, 0.0, 1e-1]
        ])
        self.z00: np.ndarray  = np.zeros((self.dim_xy, 1))
        self.Pz00: np.ndarray = np.eye(self.dim_xy)

        if __debug__:
            check_consistency(mQ=self.mQ, Pz00=self.Pz00)

    # ------------------------------------------------------------------
    def _fxy(self, x: np.ndarray, nx: np.ndarray, y: np.ndarray, dt: float) -> np.ndarray:
        """
        Nonlinear state function with retro-action on observation.

        Args:
            x: state vector (dim_x=2, 1)
            nx: process noise (dim_x=2, 1)
            y: observation at time k (dim_y=1, 1)
            dt: time increment

        Returns:
            np.ndarray: predicted state vector x_{k+1} (dim_x=2, 1)
        """
        if __debug__:
            assert isinstance(x, np.ndarray) and x.shape == (2, 1), f"x must be (2,1), got {x.shape}"
            assert isinstance(nx, np.ndarray) and nx.shape == (2, 1), f"nx must be (2,1), got {nx.shape}"
            assert isinstance(y, np.ndarray) and y.shape == (1, 1), f"y must be (1,1), got {y.shape}"
            assert isinstance(dt, (float, int)), "dt must be a float"

        nx1, nx2 = nx.flatten()
        x1, x2   = x.flatten()
        y_val    = y.flatten()[0]

        x1_next = 0.5 * x1 + 0.1 * x2 + 0.3 * np.tanh(y_val) + nx1
        x2_next = 0.8 * x2            - 0.2 * np.sin(y_val) + nx2
        return np.array([x1_next, x2_next]).reshape(-1, 1)

    # ------------------------------------------------------------------
    def _hxy(self, x: np.ndarray, ny: np.ndarray, y: np.ndarray, dt: float) -> np.ndarray:
        """
        Nonlinear observation function with retro-action on previous observation.

        Args:
            x: state vector (dim_x=2, 1)
            ny: measurement noise (dim_y=1, 1)
            y: previous observation (dim_y=1, 1)
            dt: time increment

        Returns:
            np.ndarray: observation y_k (dim_y=1, 1)
        """
        if __debug__:
            assert isinstance(x, np.ndarray) and x.shape == (2, 1), f"x must be (2,1), got {x.shape}"
            assert isinstance(ny, np.ndarray) and ny.shape == (1, 1), f"ny must be (1,1), got {ny.shape}"
            assert isinstance(y, np.ndarray) and y.shape == (1, 1), f"y must be (1,1), got {y.shape}"
            assert isinstance(dt, (float, int)), "dt must be a float"

        x1, x2 = x.flatten()
        y_val  = y.flatten()[0]
        ny_val = ny.flatten()[0]

        y_next = x1**2 + 0.5 * y_val + ny_val
        return np.array([y_next]).reshape(-1, 1)

    # ------------------------------------------------------------------
    def _g(self, x: np.ndarray, y: np.ndarray, nx: np.ndarray, ny: np.ndarray, dt: float) -> np.ndarray:
        """
        Combined state and observation using Wojciech’s formulation.

        Returns:
            np.ndarray: stacked vector z_{k+1} = [x_{k+1}; y_{k+1}] (dim_xy, 1)
        """
        if __debug__:
            assert x.shape == (2, 1), f"x must be (2,1), got {x.shape}"
            assert y.shape == (1, 1), f"y must be (1,1), got {y.shape}"
            assert nx.shape == (2, 1), f"nx must be (2,1), got {nx.shape}"
            assert ny.shape == (1, 1), f"ny must be (1,1), got {ny.shape}"
            assert isinstance(dt, (float, int)), "dt must be a float"

        fxy_val = self._fxy(x, nx, y, dt)
        hxy_val = self._hxy(fxy_val, ny, y, dt)
        g_val   = np.vstack((fxy_val, hxy_val))
        return g_val
