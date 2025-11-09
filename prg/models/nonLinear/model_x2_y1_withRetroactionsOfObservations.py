import numpy as np
from typing import Callable
from .base_model import BaseModel

class ModelX2Y1_withRetroactionsOfObservations(BaseModel):
    """
    Non linear model with retro-actions of observations
    The model includes additive Gaussian process and observation noises.
    """

    name = "x2_y1_withRetroactionsOfObservations"

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

    def _fxy(self, x: np.ndarray, nx: np.ndarray, y: np.ndarray, dt: float) -> np.ndarray:
        """
        Fonction d'état non linéaire avec rétroaction sur l'observation.

        Args:
            x: np.ndarray, vecteur d'état à l'instant k (dimension 2)
            y: float, observation à l'instant k

        Returns:
            np.ndarray: vecteur d'état prédit x_{k+1} (dimension 2)
        """
        
        nx1, nx2 = nx.flatten()
        x1, x2   =  x.flatten()
        x1_next = 0.5 * x1 + 0.1 * x2 + 0.3 * np.tanh(y.flatten()[0]) + nx1
        x2_next = 0.8 * x2            - 0.2 * np.sin(y.flatten()[0])  + nx2
        return np.array([x1_next, x2_next]).reshape(-1, 1)
  

    def _hxy(self, x: np.ndarray, ny: np.ndarray, y: np.ndarray, dt: float) -> np.ndarray:
        """
        Fonction d'observation non linéaire avec rétroaction sur l'observation précédente.

        Args:
            x: np.ndarray, vecteur d'état à l'instant k (dimension 2)
            y: float, observation à l'instant k-1

        Returns:
            float: observation y_k
        """
        
        x1, x2 = x
        y = x1**2 + 0.5 * y.flatten()[0] + ny.flatten()[0]
        return np.array([y]).reshape(-1,1)


    def _g(self, x, y, nx, ny, dt):
        """The model is classical and re-written using Wojciech formulation"""
        fxy_val = self._fxy(x,       nx, y, dt)
        hxy_val = self._hxy(fxy_val, ny, y, dt)
        g_val   = np.vstack((fxy_val, hxy_val))
        return g_val