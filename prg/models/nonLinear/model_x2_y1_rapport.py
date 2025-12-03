import numpy as np
from typing import Callable
from .base_model_nonLinear import BaseModelNonLinear

# A few utils functions that are used several times
from others.utils import check_consistency

class ModelX2Y1(BaseModelNonLinear):
    """
    Nonlinear model with:
      - 2-dimensional state vector x = [x1, x2]
      - 1-dimensional observation y

    System dynamics (nonlinear):
        f(x) = [
            x1 + T * x2,
            x2 - T*(alpha * sin(x1) + beta * x2)
        ]

    Measurement equation (non linear):
        h(x) = x1**2 / (1 + x1**2) + gamma * sin(x2)

    The model includes additive Gaussian process and observation noises.
    """

    MODEL_NAME: str = "x2_y1_rapport"

    def __init__(self) -> None:
        super().__init__(dim_x=2, dim_y=1, model_type="nonlinear")

        self.alpham = 1.
        self.betam  = 0.1
        self.gammam = 0.5
        self.mQ     = np.diag([1E-4, 1E-4, 1e-4])

        self.z00  = np.zeros((self.dim_xy, 1))
        self.Pz00 = np.eye(self.dim_xy)

        if __debug__:
            check_consistency(mQ=self.mQ, Pz00=self.Pz00)

    # ------------------------------------------------------------------
    def _fx(self, x: np.ndarray, nx: np.ndarray, dt: float) -> np.ndarray:
        """State transition function f(x) with process noise."""

        x1, x2   = x.flatten()
        nx1, nx2 = nx.flatten()
        return np.array([
            x1 + dt*x2                                         + nx1,
            x2 - dt*(self.alpham * np.sin(x1) + self.betam * x2) + nx2
        ]).reshape(-1, 1)


    # ------------------------------------------------------------------
    def _hx(self, x: np.ndarray, ny: np.ndarray, dt: float) -> np.ndarray:
        """Measurement function h(x) with observation noise."""

        x1, x2 = x.flatten()
        return np.array([
            x1**2 / (1. + x1**2) + self.gammam * np.sin(x2) + ny
        ]).reshape(-1, 1)

    # ------------------------------------------------------------------
    def _g(self, x: np.ndarray, y: np.ndarray, nx: np.ndarray, ny: np.ndarray, dt: float) -> np.ndarray:
        """Combine state and observation using Wojciech’s formulation."""
        if __debug__:
            assert x.shape  == (2, 1), f"x  must be (2,1), got {x.shape}"
            assert y.shape  == (1, 1), f"y  must be (1,1), got {y.shape}"
            assert nx.shape == (2, 1), f"nx must be (2,1), got {nx.shape}"
            assert ny.shape == (1, 1), f"ny must be (1,1), got {ny.shape}"
            assert isinstance(dt, (float, int)), "dt must be a float"

        fx_val = self._fx(x,      nx, dt)
        hx_val = self._hx(fx_val, ny, dt)
        g_val  = np.vstack((fx_val, hx_val))
        return g_val

    # ------------------------------------------------------------------
    def _jacobiens_g(self, x, y, x1, dt):
        dg1dx1 = 1
        dg1dx2 = dt
        dg2dx1 = -dt*self.alpham*np.cos(x[0])
        dg2dx2 = 1 - dt*self.betam
        F = np.array([[dg1dx1, dg1dx2], [dg2dx1.item(), dg2dx2]])
        dg3dx1 = 2*x1[0]/(1.+x1[0]**2)**2
        dg3dx2 = self.gammam*np.cos(x1[1])
        H = np.array([[dg3dx1.item(), dg3dx2.item()]])
        return np.block([[F, np.zeros((2,1))],[H@F, np.zeros((1,1))]]), np.block([[np.eye(2), np.zeros((2,1))], [H, np.eye(1)]])