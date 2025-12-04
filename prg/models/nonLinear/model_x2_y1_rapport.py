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
        self.z00    = np.zeros((self.dim_xy, 1))
        self.Pz00   = np.eye(self.dim_xy)

        if __debug__:
            check_consistency(mQ=self.mQ, Pz00=self.Pz00)

    # ------------------------------------------------------------------
    def _fx(self, x, t, dt):
        """State transition function f(x) with process noise."""

        x1, x2 = x.flatten()
        t1, t2 = t.flatten()
        return np.array([
            x1 + dt*x2                                           + t1,
            x2 - dt*(self.alpham * np.sin(x1) + self.betam * x2) + t2
        ]).reshape(-1, 1)

    # ------------------------------------------------------------------
    def _hx(self, x, u, dt):
        """Measurement function h(x) with observation noise."""

        x1, x2 = x.flatten()
        u      = u.flatten()[0]

        return np.array([
            x1**2 / (1. + x1**2) + self.gammam * np.sin(x2) + u
        ]).reshape(-1, 1)

    # ------------------------------------------------------------------
    def _g(self, x, y, t, u, dt):
        """Combine state and observation using Wojciech’s formulation."""
        if __debug__:
            assert x.shape == (2, 1), f"x must be (2,1), got {x.shape}"
            assert y.shape == (1, 1), f"y must be (1,1), got {y.shape}"
            assert t.shape == (2, 1), f"t must be (2,1), got {t.shape}"
            assert u.shape == (1, 1), f"u must be (1,1), got {u.shape}"
            assert isinstance(dt, (float, int)), "dt must be a float"

        fx_val = self._fx(x,      t, dt)
        hx_val = self._hx(fx_val, u, dt)
        return np.vstack((fx_val, hx_val))

    # ------------------------------------------------------------------
    def _jacobiens_g(self, x, y, t, u, dt):
        if __debug__:
            assert x.shape == (2, 1), f"x must be (2,1), got {x.shape}"
            assert y.shape == (1, 1), f"y must be (1,1), got {y.shape}"
            assert t.shape == (2, 1), f"t must be (2,1), got {t.shape}"
            assert u.shape == (1, 1), f"u must be (1,1), got {u.shape}"
            assert isinstance(dt, (float, int)), "dt must be a float"

        x1, x2   = x.flatten()
        t1, t2 = t.flatten()
        y1       = y.flatten()[0]
        # u       = u.flatten()[0]
        
        A = x1 + dt*x2 + t1
        B = x2 - dt*(self.alpham*np.sin(x1) + self.betam*x2) + t2
        Z = 2*A/(1.+A**2)**2
        W = self.gammam*np.cos(B)
        An = np.array([[1.,                              dt,                              0.],
                       [-self.alpham*dt*np.cos(x1),      1. - self.betam * dt,            0.],
                       [Z - self.alpham*dt*np.cos(x1)*W, Z*dt + (1. - self.betam * dt)*W, 0.]])
        Bn = np.array([[1., 0., 0.],
                       [0., 1., 0.],
                       [Z,  W,  1.]])
        
        # dg1dx1 = 1
        # dg1dx2 = dt
        # dg2dx1 = -dt*self.alpham*np.cos(x[0])
        # dg2dx2 = 1 - dt*self.betam
        # Fn = np.array([[dg1dx1, dg1dx2], [dg2dx1.item(), dg2dx2]])
        # dg3dx1 = 2*x1[0]/(1.+x1[0]**2)**2
        # dg3dx2 = self.gammam*np.cos(x1[1])
        # Hn = np.array([[dg3dx1.item(), dg3dx2.item()]])
        
        # An = np.block([[Fn, np.zeros((2,1))],[Hn@Fn, np.zeros((1,1))]])
        # Bn = np.block([[np.eye(2), np.zeros((2,1))], [Hn, np.eye(1)]])
        
        # print(f'An={An}')
        # print(f'Bn={Bn}')
        # exit(1)
        return An, Bn