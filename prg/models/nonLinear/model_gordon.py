import numpy as np
from .base_model_nonLinear import BaseModelNonLinear
from others.utils import check_consistency

class ModelGordon(BaseModelNonLinear):
    """
    Gordon et al. (1993) nonlinear model:

    State dynamics:
        f(x) = 0.5*x + 25*x/(1 + x^2) + 8*cos(1.2*dt)

    Measurement equation:
        h(x) = 0.05*x^2

    Includes additive Gaussian process and observation noise.
    """

    MODEL_NAME: str = "x1_y1_gordon"

    def __init__(self) -> None:
        super().__init__(dim_x=1, dim_y=1, model_type="nonlinear")
        
        # Covariance and initial state
        self.mQ   = np.diag([1e-4, 1e-4])
        self.z00  = np.zeros((self.dim_xy, 1))
        self.Pz00 = np.eye(self.dim_xy)

        if __debug__:
            check_consistency(mQ=self.mQ, Pz00=self.Pz00)

    # ------------------------------------------------------------------
    def _fx(self, x: np.ndarray, t: np.ndarray, dt: float) -> np.ndarray:
        """
        State transition function with additive process noise.

        Args:
            x : np.ndarray, shape (1,1) - current state
            t : np.ndarray, shape (1,1) - process noise
            dt: float - timestep

        Returns:
            np.ndarray, shape (1,1) - next state
        """
        x1 = x.flatten()[0]
        t1 = t.flatten()[0]
        return np.array([[0.5 * x1 + 25 * x1 / (1. + x1**2) + 8 * np.cos(1.2 * dt) + t1]])

    # ------------------------------------------------------------------
    def _hx(self, x: np.ndarray, u: np.ndarray, dt: float) -> np.ndarray:
        """
        Measurement function with additive observation noise.

        Args:
            x : np.ndarray, shape (1,1) - predicted state
            u : np.ndarray, shape (1,1) - observation noise
            dt: float - timestep

        Returns:
            np.ndarray, shape (1,1) - measurement
        """
        x1 = x.flatten()[0]
        u1 = u.flatten()[0]
        return np.array([[0.05 * x1**2 + u1]])

    # ------------------------------------------------------------------
    def _g(self, x: np.ndarray, y: np.ndarray, t: np.ndarray, u: np.ndarray, dt: float) -> np.ndarray:
        """
        Combine state and observation using Wojciech's formulation.

        Returns:
            np.ndarray, shape (2,1) - stacked state + observation
        """
        if __debug__:
            assert x.shape == (1,1)
            assert y.shape == (1,1)
            assert t.shape == (1,1)
            assert u.shape == (1,1)

        fx_val = self._fx(x, t, dt)
        hx_val = self._hx(fx_val, u, dt)
        return np.vstack((fx_val, hx_val))

    # ------------------------------------------------------------------
    def _jacobiens_g(self, x: np.ndarray, y: np.ndarray, t: np.ndarray, u: np.ndarray, dt: float):
        """
        Compute Jacobians of g w.r.t state and noise.

        Returns:
            Tuple[np.ndarray, np.ndarray] : (dg/dz, dg/dnoise)
        """
        if __debug__:
            assert x.shape == (1,1)
            assert y.shape == (1,1)
            assert t.shape == (1,1)
            assert u.shape == (1,1)

        x1 = x.flatten()[0]
        t1 = t.flatten()[0]
        y1 = y.flatten()[0]

        # State plus noise
        A = 0.5 * x1 + 25 * x1 / (1. + x1**2) + 8 * np.cos(1.2 * dt) + t1

        # Jacobians exactly as in original code
        An = np.array([[0.5 + 25.*(1.-x1**2)/(1.+x1**2)**2, 0.],
                       [0.1 * A * (0.5 + 25.*(1.-x1**2)/(1.+x1**2)**2), 0.]])
        Bn = np.array([[1.,    0.],
                       [0.1 * A, 1.]])

        return An, Bn
