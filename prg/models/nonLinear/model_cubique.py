import numpy as np
from .base_model_nonLinear import BaseModelNonLinear
from others.utils import check_consistency


class ModelCubique(BaseModelNonLinear):
    """
    Cubic nonlinear model with 1D state and 1D observation.

    System dynamics:
        f(x) = 0.9*x - 0.2*x^3

    Measurement equation:
        h(x) = x

    Includes additive Gaussian process and observation noise.
    """
    
    MODEL_NAME: str = "x1_y1_cubique"

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
        State transition function with cubic nonlinearity.

        Args:
            x : np.ndarray, shape (1,1) - current state
            t : np.ndarray, shape (1,1) - process noise
            dt: float - timestep

        Returns:
            np.ndarray, shape (1,1) - next state
        """
        x1 = x.flatten()[0]
        t1 = t.flatten()[0]
        return np.array([[0.9*x1 - 0.2*x1**3 + t1]])

    # ------------------------------------------------------------------
    def _hx(self, x: np.ndarray, u: np.ndarray, dt: float) -> np.ndarray:
        """
        Linear observation function with additive noise.

        Args:
            x : np.ndarray, shape (1,1) - predicted state
            u : np.ndarray, shape (1,1) - observation noise
            dt: float - timestep

        Returns:
            np.ndarray, shape (1,1) - measurement
        """
        x1 = x.flatten()[0]
        u1 = u.flatten()[0]
        return np.array([[x1 + u1]])

    # ------------------------------------------------------------------
    def _g(self, x: np.ndarray, y: np.ndarray, t: np.ndarray, u: np.ndarray, dt: float) -> np.ndarray:
        """
        Combine state and observation using Wojciech's formulation.

        Args:
            x : np.ndarray, shape (1,1) - current state
            y : np.ndarray, shape (1,1) - current observation
            t : np.ndarray, shape (1,1) - process noise
            u : np.ndarray, shape (1,1) - observation noise
            dt: float - timestep

        Returns:
            np.ndarray, shape (2,1) - combined state + observation
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

        An = np.array([[0.9 - 0.6*x1**2, 0.],
                       [0.9 - 0.6*x1**2, 0.]])
        Bn = np.array([[1., 0.],
                       [1., 1.]])

        return An, Bn