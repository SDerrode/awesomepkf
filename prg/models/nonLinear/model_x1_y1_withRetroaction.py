import numpy as np
from .base_model_nonLinear import BaseModelNonLinear
from others.utils import check_consistency

class ModelX1Y1_withRetroactions(BaseModelNonLinear):
    """
    Nonlinear model with retro-actions of observations and states.
    The model includes additive Gaussian process and observation noises.
    """

    MODEL_NAME: str = "x1_y1_withRetroactions"

    def __init__(self) -> None:
        super().__init__(dim_x=1, dim_y=1, model_type="nonlinear")

        self.Pz00 = np.eye(self.dim_xy)

        # (A) Strongly damped / stable:
        # (a,b,c,d) = (0.4,\;0.1,\;0.3,\;0.1)
        # Expected behaviour: rapid convergence to a fixed point (strong damping). Use as baseline for filter stability tests.
        # Numeric tips: \(x_0=y_0=(0.5,0.5)\), \(t_1=u_1=0\), \(\sigma_x=\sigma_y=0.01\), \(N\approx 20\).
        # self.mQ   = np.diag([0.01, 0.01]) #[1E-1, 1E-1])
        # self.z00  = np.zeros((self.dim_xy, 1)) + 0.5
        # self.a, self.b, self.c, self.d = 0.4, 0.1, 0.3, 0.1
        
        # (B) Weakly damped / lightly oscillatory:
        # Quand on regarde N=10000 ech, on voit comme des sauts
        # (a,b,c,d) = (0.9,\;0.5,\;0.95,\;0.8)
        # Expected behaviour: slow decay with oscillatory transients; sustained small-amplitude oscillations possible due to nonlinear coupling.
        # Numeric tips: \(x_0=0.1,\;y_0=1.0\), \(\sigma_x=\sigma_y=0.02\), \(N\approx 100\).
        # self.mQ   = np.diag([0.05, 0.05]) #[1E-1, 1E-1])
        # self.z00  = np.array([[0.1],[1.]])
        # self.a, self.b, self.c, self.d = 0.9, 0.5, 0.95, 0.8
        
        # (C) Sustained oscillations / limit-cycle-like: INTERESSANT
        # (a,b,c,d) = (0.99,\;1.2,\;0.9,\;1.5)
        # Expected behaviour: persistent oscillations of moderate amplitude; nonlinear terms drive and sustain the cycles.
        # Numeric tips: choose \(x_0,y_0\) small but nonzero, \(\sigma\) very small (e.g.\ 0.005) to reveal deterministic oscillation, \(N\ge 300\).
        self.mQ  = np.array([[0.1, 0.], [0., 0.5]]) #np.diag([0.1, 0.5])
        self.z00  = np.array([[0.], [0.]])
        self.a, self.b, self.c, self.d = 0.99, 1.2, 0.9, 1.5
        
        # (D) Complex / quasi-periodic dynamics:
        # (a,b,c,d) = (1.05,\;1.5,\;0.95,\;2.0)
        # Expected behaviour: rich, possibly quasi-periodic or mixed-mode oscillations; sensitive dependence on initial condition; intermittent bursts.
        # Numeric tips: try several initial conditions, \(\sigma_x=\sigma_y=0.01\), run \(N\ge 2000\) and inspect time series & phase portrait.
        # self.mQ   = np.diag([0.5, 0.5]) 
        # self.z00  = np.array([[-5],[0.5]])
        # self.a, self.b, self.c, self.d = 1.05, 1.5, 0.95, 2.0
        
        # (E) Chaotic-like / high nonlinearity (exploratory):
        # (a,b,c,d) = (1.2,\;2.0,\;0.8,\;2.5)
        # Expected behaviour: very irregular trajectories; can look chaotic or may diverge — use with caution.
        # Numeric tips: set \(\sigma_x=\sigma_y\) small (e.g.\ 0.001) to see deterministic structure; monitor for divergence; compute Lyapunov exponent if needed.
        # self.mQ   = np.diag([0.1, 0.1]) 
        # self.z00  = np.array([[-5],[0.5]])
        # self.a, self.b, self.c, self.d = 1.2, 2.0, 0.8, 2.5
        
        if __debug__:
            check_consistency(mQ=self.mQ, Pz00=self.Pz00)

    # ------------------------------------------------------------------
    def _gx(self, x: np.ndarray, y: np.ndarray, t: np.ndarray, u: np.ndarray, dt: float) -> np.ndarray:
        """
        Nonlinear state function with retro-action of observations on state.
        """
        x1 = x.flatten()[0]
        y1 = y.flatten()[0]
        t1 = t.flatten()[0]
        return np.array([[self.a * x1 + self.b * np.tanh(y1) + t1]])

    # ------------------------------------------------------------------
    def _gy(self, x: np.ndarray, y: np.ndarray, t: np.ndarray, u: np.ndarray, dt: float) -> np.ndarray:
        """
        Nonlinear state function with retro-action of states on observation.
        """
        x1 = x.flatten()[0]
        y1 = y.flatten()[0]
        u1 = u.flatten()[0]
        return np.array([[self.c * y1 + self.d * np.sin(x1) + u1]])

    # ------------------------------------------------------------------
    def _g(self, x: np.ndarray, y: np.ndarray, t: np.ndarray, u: np.ndarray, dt: float) -> np.ndarray:
        """
        Combine state and observation using Wojciech’s formulation.
        """
        if __debug__:
            assert x.shape == (1, 1)
            assert y.shape == (1, 1)
            assert t.shape == (1, 1)
            assert u.shape == (1, 1)
            assert isinstance(dt, (float, int))

        gx_val = self._gx(x, y, t, u, dt)
        gy_val = self._gy(x, y, t, u, dt)
        return np.vstack((gx_val, gy_val))

    # ------------------------------------------------------------------
    def _jacobiens_g(self, x: np.ndarray, y: np.ndarray, t: np.ndarray, u: np.ndarray, dt: float):
        """
        Compute Jacobians of g w.r.t state and noise.
        """
        if __debug__:
            assert x.shape == (1, 1)
            assert y.shape == (1, 1)
            assert t.shape == (1, 1)
            assert u.shape == (1, 1)
            assert isinstance(dt, (float, int))

        x1 = x.flatten()[0]
        y1 = y.flatten()[0]

        An = np.array([
            [self.a,              self.b / np.cosh(y1)**2],
            [self.d * np.cos(x1), self.c                 ]
        ])
        Bn = np.eye(self.dim_xy)

        return An, Bn
