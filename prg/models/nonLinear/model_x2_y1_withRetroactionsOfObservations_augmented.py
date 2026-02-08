import numpy as np
from .base_model_nonLinear import BaseModelNonLinear
from others.utils import check_consistency

class ModelX2Y1_withRetroactionsOfObservations_augmented(BaseModelNonLinear):
    """
    Nonlinear model with retro-actions of observations and of states.
    The model includes additive Gaussian process and observation noises.
    ATTENTION : ce modèle a été construit pour être utilisé avec un filtre 
                UKF et comparé avec le modèle 'ModelX2Y1_withRetroactionsOfObservations'
                pour un filtre UPKF, cf rapport.
    """

    MODEL_NAME: str = "x2_y1_withRetroactionsOfObservations_augmented"

    def __init__(self) -> None:
        super().__init__(dim_x=3, dim_y=1, model_type="nonlinear", augmented=True)

        self.mQ  = np.array([[0.1, 0.0, 0.0, 0.0], 
                             [0.0, 0.1, 0.0, 0.0], 
                             [0.0, 0.0, 0.5, 0.0],
                             [0.0, 0.0, 0.0, 0.0]]) 
        self.z00  = np.zeros((self.dim_xy, 1))
        self.Pz00 = np.eye(self.dim_xy)
        if __debug__:
            check_consistency(mQ=self.mQ, Pz00=self.Pz00)

        self.a, self.b, self.c, self.d, self.e, self.f = 1.0, 0.8, 0.05, 0.9, 0.30, 0.6


    # ------------------------------------------------------------------
    def _fx(self, x, t, dt):
        """
        Nonlinear state function with retro-action on observation.
        """
        x1, x2, x3 = x.flatten()
        t1, t2, t3 = t.flatten()

        return np.array([
            self.a * x1 + self.b * x2 + self.c * np.tanh(x3) + t1,
            self.d * x2               + self.e * np.sin(x3)  + t2,
            x1**2       + self.f*x3                          + t3
        ]).reshape(-1, 1)

    # ------------------------------------------------------------------
    def _hx(self, x, u, dt):
        """
        Nonlinear observation function with retro-action on previous observation.
        Le bruit $u$ est nul dans cette formulation.
        """
        x1, x2, x3 = x.flatten()
        return np.array([
            x3
        ]).reshape(-1, 1)

    # ------------------------------------------------------------------
    def _g(self, x, y, t, u, dt):
        """
        Combined state and observation using Wojciech’s formulation.
        """
        if __debug__:
            assert x.shape == (3, 1), f"x must be (3,1), got {x.shape}"
            assert y.shape == (1, 1), f"y must be (1,1), got {y.shape}"
            assert t.shape == (3, 1), f"t must be (3,1), got {t.shape}"
            assert u.shape == (1, 1), f"u must be (1,1), got {u.shape}"
            assert isinstance(dt, (float, int)), "dt must be a float"

        fx_val = self._fx(x, t, dt)
        hx_val = self._hx(fx_val, u, dt)
        return np.vstack((fx_val, hx_val))

    # ------------------------------------------------------------------
    def _jacobiens_g(self, x, y, t, u, dt):
        """
        Jacobians of combined state and observation function.
        """
        if __debug__:
            assert x.shape == (3, 1), f"x must be (3,1), got {x.shape}"
            assert y.shape == (1, 1), f"y must be (1,1), got {y.shape}"
            assert t.shape == (3, 1), f"t must be (3,1), got {t.shape}"
            assert u.shape == (1, 1), f"u must be (1,1), got {u.shape}"
            assert isinstance(dt, (float, int)), "dt must be a float"

        x1, x2, x3 = x.flatten()
        # t1, t2 = t.flatten()
        # y1 = y.flatten()[0]
        # u1 = u.flatten()[0]

        An = np.array([[self.a,   self.b, self.c * (1.-np.tanh(x3)**2), 0.],
                       [0,        self.d, self. e * np.cos(x3),         0.],
                       [2 * x1,       0., self.f,                       0.],
                       [2 * x1,       0., self.f,                       0.]])
        Bn = np.array([[1., 0., 0., 0.],
                       [0., 1., 0., 0.],
                       [0., 0., 1., 0.],
                       [0., 0., 1., 0.]])

        return An, Bn
