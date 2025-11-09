import numpy as np

class BaseModel:
    """
    Base class for all non linear models.
    Provides a unified structure for fx, hx, and g functions,
    as well as consistent management of parameters and covariance matrices.
    """
    
    def __init__(
        self,
        dim_x: int,
        dim_y: int,
        alpha: float = 1e-3,
        beta: float = 2.0,
        kappa: float = 0.0,
    ):
        # Dimensions of state (x) and observation (y)
        self.dim_x = dim_x
        self.dim_y = dim_y
        self.dim_xy = dim_x + dim_y

        # UKF parameters
        self.alpha = alpha
        self.beta = beta
        self.kappa = kappa

        # Default covariance and initialization matrices
        self.mQ = np.eye(self.dim_xy)
        self.z00 = np.zeros((self.dim_xy, 1))
        self.Pz00 = np.eye(self.dim_xy)
    
    # def fx(self, x, noise, dt):
    #     raise NotImplementedError
    
    # def hx(self, x, noise, dt):
    #     raise NotImplementedError
    
    def g(self, z, noise, dt):
        
        if self.fx is None or self.hx is None:
            raise NotImplementedError("Subclasses must implement both fx() and hx() methods.")

        if z.shape[0] != self.dim_xy:
            raise ValueError(f"z must have shape ({self.dim_xy}, 1), but got {z.shape}")

        # Split state and noise vectors
        x = z[:self.dim_x]
        noise_x = noise[:self.dim_x]
        noise_y = noise[self.dim_x:]

        fx_val = self.fx(x, noise_x, dt)
        hx_val = self.hx(fx_val, noise_y, dt)

        g_val = np.vstack((fx_val, hx_val))
        return g_val
    
    # -------------------------------------------------------------------------
    # Utility methods
    # -------------------------------------------------------------------------

    def check_consistency(self):
        """Check that covariance matrices are symmetric and positive semi-definite."""
        for name, M in {"mQ": self.mQ, "Pz00": self.Pz00}.items():
            if not np.allclose(M, M.T, atol=1e-12):
                logger.warning(f"⚠️ Matrix {name} is not symmetric.")
            eigvals = np.linalg.eigvals(M)
            if np.any(eigvals < -1e-12):
                logger.warning(
                    f"⚠️ Matrix {name} is not positive semi-definite (min eigenvalue = {eigvals.min():.3e})"
                )
                
    def get_params(self):
        return (self.dim_x, self.dim_y, self.g, self.mQ, self.z00, self.Pz00,
                self.alpha, self.beta, self.kappa)

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(dim_x={self.dim_x}, dim_y={self.dim_y}, "
            f"alpha={self.alpha}, beta={self.beta}, kappa={self.kappa})"
        )