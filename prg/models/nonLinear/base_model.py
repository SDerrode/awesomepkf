import inspect
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
    
    def g(self, z, noise_z, dt):

        if z.shape[0] != self.dim_xy:
            raise ValueError(f"z must have shape ({self.dim_xy}, 1), but got {z.shape}")
        
        if noise_z.shape[0] != self.dim_xy:
            raise ValueError(f"noise_z must have shape ({self.dim_xy}, 1), but got {noise_z.shape}")

        # Split state and noise_z vectors
        x, y   = z[:self.dim_x], z[self.dim_x:]
        nx, ny = noise_z[:self.dim_x], noise_z[self.dim_x:]

        # Fonction de calcul de la fonction znp1 =  g(zn) + bruit
        return self._g(x, y, nx, ny, dt)

        # # Travail sur fx
        # sig = inspect.signature(self.fx)
        # nb_parameters_fx = len(sig.parameters)
        # if nb_parameters_fx == 4:                  # Cas avec retroactions
        #     fx_val = self.fx(x, nx, y, dt)
        # else:                                      # Cas classique
        #     fx_val = self.fx(x, nx, dt)
            
        # # Travail sur hx
        # sig = inspect.signature(self.hx)
        # nb_parameters_hx = len(sig.parameters)
        # if nb_parameters_hx == 4:                  # Cas avec retroactions
        #     hx_val = self.hx(x, ny, y, dt)
        # else:                                      # Cas classique
        #     hx_val = self.hx(fx_val, ny, dt)

        # # print(f'fx_val={fx_val}')
        # # print(f'hx_val={hx_val}')
        # g_val = np.vstack((fx_val, hx_val))
        # # print(f'g_val={g_val}')
        # # input('attente')
        # return g_val
    
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