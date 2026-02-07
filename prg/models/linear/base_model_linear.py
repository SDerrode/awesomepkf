from pathlib import Path
import sys

directory = Path(__file__)
sys.path.append(str(directory.parent.parent.parent))

from scipy.linalg import cho_factor, cho_solve
import numpy as np

# A few utils functions that are used several times
from others.utils import check_consistency

class BaseModelLinear:
    """
    Base class for all linear models.

    Fournit une structure unifiée pour les 2 paramétrisations.
    """
    def __init__(
        self,
        dim_x: int,
        dim_y: int,
        model_type: str
    ):
        # Vérifications
        assert isinstance(dim_x, int) and dim_x > 0, "dim_x doit être un entier positif"
        assert isinstance(dim_y, int) and dim_y > 0, "dim_y doit être un entier positif"
    
        # Dimensions et types
        self.model_type = model_type
        self.dim_x      = dim_x
        self.dim_y      = dim_y
        self.dim_xy     = dim_x + dim_y

    # ------------------------------------------------------------------
    def g(self, z: np.ndarray, noise_z: np.ndarray, dt: float) -> np.ndarray:
        """Compute z_{n+1} = A @ z_{n} + noise."""
        return self.A @ z + noise_z

    def get_params(self):
        if self.model_type == 'linear_AmQ':
            return {'g'    : self.g,
                    'dim_x': self.dim_x,
                    'dim_y': self.dim_y,
                    'A'    : self.A,
                    'mQ'   : self.mQ,
                    'z00'  : self.z00,
                    'Pz00' : self.Pz00}
        elif self.model_type == 'linear_Sigma':
            return {'g': self.g, 
                    'dim_x': self.dim_x,
                    'dim_y': self.dim_y,
                    'sxx'  : self.sxx,
                    'syy'  : self.syy,
                    'a'    : self.a,
                    'b'    : self.b,
                    'c'    : self.c,
                    'd'    : self.d,
                    'e'    : self.e}
        else:
            raise ValueError(f"⚠️ model_type should be 'linear_AmQ' or 'linear_Sigma' - Actual value: {model_type}")


    # Pour les modèle sigma:
    def _compute_A_mq_z00_Pz00(self, Q1, Q2):
        
        # Calcul robuste de A = Q2 @ np.linalg.inv(Q1)
        c, low = cho_factor(Q1)
        self.A = Q2 @ cho_solve((c, low), np.eye(self.dim_xy))
        
        eigvals = np.linalg.eigvals(self.A)
        if np.any(np.abs(eigvals) >= 1.0):
            raise ValueError(f"⚠️ The modulus of one Eigen value of A is >= 1 : {eigvals}")

        self.mQ = Q1 - self.A @ Q2.T
        check_consistency(mQ=self.mQ)
        
        self.z00  = np.zeros((self.dim_xy, 1))
        self.Pz00 = Q1.copy()

        if __debug__:
            check_consistency(mQ=self.mQ, Pz00=self.Pz00)

    # ------------------------------------------------------------------
    def __repr__(self):
        return f"{self.__class__.__name__}(dim_x={self.dim_x}, dim_y={self.dim_y})"