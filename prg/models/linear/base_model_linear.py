import path, sys
directory = path.Path(__file__)
sys.path.append(directory.parent.parent.parent)
# print(directory.parent.parent.parent)
# exit(1)

import numpy as np

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
    
    # def info(self):
    #     print(f"Type de modèle : {self.model_type}")
    #     print(f"Dimensions : dim_x={self.dim_x}, dim_y={self.dim_y}")
    #     if self.model_type == 'type1':
    #         print("Paramètres : A, mQ, z00, Pz00")
    #     else:
    #         print("Paramètres : sxx, syy, a, b, c, d, e")

    def get_params(self):
        if self.model_type == 'linear_AmQ':
            return {'g': self.g, 'dim_x':self.dim_x, 'dim_y':self.dim_y, 'A':self.A, \
                        'mQ':self.mQ, 'z00':self.z00, 'Pz00':self.Pz00}
        elif self.model_type == 'linear_Sigma':
            return {'g': self.g, 'dim_x':self.dim_x, 'dim_y':self.dim_y, 'sxx':self.sxx, \
                        'syy':self.syy, 'a':self.a, 'b':self.b, 'c':self.c, 'd':self.d, 'e':self.e}
        else:
            raise ValueError(f"⚠️ model_type should be 'linear_AmQ' or 'linear_Sigma' - Actual value: {model_type}")

    # ------------------------------------------------------------------
    def __repr__(self):
        return f"{self.__class__.__name__}(dim_x={self.dim_x}, dim_y={self.dim_y})"