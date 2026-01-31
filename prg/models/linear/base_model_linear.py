from __future__ import annotations
from typing import Any
import numpy as np
from scipy.linalg import cho_factor, cho_solve

from pathlib import Path
import sys
directory = Path(__file__)
sys.path.append(str(directory.parent.parent.parent))

from others.utils import check_consistency


class BaseModelLinear:
    """
    Base class for linear models.

    Cette classe fournit une structure commune pour des modèles linéaires à deux paramétrisations possibles :
      1. 'linear_AmQ' : dynamique x_{n+1} = A x_n + bruit, avec covariance Q
      2. 'linear_Sigma' : paramétrisation via variances sxx, syy et coefficients a, b, c, d, e

    Attributes
    ----------
    dim_x : int
        Dimension de l'état x.
    dim_y : int
        Dimension de l'observation y.
    dim_xy : int
        Somme des dimensions dim_x + dim_y.
    model_type : str
        Type de modèle : 'linear_AmQ' ou 'linear_Sigma'.
    """

    def __init__(self, dim_x: int, dim_y: int, model_type: str):
        if not isinstance(dim_x, int) or dim_x <= 0:
            raise ValueError("dim_x doit être un entier positif")
        if not isinstance(dim_y, int) or dim_y <= 0:
            raise ValueError("dim_y doit être un entier positif")
        if model_type not in ("linear_AmQ", "linear_Sigma"):
            raise ValueError("model_type doit être 'linear_AmQ' ou 'linear_Sigma'")

        self.dim_x = dim_x
        self.dim_y = dim_y
        self.dim_xy = dim_x + dim_y
        self.model_type = model_type

    def g(self, z: np.ndarray, noise_z: np.ndarray, dt: float = 1.0) -> np.ndarray:
        """
        Fonction de transition linéaire : z_{n+1} = A z_n + noise_z

        Parameters
        ----------
        z : np.ndarray
            État courant.
        noise_z : np.ndarray
            Bruit appliqué à la transition.
        dt : float, optional
            Pas de temps (non utilisé ici, mais utile pour les extensions), par défaut 1.0.

        Returns
        -------
        np.ndarray
            État prédit.
        """
        return self.A @ z + noise_z


    def get_params(self) -> dict[str, Any]:
        """
        Retourne un dictionnaire des paramètres du modèle.

        Returns
        -------
        dict
            Paramètres du modèle selon la paramétrisation.
        """
        raise NotImplementedError("La méthode get_params() doit être définie dans les sous-classes")

    def __repr__(self):
        return f"{self.__class__.__name__}(dim_x={self.dim_x}, dim_y={self.dim_y}, type={self.model_type})"


# ----------------------------------------------------------
# Sous-classe pour la paramétrisation linear_AmQ
# ----------------------------------------------------------
class LinearAmQ(BaseModelLinear):
    """
    Modèle linéaire avec matrice de transition A et covariance Q.
    """
    def __init__(self, dim_x: int, dim_y: int, A: np.ndarray, mQ: np.ndarray, z00: np.ndarray, Pz00: np.ndarray):
        super().__init__(dim_x, dim_y, model_type="linear_AmQ")
        self.A = A
        self.mQ = mQ
        self.z00 = z00
        self.Pz00 = Pz00
        
        if __debug__:
            check_consistency(mQ=self.mQ, Pz00=self.Pz00)

    def get_params(self) -> dict[str, Any]:
        return {
            'g': self.g,
            'dim_x': self.dim_x,
            'dim_y': self.dim_y,
            'A': self.A,
            'mQ': self.mQ,
            'z00': self.z00,
            'Pz00': self.Pz00
        }


# ----------------------------------------------------------
# Sous-classe pour la paramétrisation linear_Sigma
# ----------------------------------------------------------
class LinearSigma(BaseModelLinear):
    """
    Modèle linéaire avec variances sxx, syy et coefficients a, b, c, d, e.
    """
    def __init__(self, dim_x: int, dim_y: int, sxx: float, syy: float, a: float, b: float, c: float, d: float, e: float):
        super().__init__(dim_x, dim_y, model_type="linear_Sigma")
        self.sxx = sxx
        self.syy = syy
        self.a = a
        self.b = b
        self.c = c
        self.d = d
        self.e = e

    def _initSigma(self):
        
        # Construction des matrices Q1 et Q2
        Q1 = np.block([[self.sxx, self.b.T],
                       [self.b, self.syy]])
        Q2 = np.block([[self.a, self.e],
                       [self.d, self.c]])

        # Calcul robuste de la matrice A via Cholesky
        c_factor, lower = cho_factor(Q1)
        self.A = Q2 @ cho_solve((c_factor, lower), np.eye(self.dim_xy))

        # Vérification de la stabilité (valeurs propres < 1)
        eigvals = np.linalg.eigvals(self.A)
        if np.any(np.abs(eigvals) >= 1.0):
            raise ValueError(f"⚠️ Une valeur propre de A a un module >= 1 : {eigvals}")

        # Vérification optionnelle
        if __debug__:
            check_consistency(sxx=self.sxx, syy=self.syy)

    def get_params(self) -> dict[str, Any]:
        return {
            'g': self.g,
            'dim_x': self.dim_x,
            'dim_y': self.dim_y,
            'sxx': self.sxx,
            'syy': self.syy,
            'a': self.a,
            'b': self.b,
            'c': self.c,
            'd': self.d,
            'e': self.e
        }
