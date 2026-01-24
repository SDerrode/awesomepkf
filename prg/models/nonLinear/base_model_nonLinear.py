import path, sys
directory = path.Path(__file__)
sys.path.append(directory.parent.parent.parent)
# print(directory.parent.parent.parent)
# exit(1)

import numpy as np


class BaseModelNonLinear:
    """
    Base class for all non-linear models.

    Fournit une structure unifiée pour les fonctions fx, hx et g,
    ainsi qu'une gestion cohérente des paramètres et matrices de covariance.
    En mode optimisé (lancé avec `python -O`), les vérifications sont désactivées.
    """

    def __init__(
        self,
        dim_x: int,
        dim_y: int,
        model_type: str = "nonlinear",
    ):
        # Vérifications
        assert isinstance(dim_x, int) and dim_x > 0, "dim_x doit être un entier positif"
        assert isinstance(dim_y, int) and dim_y > 0, "dim_y doit être un entier positif"

        # Dimensions et types
        self.model_type = model_type
        self.dim_x      = dim_x
        self.dim_y      = dim_y
        self.dim_xy     = dim_x + dim_y

        # UKF parameters
        self.alpha       = 0.25 # si on veut W_O^{(c)} faiblement négatif, on prend alpha entre 0.25 et 0.3
        self.beta        = 2.
        self.kappa       = 0.
        self.kappaJulier = 3. - self.dim_x

    # ------------------------------------------------------------------
    def g(self, z: np.ndarray, noise_z: np.ndarray, dt: float) -> np.ndarray:
        """Compute z_{n+1} = g(z_n) + noise."""

        if __debug__:  # ⚙️ ces vérifs seront ignorées avec python -O
            assert isinstance(z, np.ndarray), "z doit être un numpy.ndarray"
            assert isinstance(noise_z, np.ndarray), "noise_z doit être un numpy.ndarray"
            assert z.ndim == 2 and z.shape[1] == 1, f"z doit avoir une forme (N,1), reçu {z.shape}"
            assert noise_z.ndim == 2 and noise_z.shape[1] == 1, f"noise_z doit avoir une forme (N,1), reçu {noise_z.shape}"
            assert z.shape[0] == self.dim_xy, f"z doit avoir une taille {self.dim_xy}, reçu {z.shape[0]}"
            assert noise_z.shape[0] == self.dim_xy, f"noise_z doit avoir une taille {self.dim_xy}, reçu {noise_z.shape[0]}"

        # Split state and noise vectors
        x, y   = np.split(z,       [self.dim_x])
        nx, ny = np.split(noise_z, [self.dim_x])

        # Appel de la fonction spécifique du modèle
        return self._g(x, y, nx, ny, dt)

    # ------------------------------------------------------------------

    def jacobiens_g(self, z: np.ndarray, noise_z: np.ndarray, dt: float) -> np.ndarray:
        
        if __debug__:  # ⚙️ ces vérifs seront ignorées avec python -O
            assert isinstance(z, np.ndarray), "z doit être un numpy.ndarray"
            assert isinstance(noise_z, np.ndarray), "noise_z doit être un numpy.ndarray"
            assert z.ndim == 2 and z.shape[1] == 1, f"z doit avoir une forme (N,1), reçu {z.shape}"
            assert noise_z.ndim == 2 and noise_z.shape[1] == 1, f"noise_z doit avoir une forme (N,1), reçu {noise_z.shape}"
            assert z.shape[0] == self.dim_xy, f"z doit avoir une taille {self.dim_xy}, reçu {z.shape[0]}"
            assert noise_z.shape[0] == self.dim_xy, f"noise_z doit avoir une taille {self.dim_xy}, reçu {noise_z.shape[0]}"

        # Split state and noise vectors
        x, y   = np.split(z,       [self.dim_x])
        nx, ny = np.split(noise_z, [self.dim_x])

        # Appel de la fonction spécifique du modèle
        return self._jacobiens_g(x, y, nx, ny, dt)

    # ------------------------------------------------------------------
    def get_params(self):
        """Retourne les paramètres principaux du modèle."""
        return {'dim_x'      : self.dim_x,
                'dim_y'      : self.dim_y,
                'g'          : self.g,
                'jacobiens_g': self.jacobiens_g,  # pour EPKF
                'alpha'      : self.alpha,        # pour UPKF merwe
                'beta'       : self.beta,         # pour UPKF merwe
                'kappa'      : self.kappa,        # pour UPKF merwe
                'kappaJulier': self.kappaJulier,  # pour UPKF julier
                'mQ'         : self.mQ,
                'z00'        : self.z00,
                'Pz00'       : self.Pz00,
               }

    # ------------------------------------------------------------------
    def __repr__(self):
        return ( f"{self.__class__.__name__}(dim_x={self.dim_x}, dim_y={self.dim_y})")
