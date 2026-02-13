#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import sys

directory = Path(__file__)
sys.path.append(str(directory.parent.parent.parent))

import numpy as np


class BaseModelNonLinear:
    """
    Base class for all non-linear models.

    Fournit une structure unifiée pour les fonctions fx, hx et g,
    ainsi qu'une gestion cohérente des paramètres et matrices de covariance.
    En mode optimisé (lancé avec `python3 -O`), les vérifications sont désactivées.
    """

    def __init__(self, dim_x: int, dim_y: int, model_type: str = "nonlinear", augmented = False):
        assert isinstance(dim_x, int) and dim_x > 0, "dim_x doit être un entier positif"
        assert isinstance(dim_y, int) and dim_y > 0, "dim_y doit être un entier positif"

        self.model_type  = model_type
        self.augmented   = augmented
        self.dim_x       = dim_x
        self.dim_y       = dim_y
        self.dim_xy      = dim_x + dim_y

        # UKF parameters
        self.alpha       = 0.25
        self.beta        = 2.0
        self.kappa       = 0.0
        self.kappaJulier = 0.0 #3.0 - self.dim_x

        # Initialisation des matrices / vecteurs d'état
        self.mQ   = None
        self.z00  = None
        self.Pz00 = None

    # ------------------------------------------------------------------
    def g(self, z: np.ndarray, noise_z: np.ndarray, dt: float) -> np.ndarray:
        """Compute z_{n+1} = g(z_n) + noise. z et noise_z sont de shape (dim_xy,1)."""
        if __debug__:
            assert z.shape == (self.dim_xy, 1)
            assert noise_z.shape == (self.dim_xy, 1)

        x, y   = np.split(z,       [self.dim_x])
        nx, ny = np.split(noise_z, [self.dim_x])
        return self._g(x, y, nx, ny, dt)

    def jacobiens_g(self, z: np.ndarray, noise_z: np.ndarray, dt: float) -> np.ndarray:
        """Compute Jacobians of g w.r.t z and noise."""
        if __debug__:
            assert z.shape == (self.dim_xy, 1)
            assert noise_z.shape == (self.dim_xy, 1)
        x, y   = np.split(z, [self.dim_x])
        nx, ny = np.split(noise_z, [self.dim_x])
        return self._jacobiens_g(x, y, nx, ny, dt)

    # ------------------------------------------------------------------
    def _g(self, x, y, nx, ny, dt):
        """À implémenter dans la sous-classe"""
        raise NotImplementedError

    def _jacobiens_g(self, x, y, nx, ny, dt):
        """À implémenter dans la sous-classe"""
        raise NotImplementedError

    # ------------------------------------------------------------------
    def get_params(self) -> dict:
        return {'dim_x'      : self.dim_x,
                'dim_y'      : self.dim_y,
                'augmented'  : self.augmented,
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
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(dim_x={self.dim_x}, dim_y={self.dim_y})"