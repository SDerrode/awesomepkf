#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations
from typing import Any
import numpy as np
from scipy.linalg import cho_factor, cho_solve

from pathlib import Path
import sys

directory = Path(__file__)
sys.path.append(str(directory.parent.parent.parent))

from scipy.linalg import cho_factor, cho_solve
import numpy as np

from classes.MatrixDiagnostics import CovarianceMatrix


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

    def __init__(self, dim_x: int, dim_y: int, model_type: str, augmented=False):
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
        self.augmented = augmented

        # UPKF specific parameters - in case where we want to filter linear data with UPKF
        self.alpha = 0.25
        self.beta = 2.0
        self.kappa = 0.0
        self.lambda_ = self.alpha**2 * (self.dim_x + self.kappa) - self.dim_x

    # ------------------------------------------------------------------
    def g(self, z: np.ndarray, noise_z: np.ndarray, dt: float) -> np.ndarray:
        """Compute z_{n+1} = A @ z + B @ noise. z et noise_z sont de shape (dim_xy, 1)."""
        if __debug__:
            assert z.shape == (self.dim_xy, 1)
            assert noise_z.shape == (self.dim_xy, 1)

        return self.A @ z + self.B @ noise_z

    def jacobiens_g(self, z: np.ndarray, noise_z: np.ndarray, dt: float) -> np.ndarray:
        """Cette méthode doit renvoyer An et Bn. Uniquement nécessaire pour que EPKF puisse traiter aussi des données linéaires"""

        return self.A, self.B

    # ------------------------------------------------------------------
    def __repr__(self):
        return f"{self.__class__.__name__}(dim_x={self.dim_x}, dim_y={self.dim_y}, type={self.model_type})"


# ----------------------------------------------------------
# Sous-classe pour la paramétrisation linear_AmQ
# ----------------------------------------------------------
class LinearAmQ(BaseModelLinear):
    """
    Modèle linéaire avec matrice de transition A et covariance Q.
    """

    def __init__(
        self,
        dim_x: int,
        dim_y: int,
        A: np.ndarray,
        B: np.ndarray,
        mQ: np.ndarray,
        mz0: np.ndarray,
        Pz0: np.ndarray,
        augmented=False,
    ):

        super().__init__(dim_x, dim_y, model_type="linear_AmQ", augmented=augmented)

        self.A = A
        self.B = B
        self.mQ = mQ
        self.mz0 = mz0
        self.Pz0 = Pz0

        if __debug__ and not self.augmented:
            for arr in [self.mQ, self.Pz0]:
                report = CovarianceMatrix(arr).check()  # single diagnostic call
                if not report.is_valid:
                    raise ValueError(f"Matrix  is not positive semi-definite.")

    def get_params(self) -> dict[str, Any]:
        return {
            "dim_x": self.dim_x,
            "dim_y": self.dim_y,
            "augmented": self.augmented,
            "g": self.g,
            "jacobiens_g": self.jacobiens_g,  # pour EPKF
            "A": self.A,
            "B": self.B,
            "mQ": self.mQ,
            "mz0": self.mz0,
            "Pz0": self.Pz0,
            "alpha": self.alpha,  # pour UPKF
            "beta": self.beta,  # pour UPKF
            "kappa": self.kappa,  # pour UPKF
            "lambda_": self.lambda_,  # pour UPKF
        }


# ----------------------------------------------------------
# Sous-classe pour la paramétrisation linear_Sigma
# ----------------------------------------------------------
class LinearSigma(BaseModelLinear):
    """
    Modèle linéaire avec variances sxx, syy et coefficients a, b, c, d, e.
    """

    def __init__(
        self,
        dim_x: int,
        dim_y: int,
        sxx: float,
        syy: float,
        a: float,
        b: float,
        c: float,
        d: float,
        e: float,
        augmented=False,
    ):

        super().__init__(dim_x, dim_y, model_type="linear_Sigma", augmented=augmented)

        self.sxx = sxx
        self.syy = syy
        self.a = a
        self.b = b
        self.c = c
        self.d = d
        self.e = e

        self._initSigma()

    def _initSigma(self):

        # Construction des matrices Q1 et Q2
        Q1 = np.block([[self.sxx, self.b.T], [self.b, self.syy]])
        Q2 = np.block([[self.a, self.e], [self.d, self.c]])

        # Calcul robuste de la matrice A via Cholesky
        c_factor, lower = cho_factor(Q1)
        self.A = Q2 @ cho_solve((c_factor, lower), np.eye(self.dim_xy))
        stab = StabilityMatrix(self.A)
        if not stab.is_valid():
            stab.summary()  # True si aucun FAIL
            exit(1)

        # B est l'identité
        self.B = np.eye(self.A.shape[0])

        # Vérification optionnelle
        if __debug__:
            for arr in [self.sxx, self.syy, Q1]:
                report = CovarianceMatrix(arr).check()  # single diagnostic call
                if not report.is_valid:
                    raise ValueError(f"Matrix  is not positive semi-definite.")
            # check_consistency(sxx=self.sxx, syy=self.syy, Q1=Q1)

    def get_params(self) -> dict[str, Any]:
        return {
            "dim_x": self.dim_x,
            "dim_y": self.dim_y,
            "augmented": self.augmented,
            "g": self.g,
            "jacobiens_g": self.jacobiens_g,  # pour EPKF
            "sxx": self.sxx,
            "syy": self.syy,
            "a": self.a,
            "b": self.b,
            "c": self.c,
            "d": self.d,
            "e": self.e,
            "alpha": self.alpha,  # pour UPKF
            "beta": self.beta,  # pour UPKF
            "kappa": self.kappa,  # pour UPKF
            "lambda_": self.lambda_,  # pour UPKF
        }
