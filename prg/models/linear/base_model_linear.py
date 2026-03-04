#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations
import os
import matplotlib.pyplot as plt
from typing import Any
import numpy as np
from scipy.linalg import cho_factor, cho_solve, LinAlgError

from prg.classes.MatrixDiagnostics import CovarianceMatrix, StabilityMatrix
from prg.exceptions import NumericalError
from prg.utils.plot_settings import DPI, FACECOLOR, BIG_SIZE

__all__ = ["BaseModelLinear", "LinearAmQ", "LinearSigma"]


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

        try:
            return self.A @ z + self.B @ noise_z
        except (ValueError, np.exceptions.AxisError) as e:
            raise NumericalError(
                f"[{self.__class__.__name__}] g: matrix multiplication error: {e}"
            ) from e

    def jacobiens_g(self, z: np.ndarray, noise_z: np.ndarray, dt: float) -> np.ndarray:
        """Cette méthode doit renvoyer An et Bn. Uniquement nécessaire pour que EPKF puisse traiter aussi des données linéaires"""
        # print(self.A, self.B)
        # input("ATTENTE")
        return self.A, self.B

    # ------------------------------------------------------------------
    def __repr__(self):
        return f"{self.__class__.__name__}(dim_x={self.dim_x}, dim_y={self.dim_y}, type={self.model_type})"

    # ------------------------------------------------------------------
    def plot_g(self, n_points: int = 200) -> None:

        if self.dim_x != 1 or self.dim_y != 1:
            return

        import os
        import numpy as np
        import matplotlib.pyplot as plt

        z1 = np.linspace(-1, 1, n_points)
        z2 = np.linspace(-1, 1, n_points)
        Z1, Z2 = np.meshgrid(z1, z2)

        Z_stack = np.stack([Z1.ravel(), Z2.ravel()], axis=1)

        noise = np.zeros((2, 1))
        dt = 1.0

        G = np.zeros_like(Z_stack)

        with np.errstate(all="raise"):
            for k in range(Z_stack.shape[0]):
                z = Z_stack[k].reshape(2, 1)
                g_val = self.g(z, noise, dt)

                if not np.isfinite(g_val).all():
                    raise FloatingPointError("Non-finite value encountered in g")

                G[k, :] = g_val.ravel()

        G1 = G[:, 0].reshape(n_points, n_points)
        G2 = G[:, 1].reshape(n_points, n_points)

        os.makedirs("data/plot", exist_ok=True)

        fig = plt.figure(figsize=(9, 4), facecolor=FACECOLOR)

        ax1 = fig.add_subplot(1, 2, 1, projection="3d")
        ax1.plot_surface(Z1, Z2, G1)
        ax1.set_title(r"$g_x(x, y)$")
        ax1.set_xlabel(r"$x$")
        ax1.set_ylabel(r"$y$")
        ax1.view_init(elev=30, azim=45)

        ax2 = fig.add_subplot(1, 2, 2, projection="3d")
        ax2.plot_surface(Z1, Z2, G2)
        ax2.set_title(r"$g_y(x, y)$")
        ax2.set_xlabel(r"$x$")
        ax2.set_ylabel(r"$y$")
        ax2.view_init(elev=30, azim=45)

        # Normalisation échelle pour éviter distorsion visuelle
        for ax in (ax1, ax2):
            ax.set_box_aspect((1, 1, 0.8))

        model_name = getattr(self, "MODEL_NAME", self.__class__.__name__)
        filename = f"data/plot/function_g_{model_name}.png"
        plt.savefig(filename, dpi=DPI, bbox_inches="tight", facecolor=FACECOLOR)
        plt.close(fig)

    def plot_g_dynamic(self, n_points: int = 150, quiver_stride: int = 10) -> None:
        """
        Plot g(z, noise=0, dt=1) over [-1,1]^2.
        Only available for dim_x=1 and dim_y=1.
        Saves figure into data/plot/.
        """

        if self.dim_x != 1 or self.dim_y != 1:
            return

        import os
        import numpy as np
        import matplotlib.pyplot as plt

        z1 = np.linspace(-1, 1, n_points)
        z2 = np.linspace(-1, 1, n_points)
        Z1, Z2 = np.meshgrid(z1, z2)

        Z_stack = np.stack([Z1.ravel(), Z2.ravel()], axis=1)

        noise = np.zeros((2, 1))
        dt = 1.0

        G = np.zeros_like(Z_stack)

        with np.errstate(all="raise"):
            for k in range(Z_stack.shape[0]):
                z = Z_stack[k].reshape(2, 1)
                g_val = self.g(z, noise, dt)

                if not np.isfinite(g_val).all():
                    raise FloatingPointError("Non-finite value in g")

                G[k, :] = g_val.ravel()

        G1 = G[:, 0].reshape(n_points, n_points)
        G2 = G[:, 1].reshape(n_points, n_points)

        # Norme
        NormG = np.sqrt(G1**2 + G2**2)

        # Champ dynamique : g(z) - z
        Dz1 = G1 - Z1
        Dz2 = G2 - Z2

        os.makedirs("data/plot", exist_ok=True)

        fig = plt.figure(figsize=(14, 10), facecolor=FACECOLOR)

        # --- Surface g_x ---
        ax1 = fig.add_subplot(2, 2, 1, projection="3d")
        surf1 = ax1.plot_surface(Z1, Z2, G1)
        ax1.set_title(r"$g_x(x,y)$")
        ax1.set_xlabel(r"$x$")
        ax1.set_ylabel(r"$y$")
        ax1.view_init(30, 45)

        # --- Surface g_y ---
        ax2 = fig.add_subplot(2, 2, 2, projection="3d")
        surf2 = ax2.plot_surface(Z1, Z2, G2)
        ax2.set_title(r"$g_y(x,y)$")
        ax2.set_xlabel(r"$x$")
        ax2.set_ylabel(r"$y$")
        ax2.view_init(30, 45)

        # --- Norme ---
        ax3 = fig.add_subplot(2, 2, 3)
        im = ax3.contourf(Z1, Z2, NormG)
        ax3.set_title(r"$\|g(x,y)\|$")
        ax3.set_xlabel(r"$x$")
        ax3.set_ylabel(r"$y$")
        plt.colorbar(im, ax=ax3)

        # --- Champ dynamique ---
        ax4 = fig.add_subplot(2, 2, 4)
        ax4.quiver(
            Z1[::quiver_stride, ::quiver_stride],
            Z2[::quiver_stride, ::quiver_stride],
            Dz1[::quiver_stride, ::quiver_stride],
            Dz2[::quiver_stride, ::quiver_stride],
        )
        ax4.set_title(r"$g(z) - z$")
        ax4.set_xlabel(r"$x$")
        ax4.set_ylabel(r"$y$")
        ax4.set_aspect("equal")

        model_name = getattr(self, "MODEL_NAME", self.__class__.__name__)
        filename = f"data/plot/function_g_dynamic_{model_name}.png"
        plt.savefig(filename, dpi=DPI, bbox_inches="tight", facecolor=FACECOLOR)
        plt.close(fig)


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

        try:
            self.A = A
            self.B = B
            self.mQ = mQ
            self.mz0 = mz0
            self.Pz0 = Pz0
        except Exception as e:
            raise NumericalError(
                f"[{self.__class__.__name__}] LinearAmQ: parameter assignment error: {e}"
            ) from e

        if __debug__ and not self.augmented:
            for arr in [self.mQ, self.Pz0]:
                report = CovarianceMatrix(arr).check()
                if not report.is_valid:
                    raise ValueError(f"Matrix is not positive semi-definite.")

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

        try:
            # Construction des matrices Q1 et Q2
            Q1 = np.block([[self.sxx, self.b.T], [self.b, self.syy]])
            Q2 = np.block([[self.a, self.e], [self.d, self.c]])
        except (ValueError, np.exceptions.AxisError) as e:
            raise NumericalError(
                f"[{self.__class__.__name__}] _initSigma: block matrix construction error: {e}"
            ) from e

        try:
            # Calcul robuste de la matrice A via Cholesky
            c_factor, lower = cho_factor(Q1)
            self.A = Q2 @ cho_solve((c_factor, lower), np.eye(self.dim_xy))
        except LinAlgError as e:
            raise NumericalError(
                f"[{self.__class__.__name__}] _initSigma: Cholesky decomposition failed"
                f" (Q1 is not positive definite): {e}"
            ) from e
        except (ValueError, np.exceptions.AxisError) as e:
            raise NumericalError(
                f"[{self.__class__.__name__}] _initSigma: matrix solve error: {e}"
            ) from e

        stab = StabilityMatrix(self.A)
        if not stab.is_valid():
            stab.summary()
            exit(1)

        # B est l'identité
        self.B = np.eye(self.A.shape[0])

        if __debug__:
            for arr in [self.sxx, self.syy, Q1]:
                report = CovarianceMatrix(arr).check()
                if not report.is_valid:
                    raise ValueError(f"Matrix is not positive semi-definite.")

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
