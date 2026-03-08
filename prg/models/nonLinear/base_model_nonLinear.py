#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import os
import matplotlib.pyplot as plt

from prg.classes.SeedGenerator import SeedGenerator
from prg.exceptions import NumericalError
from prg.utils.plot_settings import DPI, FACECOLOR, BIG_SIZE

__all__ = ["BaseModelNonLinear", "NumericalError"]  # ré-exporté pour commodité


class BaseModelNonLinear:
    """
    Base class for all non-linear models.

    Fournit une structure unifiée pour les fonctions fx, hx et g,
    ainsi qu'une gestion cohérente des paramètres et matrices de covariance.
    En mode optimisé (lancé avec `python3 -O`), les vérifications sont désactivées.
    """

    def __init__(
        self, dim_x: int, dim_y: int, model_type: str = "nonlinear", augmented=False
    ):
        # print("BaseModelNonLinear - __init__")

        assert isinstance(dim_x, int) and dim_x > 0, "dim_x doit être un entier positif"
        assert isinstance(dim_y, int) and dim_y > 0, "dim_y doit être un entier positif"

        self.model_type = model_type
        self.augmented = augmented
        self.dim_x = dim_x
        self.dim_y = dim_y
        self.dim_xy = dim_x + dim_y

        # Specific UPKF parameters
        self.alpha = 0.25
        self.beta = 2.0
        self.kappa = 0.0
        self.lambda_ = self.alpha**2 * (self.dim_x + self.kappa) - self.dim_x

        # Initialisation des matrices / vecteurs d'état
        self.mQ = None
        self.mz0 = None
        self.Pz0 = None

        self._randMatrices = SeedGenerator(9)

    # ------------------------------------------------------------------
    def g(self, z: np.ndarray, noise_z: np.ndarray, dt: float) -> np.ndarray:

        # print("BaseModelNonLinear - g 1")

        if __debug__:
            if z.ndim == 2:
                assert all(a.shape == (self.dim_xy, 1) for a in (z, noise_z))
            else:
                assert all(
                    a.ndim == 3 and a.shape[1:] == (self.dim_xy, 1)
                    for a in (z, noise_z)
                )
                assert z.shape[0] == noise_z.shape[0]

        try:

            axis = 1 if z.ndim == 3 else 0
            x, y = np.split(z, [self.dim_x], axis=axis)
            nx, ny = np.split(noise_z, [self.dim_x], axis=axis)
            # print("BaseModelNonLinear - g 2")
            # print(x, y, nx, ny, dt)
            # print(self._g(x, y, nx, ny, dt))
            return self._g(x, y, nx, ny, dt)
        except NumericalError:
            raise
        except ValueError as e:
            raise NumericalError(
                f"[{self.__class__.__name__}] g: erreur de split/shape: {e}"
            ) from e
        # print("BaseModelNonLinear - g 3")

    def jacobiens_g(self, z: np.ndarray, noise_z: np.ndarray, dt: float) -> np.ndarray:
        if __debug__:
            if z.ndim == 2:
                assert all(a.shape == (self.dim_xy, 1) for a in (z, noise_z))
            else:
                assert all(
                    a.ndim == 3 and a.shape[1:] == (self.dim_xy, 1)
                    for a in (z, noise_z)
                )
                assert z.shape[0] == noise_z.shape[0]

        try:
            # print("BaseModelNonLinear - jacobiens_g")
            axis = 1 if z.ndim == 3 else 0
            x, y = np.split(z, [self.dim_x], axis=axis)
            nx, ny = np.split(noise_z, [self.dim_x], axis=axis)
            return self._jacobiens_g(x, y, nx, ny, dt)
        except NumericalError:
            raise
        except ValueError as e:
            raise NumericalError(
                f"[{self.__class__.__name__}] jacobiens_g: erreur de split/shape: {e}"
            ) from e

    # ------------------------------------------------------------------
    def _g(self, x, y, nx, ny, dt):
        """À implémenter dans la sous-classe"""
        raise NotImplementedError

    def _jacobiens_g(self, x, y, nx, ny, dt):
        """À implémenter dans la sous-classe"""
        raise NotImplementedError

    # ------------------------------------------------------------------
    def get_params(self) -> dict:
        return {
            "dim_x": self.dim_x,
            "dim_y": self.dim_y,
            "augmented": self.augmented,
            "g": self.g,
            "jacobiens_g": self.jacobiens_g,  # pour EPKF
            "alpha": self.alpha,  # pour UPKF
            "beta": self.beta,  # pour UPKF
            "kappa": self.kappa,  # pour UPKF
            "lambda_": self.lambda_,  # pour UPKF
            "mQ": self.mQ,
            "mz0": self.mz0,
            "Pz0": self.Pz0,
        }

    # ------------------------------------------------------------------
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(dim_x={self.dim_x}, dim_y={self.dim_y})"

    # ------------------------------------------------------------------
    def plot_g_dynamic(self, n_points: int = 150, quiver_stride: int = 10) -> None:
        """
        Plot g(z, noise=0, dt=1) over [-1,1]^2.
        Only available for dim_x=1 and dim_y=1.
        Saves figure into data/plot/.
        """

        if self.dim_x != 1 or self.dim_y != 1:
            return

        z1 = np.linspace(-3, 3, n_points)
        z2 = np.linspace(-3, 3, n_points)
        Z1, Z2 = np.meshgrid(z1, z2)

        Z_stack = np.stack([Z1.ravel(), Z2.ravel()], axis=1)

        noise = np.zeros((self.dim_xy, 1))
        dt = 1.0

        G = np.zeros_like(Z_stack)

        try:
            with np.errstate(all="raise"):
                for k in range(Z_stack.shape[0]):
                    z = Z_stack[k].reshape(self.dim_xy, 1)
                    g_val = self.g(z, noise, dt)

                    if not np.isfinite(g_val).all():
                        raise FloatingPointError("Non-finite value in g")

                    G[k, :] = g_val.ravel()

        except FloatingPointError as e:
            raise NumericalError(
                f"[{self.__class__.__name__}] plot_g: floating point failure: {e}"
            ) from e

        G1 = G[:, 0].reshape(n_points, n_points)
        G2 = G[:, 1].reshape(n_points, n_points)

        NormG = np.sqrt(G1**2 + G2**2)
        Dz1 = G1 - Z1
        Dz2 = G2 - Z2

        os.makedirs("data/plot", exist_ok=True)

        fig = plt.figure(figsize=(10, 7))

        # Surface g_x
        ax1 = fig.add_subplot(2, 2, 1, projection="3d")
        ax1.plot_surface(Z1, Z2, G1)
        ax1.set_title(r"$g_x(x,y)$", size=BIG_SIZE)
        ax1.set_xlabel(r"$x$")
        ax1.set_ylabel(r"$y$")
        ax1.view_init(30, 45)

        # Surface g_y
        ax2 = fig.add_subplot(2, 2, 2, projection="3d")
        ax2.plot_surface(Z1, Z2, G2)
        ax2.set_title(r"$g_y(x,y)$", size=BIG_SIZE)
        ax2.set_xlabel(r"$x$")
        ax2.set_ylabel(r"$y$")
        ax2.view_init(30, 45)

        # Norme
        ax3 = fig.add_subplot(2, 2, 3)
        im = ax3.contourf(Z1, Z2, NormG)
        ax3.set_title(r"$\|g(x,y)\|$", size=BIG_SIZE)
        ax3.set_xlabel(r"$x$")
        ax3.set_ylabel(r"$y$")
        plt.colorbar(im, ax=ax3)

        # Champ dynamique
        ax4 = fig.add_subplot(2, 2, 4)
        ax4.quiver(
            Z1[::quiver_stride, ::quiver_stride],
            Z2[::quiver_stride, ::quiver_stride],
            Dz1[::quiver_stride, ::quiver_stride],
            Dz2[::quiver_stride, ::quiver_stride],
        )
        ax4.set_title(r"$g(x, y) - (x, y)$", size=BIG_SIZE)
        ax4.set_xlabel(r"$x$")
        ax4.set_ylabel(r"$y$")
        ax4.set_aspect("equal")

        model_name = getattr(self, "MODEL_NAME", self.__class__.__name__)
        filename = f"data/plot/function_g_dynamic_{model_name}.png"
        plt.savefig(filename, dpi=DPI, bbox_inches="tight", facecolor=FACECOLOR)
        plt.close(fig)
