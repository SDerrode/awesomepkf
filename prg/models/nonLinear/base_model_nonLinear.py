#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import os
import matplotlib.pyplot as plt

from prg.classes.SeedGenerator import SeedGenerator
from prg.utils.exceptions import NumericalError
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

        self.pairwiseModel = None  # renseigné par l'un des 2 classes filles

        self._randMatrices = SeedGenerator(9)

    # ------------------------------------------------------------------
    def g(self, z: np.ndarray, noise_z: np.ndarray, dt: float) -> np.ndarray:

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
            return self._g(x, y, nx, ny, dt)
        except NumericalError:
            raise
        except ValueError as e:
            raise NumericalError(
                f"[{self.__class__.__name__}] g: erreur de split/shape: {e}"
            ) from e

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

    def latex_model(self) -> str:
        """
        Retourne la représentation LaTeX du modèle (équations + jacobiennes).
        À implémenter dans la sous-classe.
        """
        raise NotImplementedError

    # ------------------------------------------------------------------
    @staticmethod
    def _wrap_lambdify(f):
        """
        Garantit que le résultat de sp.lambdify est toujours un callable.

        Quand une expression SymPy est constante (aucun symbole libre,
        ex. Bn = I pour un bruit additif, ou H = [0,1] pour hx = x[-1]),
        lambdify génère une fonction qui retourne directement un ndarray
        au lieu d'un callable acceptant des arguments.
        Ce wrapper détecte ce cas et retourne une lambda qui ignore ses
        arguments et renvoie toujours la valeur constante.
        """
        if callable(f):
            return f
        constant = np.array(f, dtype=float)
        return lambda *args: constant

    # ------------------------------------------------------------------
    def get_params(self) -> dict:
        return {
            "dim_x": self.dim_x,
            "dim_y": self.dim_y,
            "augmented": self.augmented,
            "pairwiseModel": self.pairwiseModel,
            "g": self.g,
            "f": getattr(self, "_fx", None),
            "h": getattr(self, "_hx", None),
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
    # Helpers communs aux méthodes de visualisation
    # ------------------------------------------------------------------

    def _make_grid(
        self, n_points: int, z_range: tuple[float, float] = (-3.0, 3.0)
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Construit une grille régulière 2D sur z_range × z_range.

        Retourne
        --------
        Z1, Z2 : (n_points, n_points)  — meshgrid
        Z_stack : (n_points², 2)        — liste de points z = [x, y]ᵀ
        """
        z = np.linspace(*z_range, n_points)
        Z1, Z2 = np.meshgrid(z, z)
        Z_stack = np.stack([Z1.ravel(), Z2.ravel()], axis=1)
        return Z1, Z2, Z_stack

    def _eval_g_on_grid(self, Z_stack: np.ndarray, n_points: int) -> np.ndarray:
        """
        Évalue g(z, noise=0, dt=1) sur toute la grille.

        Retourne
        --------
        G : (n_points², 2) — valeurs de g ; NaN aux points singuliers.
        """
        noise = np.zeros((self.dim_xy, 1))
        G = np.full((Z_stack.shape[0], self.dim_xy), np.nan)
        for k in range(Z_stack.shape[0]):
            z = Z_stack[k].reshape(self.dim_xy, 1)
            try:
                val = self.g(z, noise, 1.0)
                if np.isfinite(val).all():
                    G[k] = val.ravel()
            except (NumericalError, FloatingPointError):
                pass  # laisse NaN en place
        return G

    def _eval_jac_on_grid(self, Z_stack: np.ndarray, n_points: int) -> np.ndarray:
        """
        Évalue An = jacobiens_g(z, noise=0, dt=1)[0] sur toute la grille.

        Retourne
        --------
        AN : (n_points², dim_xy, dim_xy) — An à chaque point ; NaN aux singularités.
        """
        nz = self.dim_xy
        noise = np.zeros((nz, 1))
        AN = np.full((Z_stack.shape[0], nz, nz), np.nan)
        for k in range(Z_stack.shape[0]):
            z = Z_stack[k].reshape(nz, 1)
            try:
                An, _ = self.jacobiens_g(z, noise, 1.0)
                if np.isfinite(An).all():
                    AN[k] = An
            except (NumericalError, FloatingPointError):
                pass
        return AN

    @staticmethod
    def _contourf_ax(
        ax: plt.Axes,
        Z1: np.ndarray,
        Z2: np.ndarray,
        data: np.ndarray,
        title: str,
        cmap: str = "RdBu_r",
        sym: bool = False,
    ) -> None:
        """
        Trace un contourf avec colorbar sur un axe existant.

        Paramètres
        ----------
        sym   : si True, centre la colormap sur 0 (utile pour valeurs signées).
        """
        kwargs = dict(cmap=cmap, levels=20)
        if sym:
            vmax = np.nanmax(np.abs(data))
            if vmax > 0:
                kwargs["vmin"] = -vmax
                kwargs["vmax"] = vmax
        im = ax.contourf(Z1, Z2, data, **kwargs)
        ax.set_title(title, size=BIG_SIZE)
        ax.set_xlabel(r"$x$")
        ax.set_ylabel(r"$y$")
        plt.colorbar(im, ax=ax)

    # ------------------------------------------------------------------
    def plot_g_dynamic(
        self,
        n_points: int = 150,
        quiver_stride: int = 10,
        z_range: tuple[float, float] = (-3.0, 3.0),
    ) -> None:
        """
        Visualise g(z, noise=0, dt=1) sur z_range × z_range.
        Disponible uniquement pour dim_x=1, dim_y=1.
        Sauvegarde la figure dans data/plot/.

        Sous-figures
        ------------
        (1,1) Surface 3D de g_x(x, y)
        (1,2) Surface 3D de g_y(x, y)
        (2,1) Carte de la norme ‖g(x, y)‖
        (2,2) Champ de déplacement g(z) − z  (quiver)
        """
        if self.dim_x != 1 or self.dim_y != 1:
            return

        Z1, Z2, Z_stack = self._make_grid(n_points, z_range)
        G = self._eval_g_on_grid(Z_stack, n_points)

        G1 = G[:, 0].reshape(n_points, n_points)
        G2 = G[:, 1].reshape(n_points, n_points)
        NormG = np.sqrt(G1**2 + G2**2)
        Dz1, Dz2 = G1 - Z1, G2 - Z2

        os.makedirs("data/plot", exist_ok=True)
        fig = plt.figure(figsize=(12, 8))

        # Surface g_x
        ax1 = fig.add_subplot(2, 2, 1, projection="3d")
        ax1.plot_surface(Z1, Z2, G1, cmap="viridis")
        ax1.set_title(r"$g_x(x,y)$", size=BIG_SIZE)
        ax1.set_xlabel(r"$x$")
        ax1.set_ylabel(r"$y$")
        ax1.view_init(30, 45)

        # Surface g_y
        ax2 = fig.add_subplot(2, 2, 2, projection="3d")
        ax2.plot_surface(Z1, Z2, G2, cmap="viridis")
        ax2.set_title(r"$g_y(x,y)$", size=BIG_SIZE)
        ax2.set_xlabel(r"$x$")
        ax2.set_ylabel(r"$y$")
        ax2.view_init(30, 45)

        # Norme ‖g‖
        ax3 = fig.add_subplot(2, 2, 3)
        self._contourf_ax(ax3, Z1, Z2, NormG, r"$\|g(x,y)\|$", cmap="plasma")

        # Champ de déplacement
        ax4 = fig.add_subplot(2, 2, 4)
        s = quiver_stride
        ax4.quiver(Z1[::s, ::s], Z2[::s, ::s], Dz1[::s, ::s], Dz2[::s, ::s])
        ax4.set_title(r"$g(x,y) - (x,y)$", size=BIG_SIZE)
        ax4.set_xlabel(r"$x$")
        ax4.set_ylabel(r"$y$")
        ax4.set_aspect("equal")

        model_name = getattr(self, "MODEL_NAME", self.__class__.__name__)
        fig.suptitle(model_name, fontsize=BIG_SIZE + 2, fontweight="bold")
        plt.tight_layout()
        path = f"data/plot/function_g_dynamic_{model_name}.png"
        plt.savefig(path, dpi=DPI, bbox_inches="tight", facecolor=FACECOLOR)
        plt.close(fig)

    # ------------------------------------------------------------------
    def plot_jacobian_dynamic(
        self,
        n_points: int = 100,
        z_range: tuple[float, float] = (-3.0, 3.0),
    ) -> None:
        """
        Visualise An = dg/dz évalué en (z, noise=0, dt=1) sur z_range × z_range.
        Disponible uniquement pour dim_x=1, dim_y=1 (An est 2×2).
        Sauvegarde la figure dans data/plot/.

        Sous-figures  (2 lignes × 3 colonnes)
        -------------
        Ligne 1 — entrées de la 1ʳᵉ ligne de An :
          (1,1) ∂g_x/∂x   (1,2) ∂g_x/∂y   (1,3) Re(λ₁)
        Ligne 2 — entrées de la 2ᵉ ligne de An :
          (2,1) ∂g_y/∂x   (2,2) ∂g_y/∂y   (2,3) Re(λ₂)

        Les champs Re(λ) donnent la stabilité locale : |Re(λ)| < 1 indique
        une contraction dans cette direction propre (temps discret).
        La frontière |Re(λ)| = 1 est tracée en pointillés blancs.
        """
        if self.dim_x != 1 or self.dim_y != 1:
            return

        Z1, Z2, Z_stack = self._make_grid(n_points, z_range)
        AN = self._eval_jac_on_grid(Z_stack, n_points)  # (n², 2, 2)

        # Reshape des 4 entrées
        A00 = AN[:, 0, 0].reshape(n_points, n_points)
        A01 = AN[:, 0, 1].reshape(n_points, n_points)
        A10 = AN[:, 1, 0].reshape(n_points, n_points)
        A11 = AN[:, 1, 1].reshape(n_points, n_points)

        # Valeurs propres : eigenvalues de chaque matrice 2×2
        # AN shape (n², 2, 2) → eigvals shape (n², 2)
        eigvals = np.linalg.eigvals(AN)  # (n², 2), complexes
        lam1 = np.real(eigvals[:, 0]).reshape(n_points, n_points)
        lam2 = np.real(eigvals[:, 1]).reshape(n_points, n_points)

        os.makedirs("data/plot", exist_ok=True)
        fig, axes = plt.subplots(2, 3, figsize=(15, 9))

        panels = [
            (axes[0, 0], A00, r"$\partial g_x / \partial x$"),
            (axes[0, 1], A01, r"$\partial g_x / \partial y$"),
            (axes[1, 0], A10, r"$\partial g_y / \partial x$"),
            (axes[1, 1], A11, r"$\partial g_y / \partial y$"),
        ]
        for ax, data, title in panels:
            self._contourf_ax(ax, Z1, Z2, data, title, cmap="RdBu_r", sym=True)

        # Champs Re(λ) avec isocourbe |Re(λ)| = 1
        for ax, lam, title in [
            (axes[0, 2], lam1, r"$\mathrm{Re}(\lambda_1)$"),
            (axes[1, 2], lam2, r"$\mathrm{Re}(\lambda_2)$"),
        ]:
            self._contourf_ax(ax, Z1, Z2, lam, title, cmap="RdBu_r", sym=True)
            # Frontière de stabilité |Re(λ)| = 1
            for level, ls in [(-1.0, "--"), (1.0, "--")]:
                try:
                    ax.contour(
                        Z1,
                        Z2,
                        lam,
                        levels=[level],
                        colors="white",
                        linestyles=ls,
                        linewidths=1.2,
                    )
                except Exception:
                    pass

        model_name = getattr(self, "MODEL_NAME", self.__class__.__name__)
        fig.suptitle(
            rf"{model_name} — Jacobienne $A_n = \partial g / \partial z$",
            fontsize=BIG_SIZE + 2,
            fontweight="bold",
        )
        plt.tight_layout()
        path = f"data/plot/jacobian_dynamic_{model_name}.png"
        plt.savefig(path, dpi=DPI, bbox_inches="tight", facecolor=FACECOLOR)
        plt.close(fig)
