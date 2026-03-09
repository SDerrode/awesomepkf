#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Any

import numpy as np

from prg.models.nonLinear import ModelFactoryNonLinear
from prg.classes.MatrixDiagnostics import CovarianceMatrix
from prg.utils.exceptions import CovarianceError, ParamError

__all__ = ["ParamNonLinear"]


# ----------------------------------------------------------------------
# ParamNonLinear class
# ----------------------------------------------------------------------
class ParamNonLinear:
    """
    Manage Non linear parameters with optional debug checks.

    Attributes:
        verbose: logging level
        dim_x, dim_y, dim_xy: state and observation dimensions
        kwargs: models parameters
    """

    def __init__(self, verbose: int, dim_x: int, dim_y: int, **kwargs) -> None:
        """
        Initialise les paramètres du filtre PKF non-linéaire.

        Parameters
        ----------
        verbose : int
            Niveau de verbosité (0, 1 ou 2).
        dim_x : int
            Dimension de l'état, doit être un entier strictement positif.
        dim_y : int
            Dimension de l'observation, doit être un entier strictement positif.
        **kwargs
            Paramètres du modèle (g, mQ, mz0, Pz0, alpha, beta, etc.).

        Raises
        ------
        ParamError
            Si ``dim_x``, ``dim_y`` ne sont pas des entiers strictement positifs,
            ou si ``verbose`` n'appartient pas à ``{0, 1, 2}``.
        CovarianceError
            Si ``mQ`` ou ``Pz0`` ne sont pas définies positives (modèle non augmenté).
        """
        if __debug__:
            if not (isinstance(dim_x, int) and dim_x > 0):
                raise ParamError("dim_x must be a strictly positive integer.")
            if not (isinstance(dim_y, int) and dim_y > 0):
                raise ParamError("dim_y must be a strictly positive integer.")
            if verbose not in [0, 1, 2]:
                raise ParamError("verbose must be 0, 1 or 2.")

        self.dim_x = dim_x
        self.dim_y = dim_y
        self.dim_xy = dim_x + dim_y
        self.verbose = verbose

        self.augmented = kwargs["augmented"]
        self.g = kwargs["g"]

        self._mQ = np.array(kwargs["mQ"], dtype=float)
        self._mz0 = np.array(kwargs["mz0"], dtype=float)
        self._Pz0 = np.array(kwargs["Pz0"], dtype=float)

        # Paramètres spécifiques UPKF
        self.alpha = kwargs["alpha"]
        self.beta = kwargs["beta"]
        self.kappa = kwargs["kappa"]
        self.lambda_ = kwargs["lambda_"]

        # Paramètres spécifiques EPKF
        self.jacobiens_g = kwargs["jacobiens_g"]

        if __debug__:
            if not self.augmented:
                self._check_covariance_matrices()

    # ------------------------------------------------------------------
    def __repr__(self) -> str:
        return (
            f"<ParamNonLinear(dim_y={self.dim_y}, dim_x={self.dim_x}, "
            f"augmented={self.augmented}, verbose={self.verbose}, "
            f"alpha={self.alpha}, beta={self.beta}, "
            f"kappa={self.kappa}, lambda_={self.lambda_})>"
        )

    # ------------------------------------------------------------------
    # Consistency checks
    # ------------------------------------------------------------------
    def _check_covariance_matrices(self) -> None:
        """
        Vérifie que ``mQ`` et ``Pz0`` sont des matrices de covariance valides.

        Raises
        ------
        CovarianceError
            Si l'une des matrices n'est pas définie positive semi-définie.
        """
        for name, arr in [("mQ", self._mQ), ("Pz0", self._Pz0)]:
            report = CovarianceMatrix(arr).check()
            if not report.is_valid:
                raise CovarianceError(
                    f"Matrix {name!r} is not positive semi-definite.",
                    matrix_name=name,
                )

    # ------------------------------------------------------------------
    # Getters / Setters and Properties
    # ------------------------------------------------------------------
    @property
    def mz0(self) -> np.ndarray:
        return self._mz0

    @property
    def Pz0(self) -> np.ndarray:
        return self._Pz0

    @property
    def mQ(self) -> np.ndarray:
        return self._mQ

    @mQ.setter
    def mQ(self, new_Q: np.ndarray) -> None:
        """
        Met à jour la matrice de bruit de process ``mQ``.

        Parameters
        ----------
        new_Q : np.ndarray
            Nouvelle matrice de covariance, shape ``(dim_xy, dim_xy)``.

        Raises
        ------
        ParamError
            Si la forme de ``new_Q`` ne correspond pas à ``(dim_xy, dim_xy)``.
        CovarianceError
            Si ``new_Q`` n'est pas définie positive (modèle non augmenté).
        """
        new_Q = np.array(new_Q, dtype=float)
        if __debug__:
            if new_Q.shape != (self.dim_xy, self.dim_xy):
                raise ParamError(
                    f"mQ must have shape ({self.dim_xy}, {self.dim_xy}), "
                    f"got {new_Q.shape}."
                )
        self._mQ = new_Q
        if __debug__:
            if not self.augmented:
                report = CovarianceMatrix(self._mQ).check()
                if not report.is_valid:
                    raise CovarianceError(
                        "Matrix 'mQ' is not positive semi-definite after update.",
                        matrix_name="mQ",
                    )

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    def summary(self) -> None:
        """Display a complete summary of vectors and matrices."""

        def fmt(M: Any) -> str:
            return np.array2string(M, formatter={"float_kind": lambda x: f"{x:6.2f}"})

        print("=== ParamNonLinear Summary ===")
        print(f"dim_x={self.dim_x}, dim_y={self.dim_y}, verbose={self.verbose}\n")
        print("g:\n", self.g)
        print("mQ:\n", fmt(self.mQ))
        print("mz0:\n", fmt(self.mz0))
        print("Pz0:\n", fmt(self.Pz0))

        if self.verbose > 0:
            print("========================")
            print("  Q_xx:\n", fmt(self._mQ[: self.dim_x, : self.dim_x]))
            print(
                "  Q_yy:\n",
                fmt(self._mQ[self.dim_x : self.dim_xy, self.dim_x : self.dim_xy]),
            )
        print("========================")
        if self.verbose > 1:
            print("mQ = np.array(", repr(self.mQ.tolist()), ")")
            print("mz0 = np.array(", repr(self.mz0.tolist()), ")")
            print("Pz0 = np.array(", repr(self.Pz0.tolist()), ")")
