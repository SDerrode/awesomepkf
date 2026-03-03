#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
prg/exceptions.py
-----------------
Hiérarchie centralisée des exceptions du projet PKF.

Arborescence
------------
PKFError
├── ParamError
└── NumericalError
│   ├── CovarianceError
│   └── InvertibilityError
└── FilterError
    └── StepValidationError
"""

from __future__ import annotations

__all__ = [
    "PKFError",
    "ParamError",
    "NumericalError",
    "CovarianceError",
    "InvertibilityError",
    "FilterError",
    "StepValidationError",
]


# ---------------------------------------------------------------------------
# Racine
# ---------------------------------------------------------------------------


class PKFError(Exception):
    """
    Racine de toutes les exceptions du projet PKF.

    Tous les ``except`` de haut niveau peuvent attraper cette classe pour
    intercepter n'importe quelle erreur du projet en un seul bloc.
    """

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.args[0]!r})"


# ---------------------------------------------------------------------------
# Erreurs de paramètres
# ---------------------------------------------------------------------------


class ParamError(PKFError):
    """
    Paramètre invalide fourni à une classe ou une méthode du projet.

    Levée par exemple si ``sKey`` n'est pas un entier strictement positif,
    ou si ``verbose`` n'appartient pas à ``{0, 1, 2}``.
    """


# ---------------------------------------------------------------------------
# Erreurs numériques
# ---------------------------------------------------------------------------


class NumericalError(PKFError):
    """
    Erreur numérique générique.

    Classe de base pour toutes les erreurs liées aux calculs matriciels.
    Porte le contexte structuré (step, matrix_name) partagé par ses
    sous-classes, ce qui évite de parser les messages texte chez l'appelant.

    Parameters
    ----------
    message : str
        Description humaine de l'erreur.
    matrix_name : str, optional
        Nom de la matrice concernée (default ``""``).
    step : int, optional
        Indice du pas de temps où l'erreur s'est produite (default ``-1``).

    Attributes
    ----------
    matrix_name : str
        Nom de la matrice concernée.
    step : int
        Indice du pas de temps où l'erreur s'est produite. ``-1`` si inconnu.

    Examples
    --------
    >>> raise CovarianceError("not PSD", matrix_name="PXX", step=42)
    >>> except NumericalError as e:
    ...     print(e.step, e.matrix_name)
    42 PXX
    """

    def __init__(self, message: str, matrix_name: str = "", step: int = -1) -> None:
        super().__init__(message)
        self.matrix_name = matrix_name
        self.step = step

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"step={self.step}, "
            f"matrix={self.matrix_name!r}, "
            f"msg={self.args[0]!r})"
        )


class CovarianceError(NumericalError):
    """
    Matrice de covariance invalide.

    Levée lorsqu'une matrice n'est pas symétrique définie positive,
    ou lorsque la tentative de régularisation a échoué.
    """


class InvertibilityError(NumericalError):
    """
    Matrice non inversible.

    Levée lorsqu'une matrice attendue inversible (ex. ``Skp1``, ``Sigma22``)
    ne passe pas le diagnostic d'inversibilité.
    """


# ---------------------------------------------------------------------------
# Erreurs de filtre
# ---------------------------------------------------------------------------


class FilterError(PKFError):
    """
    Erreur générique du filtre PKF.

    Classe de base pour les erreurs survenant pendant l'exécution du filtre,
    indépendamment des calculs matriciels internes.
    """


class StepValidationError(FilterError):
    """
    Échec de la construction d'un ``PKFStep``.

    Levée lorsque les données transmises au constructeur de ``PKFStep``
    sont invalides (mauvaises formes, incohérences entre champs, etc.).

    Parameters
    ----------
    message : str
        Description humaine de l'erreur.
    step : int, optional
        Indice du pas de temps où l'erreur s'est produite (default ``-1``).

    Attributes
    ----------
    step : int
        Indice du pas de temps où l'erreur s'est produite. ``-1`` si inconnu.
    """

    def __init__(self, message: str, step: int = -1) -> None:
        super().__init__(message)
        self.step = step

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(" f"step={self.step}, " f"msg={self.args[0]!r})"
        )
