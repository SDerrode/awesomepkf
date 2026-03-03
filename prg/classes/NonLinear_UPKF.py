#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
####################################################################
Unscented Pairwise Kalman filter (UPKF) implementation
####################################################################
"""

from __future__ import annotations
from typing import Generator, Optional
import numpy as np
from scipy.linalg import LinAlgError

from prg.classes.PKF import PKF
from prg.classes.SigmaPointsSet import SigmaPointsSet
from prg.exceptions import FilterError, InvertibilityError, NumericalError, ParamError

__all__ = ["NonLinear_UPKF"]


class NonLinear_UPKF(PKF):
    """
    Unscented Pairwise Kalman Filter (UPKF).

    Extends :class:`PKF` by introducing the UPKF.
    """

    def __init__(
        self,
        param,
        sigmaSet: str,
        sKey: Optional[int] = None,
        verbose: int = 0,
    ) -> None:
        """
        Initialise le filtre UPKF.

        Parameters
        ----------
        param : ParamLinear | ParamNonLinear
            Paramètres du modèle.
        sigmaSet : str
            Clé du jeu de sigma-points dans ``SigmaPointsSet.registry``.
        sKey : int, optional
            Graine aléatoire pour la reproductibilité.
        verbose : int, optional
            Niveau de verbosité (défaut 0).

        Raises
        ------
        ParamError
            Si ``sigmaSet`` n'est pas une clé connue du registre.
        """
        super().__init__(param, sKey, verbose)

        try:
            cls = SigmaPointsSet.registry[sigmaSet]
        except KeyError:
            raise ParamError(
                f"Jeu de sigma-points inconnu : {sigmaSet!r}. "
                f"Disponibles : {list(SigmaPointsSet.registry.keys())}."
            )

        self.sigma_point_set_obj = cls(
            dim=2 * self.dim_x + self.dim_y, param=self.param
        )

    def process_filter(
        self,
        N: Optional[int] = None,
        data_generator: Optional[
            Generator[tuple[int, np.ndarray, np.ndarray], None, None]
        ] = None,
    ) -> Generator[tuple[int, np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
        """
        Exécute le filtre UPKF comme générateur.

        Parameters
        ----------
        N : int, optional
            Nombre maximal de pas de temps. Si ``None``, tourne jusqu'à
            épuisement du générateur de données.
        data_generator : Generator, optional
            Générateur externe de données. Si ``None``, le générateur
            interne est utilisé.

        Yields
        ------
        k : int
            Indice temporel courant.
        x_true : np.ndarray or None
            Vérité terrain à l'instant ``k``.
        y_observed : np.ndarray
            Observation à l'instant ``k``.
        X_predict : np.ndarray
            Estimée a priori, shape ``(dim_x, 1)``.
        X_update : np.ndarray
            Estimée a posteriori, shape ``(dim_x, 1)``.

        Raises
        ------
        ParamError
            Si ``N`` n'est pas un entier strictement positif ou ``None``
            (levée par :meth:`_validate_N` dans le parent).
        InvertibilityError
            Si la matrice de covariance d'innovation ``Skp1`` n'est pas
            inversible lors de l'étape de mise à jour.
        NumericalError
            Si la matrice de covariance prédite ``Pkp1_predict`` n'est pas
            valide (levée par :meth:`_check_covariance`).
        FilterError
            Si une erreur inattendue survient pendant l'étape de mise à jour.
        """
        self._validate_N(N)

        generator = (
            data_generator if data_generator is not None else self._data_generation()
        )

        # --- First estimate -----------------------------------------------------------
        step = self._firstEstimate(generator)
        if step.xkp1 is None:
            self.ground_truth = False

        yield step.k, step.xkp1, step.ykp1, step.Xkp1_predict, step.Xkp1_update

        # --- Subsequent steps ---------------------------------------------------------
        za = np.zeros((2 * self.dim_x + self.dim_y, 1))
        Pa_base = np.zeros((2 * self.dim_x + self.dim_y, 2 * self.dim_x + self.dim_y))
        Pa_base[self.dim_x :, self.dim_x :] = self.param.mQ
        Pkp1_predict = self.zeros_dim_xy_xy.copy()

        while N is None or step.k < N:

            # Sigma points et leur propagation par g
            za[: self.dim_x] = step.Xkp1_update
            Pa = Pa_base.copy()
            Pa[: self.dim_x, : self.dim_x] = step.PXXkp1_update

            sigma_without_y = self.sigma_point_set_obj._sigma_point(za, Pa)
            sigma_with_y = [
                np.concatenate([s[: self.dim_x], step.ykp1, s[self.dim_x :]], axis=0)
                for s in sigma_without_y
            ]
            sigma_propag = [
                self.param.g(*np.split(spoint, [self.dim_xy]), self.dt)
                for spoint in sigma_with_y
            ]

            # Prediction
            Zkp1_predict = np.sum(
                self.sigma_point_set_obj.Wm[:, None, None] * sigma_propag, axis=0
            )

            Pkp1_predict.fill(0.0)
            diffs = np.array(sigma_propag) - Zkp1_predict  # (n, dim, 1)
            Pkp1_predict = np.einsum(
                "i,ijk,ilk->jl", self.sigma_point_set_obj.Wc, diffs, diffs
            )

            # Validate predicted covariance — lève CovarianceError si invalide
            self._check_covariance(Pkp1_predict, step.k, name="Pkp1_predict")

            # Consume the next observation
            try:
                new_k, new_xkp1, new_ykp1 = next(generator)
            except StopIteration:
                return  # Data generator exhausted — arrêt normal, pas une erreur

            # Update step — les exceptions custom remontent naturellement
            try:
                step = self._nextUpdating(
                    new_k, new_xkp1, new_ykp1, Zkp1_predict, Pkp1_predict
                )
            except (InvertibilityError, NumericalError):
                # Erreurs numériques connues — on les laisse remonter telles quelles
                raise
            except Exception as e:
                raise FilterError(
                    f"Step {new_k}: unexpected error during update step."
                ) from e

            yield step.k, step.xkp1, step.ykp1, step.Xkp1_predict, step.Xkp1_update
