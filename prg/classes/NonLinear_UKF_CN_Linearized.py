#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
####################################################################
Unscented Kalman Filter (UKF) — non-linéaire, bruit additif
####################################################################

Différences par rapport à l'UPKF :
  - Pas de structure « pairwise » : l'état n'est pas augmenté avec
    l'observation.
  - Deux jeux de sigma-points indépendants :
      * ``sigma_pred_set``  (dim = dim_x) pour l'étape de prédiction,
      * ``sigma_upd_set``   (dim = dim_x) pour l'étape de mise à jour.
  - Q et R sont injectés de façon **additive** (UKF non-augmenté).
  - La corrélation croisée M = E[t·uᵀ] est prise en compte dans
    P_xy et P_yy (correction au premier ordre via la jacobienne H).
  - ``_fx`` et ``_hx`` encapsulent respectivement l'équation d'état
    et l'équation d'observation ; les deux sont vectorisées sur
    l'axe des sigma-points (batch axis 0).
  - À la fin de chaque cycle, un bloc (Zkp1_predict, Pkp1_predict)
    de dimension (dim_xy × dim_xy) est assemblé afin de réutiliser
    :meth:`PKF._nextUpdating` sans modification.
"""

from __future__ import annotations
from typing import Generator, Optional

import numpy as np
from scipy.linalg import LinAlgError

from prg.classes.PKF import PKF
from prg.classes.SigmaPointsSet import SigmaPointsSet
from prg.utils.exceptions import (
    FilterError,
    InvertibilityError,
    NumericalError,
    ParamError,
)

__all__ = ["NonLinear_UKF"]


class NonLinear_UKF(PKF):
    """
    Unscented Kalman Filter (UKF) non-linéaire à bruit additif.

    Étend :class:`PKF` en implémentant le cycle UKF standard :

    1. **Prédiction** — sigma-points sur l'état courant ``(x, P_xx)``,
       propagation par :meth:`_fx`, covariance prédite augmentée de Q.
    2. **Mise à jour** — sigma-points sur l'état prédit ``(x_pred, P_xx_pred)``,
       propagation par :meth:`_hx`, covariance d'innovation augmentée de R,
       correction de corrélation croisée M via la jacobienne H,
       calcul du gain via :meth:`PKF._nextUpdating`.

    Parameters
    ----------
    param : ParamLinear | ParamNonLinear
        Paramètres du modèle.  Doit exposer :

        * ``mQ`` — matrice de covariance complète du bruit, shape ``(dim_xy, dim_xy)`` :

          .. code-block:: text

              mQ = [ Q   M  ]
                   [ M^T R  ]

          avec ``Q = E[t·tᵀ]``, ``R = E[u·uᵀ]``, ``M = E[t·uᵀ]``.

        * ``f(x, t, dt)`` — fonction de transition vectorisée ;
        * ``h(x, u, dt)`` — fonction d'observation vectorisée ;
        * ``jacobiens_g(z, noise_z, dt)`` — jacobiennes de g, dont on extrait dh/dx.

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
    FilterError
        Si ``param.pairwiseModel`` est ``True``.
    """

    def __init__(
        self,
        param,
        sigmaSet: str,
        sKey: Optional[int] = None,
        verbose: int = 0,
    ) -> None:

        super().__init__(param, sKey, verbose)

        try:
            cls = SigmaPointsSet.registry[sigmaSet]
        except KeyError:
            raise ParamError(
                f"Jeu de sigma-points inconnu : {sigmaSet!r}. "
                f"Disponibles : {list(SigmaPointsSet.registry.keys())}."
            )

        if self.param.pairwiseModel:
            raise FilterError(f"Failed to process a pairwise model with UKF.")

        # Jeu de sigma-points pour l'étape de prédiction (espace d'état dim_x)
        self.sigma_pred_set = cls(dim=self.dim_x, param=self.param)

        # Jeu de sigma-points pour l'étape de mise à jour (espace d'état dim_x)
        self.sigma_upd_set = cls(dim=self.dim_x, param=self.param)

        # Extraction des blocs de mQ une seule fois — évite les découpages en boucle.
        #
        #   mQ = [ Q_x   M  ]
        #        [ M^T   R  ]
        #
        self._Q_x: np.ndarray = self.param.mQ[: self.dim_x, : self.dim_x]
        self._R: np.ndarray = self.param.mQ[self.dim_x :, self.dim_x :]
        self._M: np.ndarray = self.param.mQ[: self.dim_x, self.dim_x :]

    # ------------------------------------------------------------------
    # Boucle principale du filtre
    # ------------------------------------------------------------------

    def process_filter(
        self,
        N: Optional[int] = None,
        data_generator: Optional[
            Generator[tuple[int, np.ndarray, np.ndarray], None, None]
        ] = None,
    ) -> Generator[tuple[int, np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
        """
        Exécute le filtre UKF comme générateur.

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
            Si ``N`` n'est pas un entier strictement positif ou ``None``.
        InvertibilityError
            Si la matrice d'innovation ``Skp1`` n'est pas inversible.
        NumericalError
            Si la covariance prédite ``P_xx_pred`` n'est pas valide.
        FilterError
            Si une erreur inattendue survient pendant la mise à jour.
        """

        self._validate_N(N)
        self.history.clear()

        generator = (
            data_generator if data_generator is not None else self._data_generation()
        )

        # --- Première estimée (conditionnement gaussien sur y_0) ------------------
        step = self._firstEstimate(generator)
        if step.xkp1 is None:
            self.ground_truth = False

        yield step.k, step.xkp1, step.ykp1, step.Xkp1_predict, step.Xkp1_update

        # Pré-allocation du bloc de covariance augmentée réutilisé à chaque pas
        Pkp1_predict = self.zeros_dim_xy_xy.copy()

        # Vecteurs nuls pour bruit — alloués une seule fois au premier pas
        zeros_x: Optional[np.ndarray] = None
        zeros_y: Optional[np.ndarray] = None

        # --- Boucle principale ----------------------------------------------------
        while N is None or step.k < N:

            # ================================================================
            # ÉTAPE DE PRÉDICTION
            # ================================================================

            # Sigma-points sur (x_k, P_xx_k) — dimension dim_x
            sigma_pred_list = self.sigma_pred_set._sigma_point(
                step.Xkp1_update, step.PXXkp1_update
            )
            sigma_pred = np.array(sigma_pred_list)  # (n_sigma, dim_x, 1)
            n_sigma = sigma_pred.shape[0]

            if zeros_x is None:
                zeros_x = np.zeros((n_sigma, self.dim_x, 1))

            # Propagation vectorisée par f  →  f(σ_i)
            sigma_f = self.param.f(sigma_pred, zeros_x, self.dt)  # (n_sigma, dim_x, 1)

            # Moyenne prédite  x_pred = Σ Wm_i · f(σ_i)
            x_pred: np.ndarray = np.sum(
                self.sigma_pred_set.Wm[:, None, None] * sigma_f, axis=0
            )  # (dim_x, 1)

            # Covariance prédite  P_xx_pred = Σ Wc_i · δf_i δf_iᵀ  +  Q
            diffs_f = sigma_f - x_pred  # (n_sigma, dim_x, 1)
            P_xx_pred: np.ndarray = (
                np.einsum("i,ijk,ilk->jl", self.sigma_pred_set.Wc, diffs_f, diffs_f)
                + self._Q_x
            )  # (dim_x, dim_x)

            # Validation — lève NumericalError si invalide
            self._check_covariance(P_xx_pred, step.k, name="P_xx_pred")

            # ================================================================
            # ÉTAPE DE MISE À JOUR (sigma-points sur l'état prédit)
            # ================================================================

            # Sigma-points sur (x_pred, P_xx_pred) — dimension dim_x
            sigma_upd_list = self.sigma_upd_set._sigma_point(x_pred, P_xx_pred)
            sigma_upd = np.array(sigma_upd_list)  # (n_sigma, dim_x, 1)

            # Terme auxiliaire nul pour _hx — alloué une seule fois
            if zeros_y is None:
                zeros_y = np.zeros((n_sigma, self.dim_y, 1))

            # Propagation vectorisée par h — bruit nul (additif : R ajouté sur P_yy)
            sigma_h = self.param.h(sigma_upd, zeros_y, self.dt)  # (n_sigma, dim_y, 1)

            # Observation prédite  y_pred = Σ Wm_i · h(σ_i)
            y_pred: np.ndarray = np.sum(
                self.sigma_upd_set.Wm[:, None, None] * sigma_h, axis=0
            )  # (dim_y, 1)

            # Jacobienne H = dh/dx évaluée en x_pred — nécessaire pour la correction M.
            # jacobiens_g attend (z, noise_z, dt) avec z de shape (dim_xy, 1).
            # Bn[dim_x:, :dim_x] = dh/dx (cf. _jacobiens_g dans base_model_fxhx).
            z_pred_aug = np.concatenate(
                [x_pred, np.zeros((self.dim_y, 1))], axis=0
            )  # (dim_xy, 1)
            _, Bn = self.param.jacobiens_g(
                z_pred_aug, np.zeros((self.dim_xy, 1)), self.dt
            )
            H_pred: np.ndarray = Bn[self.dim_x :, : self.dim_x]  # (dim_y, dim_x)

            # Covariance d'innovation
            #   P_yy = Σ Wc_i · δh_i δh_iᵀ  +  R  +  H·M + Mᵀ·Hᵀ
            # Le terme H·M + Mᵀ·Hᵀ (symétrique) assure que P_yy est cohérente
            # avec P_xy lorsque E[t·uᵀ] = M ≠ 0 (règle de la chaîne sur h∘f).
            # Quand M = 0, le terme disparaît → comportement identique au cas non corrélé.
            diffs_h = sigma_h - y_pred  # (n_sigma, dim_y, 1)
            P_yy: np.ndarray = (
                np.einsum("i,ijk,ilk->jl", self.sigma_upd_set.Wc, diffs_h, diffs_h)
                + self._R
                + H_pred @ self._M
                + self._M.T @ H_pred.T
            )  # (dim_y, dim_y)

            # Covariance croisée
            #   P_xy = Σ Wc_i · δx_i δh_iᵀ  +  M
            diffs_x = sigma_upd - x_pred  # (n_sigma, dim_x, 1)
            P_xy: np.ndarray = (
                np.einsum("i,ijk,ilk->jl", self.sigma_upd_set.Wc, diffs_x, diffs_h)
                + self._M
            )  # (dim_x, dim_y)

            # ================================================================
            # Assemblage du bloc augmenté attendu par _nextUpdating :
            #
            #   Zkp1_predict = [ x_pred ]   shape (dim_xy, 1)
            #                  [ y_pred ]
            #
            #   Pkp1_predict = [ P_xx_pred  P_xy ]   shape (dim_xy, dim_xy)
            #                  [ P_xy.T     P_yy ]
            # ================================================================
            Zkp1_predict: np.ndarray = np.concatenate(
                [x_pred, y_pred], axis=0
            )  # (dim_xy, 1)

            Pkp1_predict[: self.dim_x, : self.dim_x] = P_xx_pred
            Pkp1_predict[: self.dim_x, self.dim_x :] = P_xy
            Pkp1_predict[self.dim_x :, : self.dim_x] = P_xy.T
            Pkp1_predict[self.dim_x :, self.dim_x :] = P_yy

            # Consommation de la prochaine observation
            try:
                new_k, new_xkp1, new_ykp1 = next(generator)
            except StopIteration:
                return  # générateur épuisé — arrêt normal, pas une erreur

            # Mise à jour de Kalman — les exceptions custom remontent naturellement
            try:
                step = self._nextUpdating(
                    new_k, new_xkp1, new_ykp1, Zkp1_predict, Pkp1_predict
                )
            except (InvertibilityError, NumericalError):
                raise
            except Exception as e:
                raise FilterError(
                    f"Step {new_k}: unexpected error during update step."
                ) from e

            yield step.k, step.xkp1, step.ykp1, step.Xkp1_predict, step.Xkp1_update
