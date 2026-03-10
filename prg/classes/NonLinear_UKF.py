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
       calcul du gain via :meth:`PKF._nextUpdating`.

    Parameters
    ----------
    param : ParamLinear | ParamNonLinear
        Paramètres du modèle.  Doit exposer :

        * ``mQ`` — covariance du bruit de processus,
          shape ``(dim_xy, dim_xy)`` ou ``(dim_x, dim_x)`` ;
        * ``mR`` — covariance du bruit d'observation,
          shape ``(dim_y, dim_y)`` ;
        * ``f(x, t, dt)`` — fonction de transition vectorisée ;
        * ``h(x, u, dt)`` — fonction d'observation vectorisée.

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

        if self.param.pairwiseModel == True:
            raise PKFError(
                f"Failed to process a pairwise model {model_name!r} with UKF."
            )

        # Jeu de sigma-points pour l'étape de prédiction (espace d'état dim_x)
        self.sigma_pred_set = cls(dim=self.dim_x, param=self.param)

        # Jeu de sigma-points pour l'étape de mise à jour (espace d'état dim_x)
        self.sigma_upd_set = cls(dim=self.dim_x, param=self.param)

        # Extraction de Q_x et R une seule fois — évite les découpages en boucle.
        # mQ peut être (dim_xy, dim_xy) si le param est partagé avec l'UPKF.
        raw_Q = self.param.mQ
        self._Q_x: np.ndarray = raw_Q[: self.dim_x, : self.dim_x]
        self._R: np.ndarray = raw_Q[self.dim_x :, self.dim_x :]

    # ------------------------------------------------------------------
    # Équations du modèle — vectorisées sur l'axe batch des sigma-points
    # ------------------------------------------------------------------

    def _fx(self, x: np.ndarray, t: int, dt: int) -> np.ndarray:
        """
        Équation d'état vectorisée (sans bruit — Q injecté additivement).

        Parameters
        ----------
        x : np.ndarray
            Sigma-points d'état, shape ``(n_sigma, dim_x, 1)``.
        t : int
            Indice temporel courant.
        dt : int
            Pas de temps.

        Returns
        -------
        np.ndarray
            Sigma-points propagés, shape ``(n_sigma, dim_x, 1)``.
        """
        return self.param.f(x, t, dt)

    def _hx(self, x: np.ndarray, u: np.ndarray, dt: int) -> np.ndarray:
        """
        Équation d'observation vectorisée (sans bruit — R injecté additivement).

        Parameters
        ----------
        x : np.ndarray
            Sigma-points d'état prédit, shape ``(n_sigma, dim_x, 1)``.
        u : np.ndarray
            Terme de bruit nul ou entrée auxiliaire,
            shape ``(n_sigma, dim_y, 1)``.
        dt : int
            Pas de temps.

        Returns
        -------
        np.ndarray
            Sigma-points dans l'espace observation,
            shape ``(n_sigma, dim_y, 1)``.
        """
        return self.param.h(x, u, dt)

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

        input("  ATTENTE - process_filter")
        self._validate_N(N)

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

        # Terme auxiliaire nul pour _hx — alloué une seule fois au premier pas
        zeros_u: Optional[np.ndarray] = None

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

            input("  ATTENTE - process_filter aaa")

            # Propagation vectorisée par f  →  f(σ_i)
            sigma_f = self._fx(sigma_pred, step.k, self.dt)  # (n_sigma, dim_x, 1)

            input("  ATTENTE - process_filter aaa")

            # Moyenne prédite  x_pred = Σ Wm_i · f(σ_i)
            x_pred: np.ndarray = np.sum(
                self.sigma_pred_set.Wm[:, None, None] * sigma_f, axis=0
            )  # (dim_x, 1)

            input("  ATTENTE - process_filter bbb")

            # Covariance prédite  P_xx_pred = Σ Wc_i · δf_i δf_iᵀ  +  Q
            diffs_f = sigma_f - x_pred  # (n_sigma, dim_x, 1)
            P_xx_pred: np.ndarray = (
                np.einsum("i,ijk,ilk->jl", self.sigma_pred_set.Wc, diffs_f, diffs_f)
                + self._Q_x
            )  # (dim_x, dim_x)

            # Validation — lève NumericalError si invalide
            self._check_covariance(P_xx_pred, step.k, name="P_xx_pred")

            input("  ATTENTE - process_filter cccc")

            # ================================================================
            # ÉTAPE DE MISE À JOUR (sigma-points sur l'état prédit)
            # ================================================================

            # Sigma-points sur (x_pred, P_xx_pred) — dimension dim_x
            sigma_upd_list = self.sigma_upd_set._sigma_point(x_pred, P_xx_pred)
            sigma_upd = np.array(sigma_upd_list)  # (n_sigma, dim_x, 1)

            # Terme auxiliaire nul pour _hx — alloué une seule fois
            if zeros_u is None:
                zeros_u = np.zeros((n_sigma, self.dim_y, 1))

            # Propagation vectorisée par h  →  h(σ_i)
            sigma_h = self._hx(sigma_upd, zeros_u, self.dt)  # (n_sigma, dim_y, 1)

            input("  ATTENTE - process_filter dddd")

            # Observation prédite  y_pred = Σ Wm_i · h(σ_i)
            y_pred: np.ndarray = np.sum(
                self.sigma_upd_set.Wm[:, None, None] * sigma_h, axis=0
            )  # (dim_y, 1)

            # Covariance d'innovation  P_yy = Σ Wc_i · δh_i δh_iᵀ  +  R
            diffs_h = sigma_h - y_pred  # (n_sigma, dim_y, 1)
            P_yy: np.ndarray = (
                np.einsum("i,ijk,ilk->jl", self.sigma_upd_set.Wc, diffs_h, diffs_h)
                + self._R
            )  # (dim_y, dim_y)

            input("  ATTENTE - process_filter eeee")

            # Covariance croisée  P_xy = Σ Wc_i · δx_i δh_iᵀ
            diffs_x = sigma_upd - x_pred  # (n_sigma, dim_x, 1)
            P_xy: np.ndarray = np.einsum(
                "i,ijk,ilk->jl", self.sigma_upd_set.Wc, diffs_x, diffs_h
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

            input("  ATTENTE - process_filter fffff")

            # Consommation de la prochaine observation
            try:
                new_k, new_xkp1, new_ykp1 = next(generator)
            except StopIteration:
                return  # générateur épuisé — arrêt normal, pas une erreur

            input("  ATTENTE - process_filter gggg")

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

            input("  ATTENTE - process_filter hhhhh")

            yield step.k, step.xkp1, step.ykp1, step.Xkp1_predict, step.Xkp1_update
