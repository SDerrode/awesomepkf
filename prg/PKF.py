#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Module PKF
Implémente un filtre de Kalman probabiliste (PKF) avec enregistrement optionnel.
"""

from __future__ import annotations

import os
import math
import logging
from typing import Generator, Optional, Tuple

import numpy as np
import pandas as pd
import scipy as sc

from classes.ParamPKF import ParamPKF
from classes.HistoryTracker import HistoryTracker
from classes.SeedGenerator import SeedGenerator


# ----------------------------------------------------------------------
# Configuration du logging
# ----------------------------------------------------------------------
logging.basicConfig(
    format="[%(levelname)s] %(message)s",
    level=logging.INFO
)
logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------
# Classe PKF
# ----------------------------------------------------------------------
class PKF:
    """Implémentation du Probabilistic Kalman Filter."""

    def __init__(
        self,
        param: ParamPKF,
        s_key: Optional[int] = None,
        save_pickle: bool = False,
        verbose: int = 0
    ):
        if not isinstance(param, ParamPKF):
            raise TypeError("param doit être un objet de la classe ParamPKF")
        if not ((isinstance(s_key, int) and s_key > 0) or s_key is None):
            raise ValueError("s_key doit être un entier strictement positif ou None")
        if verbose not in (0, 1, 2):
            raise ValueError("verbose doit être 0, 1 ou 2")
        if not isinstance(save_pickle, bool):
            raise TypeError("save_pickle doit être un booléen")

        self.param = param
        self.verbose = verbose
        self.save_pickle = save_pickle
        self._seed_gen = SeedGenerator(s_key)
        self._history = HistoryTracker() if save_pickle else None

        if self.verbose >= 1:
            logger.info(f"[PKF] Initialisé avec s_key={s_key}, verbose={verbose}, save_pickle={save_pickle}")

    # ------------------------------------------------------------------
    # Propriétés
    # ------------------------------------------------------------------
    @property
    def seed_gen(self) -> int:
        """Retourne le seed du générateur."""
        return self._seed_gen.seed

    @property
    def history(self) -> Optional[HistoryTracker]:
        """Retourne l'objet HistoryTracker si save_pickle=True, sinon None."""
        return self._history

    # ------------------------------------------------------------------
    # Générateurs
    # ------------------------------------------------------------------
    def _iterate_gen_simul(self, N: Optional[int] = None) -> Generator[Tuple[int, np.ndarray], None, None]:
        """
        Générateur qui simule Z_{k+1} = A * Z_k + W_{k+1}, 
        avec W_{k+1} ~ N(0, mQ).
        """
        k = 0
        Zkp1 = self._seed_gen.rng.multivariate_normal(mean=self.param.mu0, cov=self.param.Q1)
        yield k, Zkp1.copy()

        A = self.param.A
        while N is None or k < N:
            Zk = Zkp1
            k += 1
            Zkp1 = A @ Zk + self._seed_gen.rng.multivariate_normal(mean=self.param.mu0, cov=self.param.mQ)
            yield k, Zkp1.copy()

    # ------------------------------------------------------------------
    # PKF : Méthode Wojciech
    # ------------------------------------------------------------------
    def iterate_gen_wojciech(self, N: Optional[int] = None) -> Generator[Tuple[int, np.ndarray, np.ndarray, np.ndarray, np.ndarray], None, None]:
        """
        Générateur qui simule et filtre selon un PKF (méthode de Wojciech).
        """
        generator_simul = self._iterate_gen_simul()

        # Première itération
        k, Zkp1 = next(generator_simul)
        xkp1, ykp1 = self._split_xy(Zkp1)
        Xkp1_upd, PXXkp1_upd = self._first_update(ykp1)

        self._record_if_needed(k, xkp1, ykp1, Xkp1_upd, PXXkp1_upd)
        yield k, xkp1, ykp1, Xkp1_upd, PXXkp1_upd

        # Itérations suivantes
        while N is None or k < N:
            yk = ykp1
            Xkp1_pred, Ykp1_pred, PXXkp1_pred, PXYkp1_pred, PYXkp1_pred, PYYkp1_pred = \
                self._predict(Xkp1_upd, yk, PXXkp1_upd)

            k, Zkp1 = next(generator_simul)
            xkp1, ykp1 = self._split_xy(Zkp1)
            Xkp1_upd, PXXkp1_upd = self._update(
                Xkp1_pred, Ykp1_pred, PXXkp1_pred,
                PXYkp1_pred, PYXkp1_pred, PYYkp1_pred, ykp1
            )

            self._record_if_needed(k, xkp1, ykp1, Xkp1_upd, PXXkp1_upd)
            yield k, xkp1, ykp1, Xkp1_upd, PXXkp1_upd

    # ------------------------------------------------------------------
    # Étapes internes du filtre
    # ------------------------------------------------------------------
    def _split_xy(self, Z: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Sépare le vecteur Z en x et y."""
        return Z[:self.param.dim_x], Z[self.param.dim_x:self.param.dim_xy]

    def _first_update(self, ykp1: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Premier filtrage de la séquence."""
        b, Sigma_Y1, Sigma_X1 = self.param.b, self.param.Sigma_Y1, self.param.Sigma_X1
        Xkp1_upd = b.T @ np.linalg.inv(Sigma_Y1) @ ykp1
        PXXkp1_upd = Sigma_X1 - b.T @ np.linalg.inv(Sigma_Y1) @ b
        return Xkp1_upd, PXXkp1_upd

    def _predict(self, Xkp1_upd: np.ndarray, yk: np.ndarray, PXXkp1_upd: np.ndarray):
        """Étape de prédiction."""
        A, mQ = self.param.A, self.param.mQ
        temp1 = np.block([[Xkp1_upd], [yk]])
        temp2 = np.zeros((self.param.dim_xy, self.param.dim_xy))
        temp2[:self.param.dim_x, :self.param.dim_x] = PXXkp1_upd

        Zkp1_pred = A @ temp1
        Pkp1_pred = A @ temp2 @ A.T + mQ

        Xkp1_pred = Zkp1_pred[:self.param.dim_x]
        Ykp1_pred = Zkp1_pred[self.param.dim_x:self.param.dim_xy]

        PXXkp1_pred = Pkp1_pred[:self.param.dim_x, :self.param.dim_x]
        PXYkp1_pred = Pkp1_pred[:self.param.dim_x, self.param.dim_x:self.param.dim_xy]
        PYXkp1_pred = Pkp1_pred[self.param.dim_x:self.param.dim_xy, :self.param.dim_x]
        PYYkp1_pred = Pkp1_pred[self.param.dim_x:self.param.dim_xy, self.param.dim_x:self.param.dim_xy]

        return Xkp1_pred, Ykp1_pred, PXXkp1_pred, PXYkp1_pred, PYXkp1_pred, PYYkp1_pred

    def _update(
        self,
        Xkp1_pred: np.ndarray,
        Ykp1_pred: np.ndarray,
        PXXkp1_pred: np.ndarray,
        PXYkp1_pred: np.ndarray,
        PYXkp1_pred: np.ndarray,
        PYYkp1_pred: np.ndarray,
        ykp1: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Étape de mise à jour."""
        inv_PYY = np.linalg.inv(PYYkp1_pred)
        Xkp1_upd = Xkp1_pred + (PXYkp1_pred @ inv_PYY) @ (ykp1 - Ykp1_pred)
        PXXkp1_upd = PXXkp1_pred - PXYkp1_pred @ inv_PYY @ PYXkp1_pred
        return Xkp1_upd, PXXkp1_upd

    def _record_if_needed(self, k: int, xkp1, ykp1, Xkp1_upd, PXXkp1_upd):
        """Enregistre les données si l’option save_pickle est activée."""
        if self.save_pickle and self._history is not None:
            self._history.record(
                iter=k,
                xkp1=xkp1.copy(),
                ykp1=ykp1.copy(),
                Xkp1_update=Xkp1_upd.copy(),
                PXXkp1_update=PXXkp1_upd.copy()
            )

    # ------------------------------------------------------------------
    # Méthode utilitaire
    # ------------------------------------------------------------------
    def iterate_list(self, generator):
        """Exécute le générateur et renvoie la liste complète des états."""
        return list(generator)


if __name__ == "__main__":
    """
    Exemple d'utilisation du PKF.
    Pour exécuter :
        python -m prg.classes.pkf
    """

    import os
    import numpy as np
    from classes.ParamPKF import ParamPKF

    # ------------------------------------------------------------------
    # Configuration des répertoires de sortie
    # ------------------------------------------------------------------
    base_dir    = os.path.join(".", "dataGenerated")
    tracker_dir = os.path.join(base_dir, "historyTracker")
    graph_dir   = os.path.join(base_dir, "plot")

    os.makedirs(tracker_dir, exist_ok=True)
    os.makedirs(graph_dir, exist_ok=True)

    save_pickle = True
    verbose = 1

    # ------------------------------------------------------------------
    # Paramètres de test (exemple simple)
    # ------------------------------------------------------------------
    dim_x = 1
    dim_y = 1
    dim_xy = dim_x + dim_y

    A = np.array([
        [0.8, 0.1],
        [0.0, 0.9]
    ])

    mQ = np.array([
        [1.0, 0.2],
        [0.2, 1.0]
    ])

    # Création de l’objet ParamPKF
    param = ParamPKF(dim_y=dim_y, dim_x=dim_x, A=A, mQ=mQ, verbose=verbose)

    # ------------------------------------------------------------------
    # Exemple 1 : PKF sans seed (s_key=None)
    # ------------------------------------------------------------------
    pkf = PKF(param, s_key=None, save_pickle=save_pickle, verbose=verbose)
    all_steps = pkf.iterate_list(pkf.iterate_gen_wojciech(N=200))

    if save_pickle and pkf.history is not None:
        df = pkf.history.as_dataframe()
        print("\nHistorique complet :")
        print(df.head())

        # Sauvegarde pickle et graphiques
        pkf.history.save_pickle(os.path.join(tracker_dir, "history_run_pkf.pkl"))
        ax, fig = pkf.history.plot("xkp1", color="blue", show=False, base_dir=graph_dir)
        pkf.history.plot("Xkp1_update", color="green", ax=ax, fig=fig, show=True, base_dir=graph_dir)

    # ------------------------------------------------------------------
    # Exemple 2 : Reprise avec même seed
    # ------------------------------------------------------------------
    input("\nAppuyez sur Entrée pour relancer avec la même seed... ")

    pkf_reloaded = PKF(param, s_key=pkf.seed_gen, save_pickle=save_pickle, verbose=verbose)
    all_steps_reloaded = pkf_reloaded.iterate_list(pkf_reloaded.iterate_gen_wojciech(N=200))

    if save_pickle and pkf_reloaded.history is not None:
        df_reload = pkf_reloaded.history.as_dataframe()
        print("\nHistorique relancé :")
        print(df_reload.head())

        pkf_reloaded.history.save_pickle(os.path.join(tracker_dir, "history_run_pkf_reload.pkl"))
        ax, fig = pkf_reloaded.history.plot("xkp1", color="red", show=False, base_dir=graph_dir)
        pkf_reloaded.history.plot("Xkp1_update", color="black", ax=ax, fig=fig, show=True, base_dir=graph_dir)

