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

class PKF:
    """Implémentation du Pairwise Kalman Filter."""
    def __init__(
        self,
        param: ParamPKF,
        sKey: Optional[int] = None,
        save_pickle: bool = False,
        verbose: int = 0
    ):
        if not isinstance(param, ParamPKF):
            raise TypeError("param doit être un objet de la classe ParamPKF")
        if not ((isinstance(sKey, int) and sKey > 0) or sKey is None):
            raise ValueError("sKey doit être un entier strictement positif")
        if verbose not in [0, 1, 2]:
            raise ValueError("verbose doit être 0, 1 ou 2")
        if not isinstance(save_pickle, bool):
            raise TypeError("save_pickle doit être un booléen")

        self.param     = param
        self.verbose   = verbose
        self._seed_gen = SeedGenerator(sKey)

        # Crée HistoryTracker uniquement si save_pickle est True
        self.save_pickle = save_pickle
        self._history = HistoryTracker() if save_pickle else None

        if self.verbose >= 1:
            print(f"[PKF] Initialisé avec sKey={sKey}, verbose={verbose}, save_pickle={save_pickle}")

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
        
        # Initialisation
        k          = 0
        Zkp1_simul = self._seed_gen.rng.multivariate_normal(mean=self.param.mu0, cov=self.param.Q1).reshape(-1,1)
        yield k, Zkp1_simul.copy()

        # la suite...
        A = self.param.A
        while N is None or k < N:
             # preparation tour suivant
            Zk_simul   = Zkp1_simul
            k         += 1
            Zkp1_simul = A @ Zk_simul + self._seed_gen.rng.multivariate_normal(mean=self.param.mu0, cov=self.param.mQ).reshape(-1,1)
            yield k, Zkp1_simul.copy()

    # ------------------------------------------------------------------
    # PKF : Méthode Classique
    # ------------------------------------------------------------------
    def process_pkf_classique(self, N: Optional[int] = None) -> Generator[Tuple[int, np.ndarray, np.ndarray, np.ndarray, np.ndarray], None, None]:
        """
        Générateur qui simule et filtre selon un PKF (méthode classique).
        """
        generatorSimul = self._iterate_gen_simul()
        A, mQ          = self.param.A, self.param.mQ
        dimx  = self.param.dim_x
        dimxy = self.param.dim_xy
        
        ###################
        # Le premier
        ###################
        
        # On génère une échantillon
        k, Zkp1_simul = next(generatorSimul)
        xkp1          = Zkp1_simul[0:dimx]
        ykp1          = Zkp1_simul[dimx:dimxy]
        
        # on a le filtrage du premier échantillon
        Xkp1_update   = self.param.b.T @ np.linalg.inv(self.param.Sigma_Y1) @ ykp1
        PXXkp1_update = self.param.Sigma_X1 - self.param.b.T @ np.linalg.inv(self.param.Sigma_Y1) @ self.param.b

        # Enregistrement uniquement si save_pickle=True
        if self.save_pickle and self._history is not None:
            self._history.record(iter=k, xkp1=xkp1.copy(), ykp1=ykp1.copy(), Xkp1_update=Xkp1_update.copy(), PXXkp1_update=PXXkp1_update.copy())
        
        yield k, xkp1, ykp1, Xkp1_update, PXXkp1_update

        ###################
        # Les suivants
        ###################
        
        

    # ------------------------------------------------------------------
    # PKF : Méthode Wojciech
    # ------------------------------------------------------------------
    def process_pkf_wojciech(self, N: Optional[int] = None) -> Generator[Tuple[int, np.ndarray, np.ndarray, np.ndarray, np.ndarray], None, None]:
        """
        Générateur qui simule et filtre selon un PKF (méthode de Wojciech).
        """
        generatorSimul = self._iterate_gen_simul()
        A, mQ          = self.param.A, self.param.mQ
        dimx  = self.param.dim_x
        dimxy = self.param.dim_xy
        
        ###################
        # Le premier
        ###################
        
        # On génère une échantillon
        k, Zkp1_simul = next(generatorSimul)
        xkp1          = Zkp1_simul[0:dimx]
        ykp1          = Zkp1_simul[dimx:dimxy]
        
        # on a le filtrage du premier échantillon
        Xkp1_update   = self.param.b.T @ np.linalg.inv(self.param.Sigma_Y1) @ ykp1
        PXXkp1_update = self.param.Sigma_X1 - self.param.b.T @ np.linalg.inv(self.param.Sigma_Y1) @ self.param.b

        # Enregistrement uniquement si save_pickle=True
        if self.save_pickle and self._history is not None:
            self._history.record(iter=k, xkp1=xkp1.copy(), ykp1=ykp1.copy(), Xkp1_update=Xkp1_update.copy(), PXXkp1_update=PXXkp1_update.copy())
        
        yield k, xkp1, ykp1, Xkp1_update, PXXkp1_update

        ###################
        # Les suivants
        ###################
        
        temp2 = np.zeros((dimxy, dimxy))
        while N is None or k < N:
            
            # preparation tour suivant
            yk = ykp1
            
            #######################################
            # Prédiction
            #######################################
            temp1 = np.vstack((Xkp1_update, yk))
            temp2[0:dimx, 0:dimx] = PXXkp1_update
    
            # on prédit
            Zkp1_predict   = A @ temp1
            Xkp1_predict   = Zkp1_predict[0:dimx]
            Ykp1_predict   = Zkp1_predict[dimx:dimxy]
            
            Pkp1_predict   = A @ temp2 @ A.T+mQ
            PXXkp1_predict = Pkp1_predict[0:dimx,     0:dimx]
            PXYkp1_predict = Pkp1_predict[0:dimx,     dimx:dimxy]
            PYXkp1_predict = Pkp1_predict[dimx:dimxy, 0:dimx]
            PYYkp1_predict = Pkp1_predict[dimx:dimxy, dimx:dimxy]

            #######################################
            # MAJ avec la nouvelle observation
            #######################################
            # On récupère un nouvel échantillon pour la mise à jour
            k, Zkp1_simul = next(generatorSimul)
            xkp1          = Zkp1_simul[0:dimx]
            ykp1          = Zkp1_simul[dimx:dimxy]
            
            # on update
            Xkp1_update   = Xkp1_predict + (PXYkp1_predict @ np.linalg.inv(PYYkp1_predict)) @ (ykp1 - Ykp1_predict)
            PXXkp1_update = PXXkp1_predict - PXYkp1_predict @ np.linalg.inv(PYYkp1_predict) @ PYXkp1_predict
            
            # Enregistrement uniquement si save_pickle=True
            if self.save_pickle and self._history is not None:
                self._history.record(iter=k, xkp1=xkp1.copy(), ykp1=ykp1.copy(), Xkp1_update=Xkp1_update.copy(), PXXkp1_update=PXXkp1_update.copy())
            
            yield k, xkp1, ykp1, Xkp1_update, PXXkp1_update


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
        python prg/PKF.py
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
    verbose     = 1
    N           = 50

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
    
    # dim_x, dim_y = 2, 2
    # A = np.array([[5, 2, 1, 0],
    #               [3, 8, 0, 2],
    #               [2, 2, 10, 6],
    #               [1, 1, 5, 9]], float)

    # mQ = np.array([[1.0, 0.5, 0.1, 0.2],
    #                [0.5, 1.0, 0.1, 0.1],
    #                [0.1, 0.1, 1.0, 0.5],
    #                [0.2, 0.1, 0.5, 1.0]], float)

    param = ParamPKF(dim_y=dim_y, dim_x=dim_x, A=A, mQ=mQ, verbose=2)
    param.summary()

    # Création de l’objet ParamPKF
    param = ParamPKF(dim_y=dim_y, dim_x=dim_x, A=A, mQ=mQ, verbose=verbose)

    # ------------------------------------------------------------------
    # Exemple 1 : PKF sans seed (sKey=None)
    # ------------------------------------------------------------------
    pkf = PKF(param, sKey=None, save_pickle=save_pickle, verbose=verbose)
    all_steps = pkf.iterate_list(pkf.process_pkf_wojciech(N=N))

    if save_pickle and pkf.history is not None:
        df = pkf.history.as_dataframe()
        print("\nHistorique complet :")
        print(df.head())
        # print(df.info())

        # Sauvegarde pickle et graphiques
        pkf.history.save_pickle(os.path.join(tracker_dir, "history_run_pkf.pkl"))
        ax, fig = pkf.history.plot("xkp1", color="blue", show=False, base_dir=graph_dir)
        pkf.history.plot("Xkp1_update", color="green", ax=ax, fig=fig, show=False, base_dir=graph_dir)

    # ------------------------------------------------------------------
    # Exemple 2 : Reprise avec même seed
    # ------------------------------------------------------------------
    input("\nAppuyez sur Entrée pour relancer avec la même seed, mais méthode classique... ")

    pkf_reloaded = PKF(param, sKey=pkf.seed_gen, save_pickle=save_pickle, verbose=verbose)
    all_steps_reloaded = pkf_reloaded.iterate_list(pkf_reloaded.process_pkf_classique(N=N))

    if save_pickle and pkf.history is not None:
        df_reloaded = pkf_reloaded.history.as_dataframe()
        print("\nHistorique complet :")
        print(df.head())
        # print(df_reloaded.info())
        # input('attente')

        # Sauvegarde pickle et graphiques
        pkf_reloaded.history.save_pickle(os.path.join(tracker_dir, "history_run_pkf_reloaded.pkl"))
        ax, fig = pkf_reloaded.history.plot("xkp1", color="blue", show=False, base_dir=graph_dir)
        pkf_reloaded.history.plot("Xkp1_update", color="green", ax=ax, fig=fig, show=False, base_dir=graph_dir)

