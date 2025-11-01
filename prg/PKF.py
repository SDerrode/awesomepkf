#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, sys, path
directory = path.Path(__file__)
sys.path.append(directory.parent.parent)

import math

import numpy  as np
import scipy  as sc
import pandas as pd

from prg.classes.ParamPKF       import ParamPKF
from prg.classes.HistoryTracker import HistoryTracker
from prg.classes.SeedGenerator  import SeedGenerator

class PKF:
    def __init__(self, param, sKey: int = 42, save_pickle: bool = False, verbose: int = 0):
        if not isinstance(param, ParamPKF):
            raise TypeError("param doit être un objet de la classe ParamPKF")
        if not isinstance(sKey, int) or sKey <= 0:
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

    @property
    def history(self):
        """Retourne l'objet HistoryTracker si save_pickle=True, sinon None."""
        return self._history

    def _iterate_gen_simul(self, N=None):
        """
        Générateur qui simule Z_{k+1} = A * Z_k + W_{k+1}, 
        avec W_{k+1} ~ N(0, mQ) selon ParamPKF.
        Enregistre Zk_simul dans history uniquement si save_pickle=True.
        """
        
        # Initialisation
        k          = 0
        Zkp1_simul = self._seed_gen.rng.multivariate_normal(mean=self.param.mu0, cov=self.param.Q1)
        yield k, Zkp1_simul.copy()

        # la suite...
        A = self.param.A
        while N is None or k < N:
             # preparation tour suivant
            Zk_simul = Zkp1_simul
            k       += 1
            Zkp1_simul = A @ Zk_simul + self._seed_gen.rng.multivariate_normal(mean=self.param.mu0, cov=self.param.mQ)
            yield k, Zkp1_simul.copy()


    def iterate_gen_Wojciech(self, N=None):
        """
        Générateur qui simule et filtre selon un PKF
        Méthode Wojciech
        """
        generatorSimul     = self._iterate_gen_simul()
        
        ###################
        # Le premier
        ###################
        
        # On génère une échantillon
        k, Zkp1_simul = next(generatorSimul)
        xkp1          = Zkp1_simul[0:self.param.dim_x]
        ykp1          = Zkp1_simul[self.param.dim_x:self.param.dim_xy]
        
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
        
        temp2 = np.block([
                    [np.zeros(shape=(self.param.dim_x, self.param.dim_x)), np.zeros(shape=(self.param.dim_x, self.param.dim_y))],
                    [np.zeros(shape=(self.param.dim_y, self.param.dim_x)), np.zeros(shape=(self.param.dim_y, self.param.dim_y))]
                ])
        
        while N is None or k < N:
            
            # preparation tour suivant
            yk = ykp1
            
            #######################################
            # Prédiction
            #######################################
            temp1 = np.block([
                [Xkp1_update],
                [yk]
            ])
            temp2[0:self.param.dim_x, 0:self.param.dim_x] = PXXkp1_update
            
             # on prédit
            Zkp1_predict   = self.param.A @ temp1
            Xkp1_predict   = Zkp1_predict[0:self.param.dim_x]
            Ykp1_predict   = Zkp1_predict[self.param.dim_x:self.param.dim_xy]
            Pkp1_predict   = self.param.A @ temp2 @ self.param.A.T+self.param.mQ
            PXXkp1_predict = Pkp1_predict[0:self.param.dim_x,                 0:self.param.dim_x]
            PXYkp1_predict = Pkp1_predict[0:self.param.dim_x,                 self.param.dim_x:self.param.dim_xy]
            PYXkp1_predict = Pkp1_predict[self.param.dim_x:self.param.dim_xy, 0:self.param.dim_x]
            PYYkp1_predict = Pkp1_predict[self.param.dim_x:self.param.dim_xy, self.param.dim_x:self.param.dim_xy]
           
            #######################################
            # MAJ avec la nouvelle observation
            #######################################
            # On récupère un nouvel échantillon pour la mise à jour
            k, Zkp1_simul = next(generatorSimul)
            xkp1 = Zkp1_simul[0:self.param.dim_x]
            ykp1 = Zkp1_simul[self.param.dim_x:self.param.dim_xy]
            
            # on update
            Xkp1_update   = Xkp1_predict + (PXYkp1_predict @ np.linalg.inv(PYYkp1_predict)) @ (ykp1 - Ykp1_predict)
            PXXkp1_update = PXXkp1_predict - PXYkp1_predict @ np.linalg.inv(PYYkp1_predict) @ PYXkp1_predict
            
            # Enregistrement uniquement si save_pickle=True
            if self.save_pickle and self._history is not None:
                self._history.record(iter=k, xkp1=xkp1.copy(), ykp1=ykp1.copy(), Xkp1_update=Xkp1_update.copy(), PXXkp1_update=PXXkp1_update.copy())
            
            yield k, xkp1, ykp1, Xkp1_update, PXXkp1_update


    # ------------------------------------------------------------------
    # Méthode pour obtenir la liste complète des itérations
    # ------------------------------------------------------------------
    def iterate_list(self, N):
        """
        Lance le générateur pour obtenir la liste complète des états.
        """
        return list(self.iterate_gen_Wojciech(N))


# ----------------------------------------------------------------------
# Exemple d'utilisation
# ----------------------------------------------------------------------
if __name__ == '__main__':
    """
    python prg/classes/PKF.py
    """
    
    save_pickle = True
    verbose     = 0

    # ------------------------------------------------------
    # Exemples de jeux de paramètres
    # ------------------------------------------------------
    
    # exemple #1
    # dim_x  = 2  # dimension des états
    # dim_y  = 2  # dimension des observations
    # dim_xy = dim_x + dim_y

    # A = np.array([[5, 2, 1, 0],
    #               [3, 8, 0, 2],
    #               [2, 2, 10, 6],
    #               [1, 1, 5, 9]], dtype=float)

    # a, b, c, d, e = 0.1, 0.5, 0.1, 0.2, 0.1
    # mQ = np.array([[1.0,  b,   a,   d]  ,
    #                 [b,   1.0, e,   c  ],
    #                 [a,   e,   1.0, b  ],
    #                 [d,   c,   b,   1.0]]) *3.
    
    # exemple #2
    dim_x  = 1  # dimension des états
    dim_y  = 1  # dimension des observations
    dim_xy = dim_x + dim_y

    A = np.array([[0.8, 0.1],
                [0.0, 0.9]])
    mQ = np.array([[1.0, 0.2],
                [0.2, 1.0]])

    # ------------------------------------------------------
    # Création de l'objet ParamPKF
    # ------------------------------------------------------
    param = ParamPKF(dim_y=dim_y, dim_x=dim_x, A=A, mQ=mQ, verbose=verbose)

    # ------------------------------------------------------
    # Création d'un PKF
    # ------------------------------------------------------
    # pkf1 = PKF(param, sKey=123, save_pickle=save_pickle, verbose=verbose)
    # print("\nItérations individuelles:")
    # for step in pkf1.iterate_gen(5):
    #     print(step)

    # ------------------------------------------------------
    # Création d'un autre PKF
    # ------------------------------------------------------
    pkf2 = PKF(param, sKey=123, save_pickle=save_pickle, verbose=verbose)
    all_steps = pkf2.iterate_list_wojciech(200)
    # print(f"\nListe complète des états:\n{all_steps}")
    
    # Affichage de l'historique sous forme de DataFrame
    if save_pickle:
        # Affiche le graphique
        graph_dir = os.path.join('.', 'dataGenerated', 'plot')
        os.makedirs(graph_dir, exist_ok=True)
        ax, fig = pkf2.history.plot("xkp1", color="blue", show=False, base_dir=graph_dir)
        pkf2.history.plot("Xkp1_update", color="green", ax=ax, fig=fig, show=False, base_dir=graph_dir)

        # Sauvegarde l'historique en pickle
        print(f"\nHistorique complet: {pkf2.history.as_dataframe()}")
        tracker_dir = os.path.join('.', 'dataGenerated', 'historyTracker')
        os.makedirs(tracker_dir, exist_ok=True)
        # pkf1.history.save_pickle(os.path.join('.', 'dataGenerated', 'historyTracker', 'history_run_pkf1.pkl'))
        pkf2.history.save_pickle(os.path.join('.', 'dataGenerated', 'historyTracker', 'history_run_pkf2.pkl'))
