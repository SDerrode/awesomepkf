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

#!/usr/bin/env python3
# -*- coding: utf-8 -*-


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

        self.param       = param
        self.verbose     = verbose
        self._seed_gen   = SeedGenerator(sKey)

        # Crée HistoryTracker uniquement si save_pickle est True
        self.save_pickle = save_pickle
        self._history = HistoryTracker() if save_pickle else None

        if self.verbose >= 1:
            print(f"[PKF] Initialisé avec sKey={sKey}, verbose={verbose}, save_pickle={save_pickle}")

    @property
    def history(self):
        """Retourne l'objet HistoryTracker si save_pickle=True, sinon None."""
        return self._history

    def iterate_gen(self, n=None):
        """
        Générateur qui simule Z_{k+1} = A * Z_k + W_{k+1}, 
        avec W_{k+1} ~ N(0, mQ) selon ParamPKF.
        Enregistre Zk_simul dans history uniquement si save_pickle=True.
        """
        
        # Initialisation
        k = 0
        Zk_simul = np.zeros(self.param.n)
        if self.save_pickle and self._history is not None:
            self._history.record(iter=k, Zk_simul=Zk_simul.copy())
            
        yield {"iter": k, "Zk_simul": Zk_simul.copy()}
        
        # la uite...
        while n is None or k < n:
            W = self._seed_gen.rng.multivariate_normal(
                mean=np.zeros(self.param.n), cov=self.param.mQ
            )
            Zkp1_simul = self.param.A @ Zk_simul + W

            # Enregistrement uniquement si save_pickle=True
            if self.save_pickle and self._history is not None:
                self._history.record(iter=k, Zk_simul=Zkp1_simul.copy())

            yield {"iter": k, "Zk_simul": Zkp1_simul.copy()}

            Zk_simul = Zkp1_simul
            k += 1

    # ------------------------------------------------------------------
    # Méthode pour obtenir la liste complète des itérations
    # ------------------------------------------------------------------
    def iterate_list(self, n):
        """
        Lance le générateur pour obtenir la liste complète des états.
        """
        return list(self.iterate_gen(n))


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
    # Définition des dimensions
    # ------------------------------------------------------
    dim_x = 2  # dimension des états 
    dim_y = 2  # dimension des observations
    n = dim_x + dim_y

    # ------------------------------------------------------
    # Création de matrices A et mQ
    # ------------------------------------------------------

    A = np.array([[0.5, 0.0, 0.1, 0.0],
                    [0.0, 0.8, 0.0, 0.2],
                    [0.1, 0.2, 1.0, 0.0],
                    [0.0, 0.1, 0.0, 0.9]])

    a, b, c, d, e = 0.1, 0.3, 0.1, 0.2, 0.1
    mQ = np.array([[1.0, b,   a,   d]  ,
                    [b,   1.0, e,   c  ],
                    [a,   e,   1.0, b  ],
                    [d,   c,   b,   1.0]])

    param = ParamPKF(dim_y=dim_y, dim_x=dim_x, A=A, mQ=mQ, verbose=verbose)


    # ------------------------------------------------------
    # Création d'un PKF
    # ------------------------------------------------------
    
    pkf1 = PKF(param, sKey=123, save_pickle=save_pickle, verbose=verbose)
    print("\nItérations individuelles:")
    for step in pkf1.iterate_gen(5):
        print(step)

    # ------------------------------------------------------
    # Création d'un autre PKF
    # ------------------------------------------------------
    pkf2 = PKF(param, sKey=123, save_pickle=save_pickle, verbose=verbose)
    all_steps = pkf2.iterate_list(10)
    print("\nListe complète des états:")
    print(all_steps)
    
    # Affichage de l'historique sous forme de DataFrame
    if save_pickle:
        print("\nHistorique complet:")
        print(pkf2.history.as_dataframe())

        # # Affiche le graphique
        # pkf2.history.plot("Zk_simul", color="blue")

        # # Sauvegarde le graphique sans l'afficher
        # graph_dir = os.path.join('.', 'dataGenerated', 'plot')
        # os.makedirs(graph_dir, exist_ok=True)
        # b.history.plot("diff", color="red", marker="s", linestyle="-", show=False, base_dir=graph_dir)

        # Sauvegarde l'historique en pickle
        tracker_dir = os.path.join('.', 'dataGenerated', 'historyTracker')
        os.makedirs(tracker_dir, exist_ok=True)
        pkf1.history.save_pickle(os.path.join('.', 'dataGenerated', 'historyTracker', 'history_run_a.pkl'))
        pkf2.history.save_pickle(os.path.join('.', 'dataGenerated', 'historyTracker', 'history_run_b.pkl'))
