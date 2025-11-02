#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Module PKF #########################################################
####################################################################
Implémente un filtre de Kalman couple (PKF) 
  selon la formulation mathématique (Wojciech), ou
  selon la formulation phsique (classique, avec expression du gain),
  avec enregistrement optionnel.
Un exemple d'usage est donné dans le programme principal ci-dessous,
qui compare les 2 implémentations (mêmes résultats attendus).
####################################################################
"""

from __future__ import annotations

import os
import math
import logging
import warnings
from typing import Generator, Optional, Tuple

import numpy as np
import pandas as pd
import scipy as sc

from classes.ParamPKF import ParamPKF  # Manage parameters for the PKF
# Keep trace of exécution (all parameters at all iteration)
from classes.HistoryTracker import HistoryTracker
# To manage the seed for random generation
from classes.SeedGenerator import SeedGenerator


class PKF:
    """Implementation of PKF."""

    def __init__(
        self,
        param: ParamPKF,
        sKey: Optional[int] = None,
        save_pickle: bool = False,
        verbose: int = 0):
        
        if not isinstance(param, ParamPKF):
            raise TypeError("param msut be an object from class ParamPKF")
        if not ((isinstance(sKey, int) and sKey > 0) or sKey is None):
            raise ValueError("sKey must be None or a number >0")
        if verbose not in [0, 1, 2]:
            raise ValueError("verbose must be 0, 1 or 2")
        if not isinstance(save_pickle, bool):
            raise TypeError("save_pickle must be a boolean")

        self.param = param
        self.verbose = verbose
        self._seed_gen = SeedGenerator(sKey)

        # Create HistoryTracker only if save_pickle is True
        self.save_pickle = save_pickle
        self._history = HistoryTracker() if save_pickle else None

        if self.verbose >= 1:
            print(f"[PKF] Init with sKey={sKey}, verbose={verbose}, save_pickle={save_pickle}.")


    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------
    @property
    def seed_gen(self) -> int:
        """Return generator seed."""
        return self._seed_gen.seed
    @property
    def history(self) -> Optional[HistoryTracker]:
        """Return HistoryTracker object if save_pickle==True, else None."""
        return self._history

    # ------------------------------------------------------------------
    # Generators
    # ------------------------------------------------------------------
    def _data_generation(
            self, 
            N: Optional[int] = None
        ) -> Generator[Tuple[int, np.ndarray], None, None]:
        """
        Genrator for the simulation of Z_{k+1} = A * Z_k + W_{k+1}, 
        with W_{k+1} ~ N(0, mQ) and Z_1 ~ N(0, Q1).
        This generator can be replaced by a some data acquired in real-time.
        """
        # Short-cuts
        mu0, A, mQ, dimx = self.param.mu0, self.param.A, self.param.mQ, self.param.dim_x

        # The first
        k = 0
        Zkp1_simul = self._seed_gen.rng.multivariate_normal(mean=mu0, cov=self.param.Q1).reshape(-1,1)
        yield k, np.split(Zkp1_simul, [dimx])

        # The next...
        while N is None or k < N:
            k += 1
            Zkp1_simul = A @ Zkp1_simul + \
                self._seed_gen.rng.multivariate_normal(mean=mu0, cov=mQ).reshape(-1,1)
            yield k, np.split(Zkp1_simul, [dimx])

    # ------------------------------------------------------------------
    # PKF : Mathematic and Physicist formulations
    # ------------------------------------------------------------------

    def process_pkf(self, N=None):
        """
        Generator of PKF filter (mathematic and physicist formulations).
        It makes use of data generator called _data_generation().
        """
        
        if not ((isinstance(N, int) and N > 0) or N is None):
            raise ValueError("sKey must be None or a number >0")

        # Short-cuts
        A, mQ             = self.param.A, self.param.mQ
        dimx, dimy, dimxy = self.param.dim_x, self.param.dim_y, self.param.dim_xy

        # data simulation generator
        generatorSimul = self._data_generation()

        # The first
        ###################
       
        # First generated data sample
        k, (xkp1, ykp1) = next(generatorSimul) # les parenthèses servent à déballer la liste de 2 élements

        # Filtering of the first sample
        Xkp1_update = self.param.b.T @ np.linalg.inv(self.param.Sigma_Y1) @ ykp1
        PXXkp1_update = self.param.Sigma_X1 - \
            self.param.b.T @ np.linalg.inv(self.param.Sigma_Y1) @ self.param.b

        # Store if save_pickle==True
        if self.save_pickle and self._history is not None:
            self._history.record(iter=k,
                                    xkp1                 = xkp1.copy(),
                                    ykp1                 = ykp1.copy(),
                                    Xkp1_update_math     = Xkp1_update.copy(),
                                    PXXkp1_update_math   = PXXkp1_update.copy(),
                                    ikp1                 = np.zeros(shape=(dimy, 1)),
                                    Skp1                 = np.zeros(shape=(dimy, dimy)),
                                    Kkp1                 = np.zeros(shape=(dimx, dimy)),
                                    Xkp1_update_phys     = Xkp1_update.copy(),
                                    PXXkp1_update_phys   = PXXkp1_update.copy(),
                                    PXXkp1_update_Joseph = PXXkp1_update.copy())
        
        yield xkp1, ykp1, Xkp1_update, Xkp1_update

        ###################
        # The next

        temp2 = np.zeros(shape=(dimxy, dimxy))
        while N is None or k < N:
            
            # nécessaire pour la forme de Joseph
            PXXk_update = PXXkp1_update.copy()

            #######################################
            # Prediction
            #######################################
            temp1 = np.vstack((Xkp1_update, ykp1)) # here ykp1 still gives the previous : it is yk indeed!
            temp2[0:dimx, 0:dimx] = PXXkp1_update

            # Prediction
            Xkp1_predict, Ykp1_predict = np.split(A @ temp1, [dimx]) # Zkp1_predict = A @ temp1
            Pkp1_predict = A @ temp2 @ A.T + mQ
            # Cutting into 4 blocks
            M_top, M_bottom = np.vsplit(Pkp1_predict, [dimx])
            PXXkp1_predict, PXYkp1_predict = np.hsplit(M_top, [dimx])
            PYXkp1_predict, PYYkp1_predict = np.hsplit(M_bottom, [dimx])

            #######################################
            # Update with a new observation
            #######################################
            
            # Get new obervation from the data generator
            k, (xkp1, ykp1) = next(generatorSimul) # parenthesis is used to flatten the list of two elements

            # Updating with mathematical formulation
            ###############################################
            Xkp1_update = Xkp1_predict + \
                (PXYkp1_predict @ np.linalg.inv(PYYkp1_predict)) @ (ykp1 - Ykp1_predict)
            PXXkp1_update = PXXkp1_predict - \
                PXYkp1_predict @ np.linalg.inv(PYYkp1_predict) @ PYXkp1_predict
            # print(f'\nMATH : Xkp1_update={Xkp1_update}\nPXXkp1_update={PXXkp1_update}')
            
            Xkp1_update_math   = Xkp1_update.copy()
            PXXkp1_update_math = PXXkp1_update.copy()
            
            # Updating with physical formulation
            ###############################################
            # innovation (expectation and variance)
            ikp1 = ykp1 - Ykp1_predict
            Skp1 = PYYkp1_predict
            # Kalman gain 
            Kkp1  = PXYkp1_predict @ np.linalg.inv(Skp1)
            # Updating expectation and variance and variance in Joseph form
            Xkp1_update = Xkp1_predict + Kkp1 @ ikp1
            PXXkp1_update = PXXkp1_predict - Kkp1 @ PYXkp1_predict
            # Dans la forme de Joseph, j'utilise les sous matrices de mQ, qui ne sont pas des matrices mais des ActiveView.
            # pour revenir à un forme np.ndarray, j'utulise l'opérateur value (méthodes définie dans la classe ActiveView)
            PXXkp1_update_Joseph =  (self.param.A_xx.value - Kkp1 @ self.param.A_yx.value) @ PXXk_update @ (self.param.A_xx.value - Kkp1 @ self.param.A_yx.value).T\
                + self.param.Q_xx.value - Kkp1 @ self.param.Q_yx.value - self.param.Q_xy.value @ Kkp1.T + Kkp1 @ self.param.Q_yy.value @ Kkp1.T
            if not np.allclose(PXXkp1_update, PXXkp1_update_Joseph, rtol=1e-3, atol=1e-5):
                warnings.warn("Les matrices PXXkp1_update et PXXkp1_update_Joseph sont différentes au-delà de la tolérance spécifiée !")
                print(f'\nPXXkp1_update = {PXXkp1_update}')
                print(f'\nPXXkp1_update_Joseph = {PXXkp1_update_Joseph}')
                input('attente')

            Xkp1_update_phys          = Xkp1_update.copy()
            PXXkp1_update_phys        = PXXkp1_update.copy()
            PXXkp1_update_Joseph_phys = PXXkp1_update_Joseph.copy()

            # Store if save_pickle==True
            if self.save_pickle and self._history is not None:
                self._history.record(iter=k,
                                     xkp1                 = xkp1.copy(),
                                     ykp1                 = ykp1.copy(),
                                     Xkp1_update_math     = Xkp1_update_math,
                                     PXXkp1_update_math   = PXXkp1_update_math,
                                     ikp1                 = ikp1.copy(),
                                     Skp1                 = Skp1.copy(),
                                     Kkp1                 = Kkp1.copy(),
                                     Xkp1_update_phys     = Xkp1_update_phys,
                                     PXXkp1_update_phys   = PXXkp1_update_phys,
                                     PXXkp1_update_Joseph = PXXkp1_update_Joseph)

            yield xkp1, ykp1, Xkp1_update_math, Xkp1_update_phys

    def process_N_data(self, N):
        return list(self.process_pkf(N=N))

if __name__ == "__main__":
    """
    Exemple d'utilisation du PKF.
    Pour exécuter :
        python prg/PKF.py
    """
    # ------------------------------------------------------------------
    # Constants
    # ------------------------------------------------------------------
    save_pickle = True
    verbose     = 1
    N           = 200

    # ------------------------------------------------------------------
    # Test parameters
    # ------------------------------------------------------------------
    dim_x, dim_y = 1, 1
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



    # ------------------------------------------------------------------
    # Output repo for data, traces and plots
    # ------------------------------------------------------------------
    base_dir = os.path.join(".",         "dataGenerated")
    tracker_dir = os.path.join(base_dir, "historyTracker")
    graph_dir = os.path.join(base_dir,   "plot")
    os.makedirs(tracker_dir, exist_ok=True)
    os.makedirs(graph_dir,   exist_ok=True)

    # ------------------------------------------------------------------
    # Let's go.....
    # ------------------------------------------------------------------

    # PKF parameters object manager
    param = ParamPKF(dim_y=dim_y, dim_x=dim_x, A=A, mQ=mQ, verbose=verbose)
    if verbose > 0:
        param.summary()

    print("\nPKF filtering with the Mathematic formulation of PKF... ")
    sKey = 41
    pkf = PKF(param, sKey=sKey, save_pickle=save_pickle, verbose=verbose)
    listePKF = pkf.process_N_data(N=N)
    # print(f'listePKF={listePKF}')

    if save_pickle and pkf.history is not None:
        df = pkf.history.as_dataframe()
        if verbose > 0:
            print("\nHistorique complet :")
            print(df.head())
            # print(df.info())

        # pickle storing and plots
        pkf.history.save_pickle(os.path.join(tracker_dir, "history_run_pkf.pkl"))
        ax = pkf.history.plot( ["xkp1", "Xkp1_update_math","Xkp1_update_phys"], \
                         label=["X - Ground Truth", "X - Filtered (mathematical version)", "X - Filtered (physical version)"], \
                         show=False, base_dir=graph_dir)


    # input("\nEnter to re-run with the same seed, but Physical formulation... ")
    # pkf_reloaded = PKF(param, sKey=pkf.seed_gen, save_pickle=save_pickle, verbose=verbose)
    # pkf_reloaded.process_N_data(N=N)

    # if save_pickle and pkf.history is not None:
    #     df_reloaded = pkf_reloaded.history.as_dataframe()
    #     if verbose > 0:
    #         print("\nHistorique complet :")
    #         print(df_reloaded.head())
    #         # print(df_reloaded.info())

    #     # pickle storing and plots
    #     pkf_reloaded.history.save_pickle(os.path.join(
    #         tracker_dir, "history_run_pkf_reloaded.pkl"))
    #     ax, fig = pkf_reloaded.history.plot(
    #         "xkp1",        color="blue",  show=False, base_dir=graph_dir)
    #     pkf_reloaded.history.plot("Xkp1_update_physi", color="green", show=False, base_dir=graph_dir, ax=ax, fig=fig)
