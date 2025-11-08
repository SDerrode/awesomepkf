#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Module UPKF #########################################################
####################################################################
Implémente un filtre de Kalman couple Unscented (UPKF) 
####################################################################
"""

from __future__ import annotations

import os
import math
import logging
import warnings
from typing import Generator, Optional, Tuple

import numpy as np

# A few utils functions that are used several times
from others.Utils import rmse, file_data_generator
# Manage parameters for the UPKF
from classes.ParamUPKF import ParamUPKF
# Keep trace of execution (all parameters at all iterations)
from classes.HistoryTracker import HistoryTracker
# To manage the seed for random generation
from classes.SeedGenerator import SeedGenerator

# ----------------------------------------------------------------------
# Configuration du logging global
# ----------------------------------------------------------------------
logging.basicConfig(format="[%(levelname)s] %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)


class UPKF:
    """Implementation of UPKF."""

    def __init__(
        self,
        param: ParamUPKF,
        sKey: Optional[int] = None,
        save_pickle: bool = False,
        verbose: int = 0):
        
        if not isinstance(param, ParamUPKF):
            raise TypeError("param must be an object from class ParamUPKF")
        if not ((isinstance(sKey, int) and sKey > 0) or sKey is None):
            raise ValueError("sKey must be None or a number>0")
        if not isinstance(save_pickle, bool):
            raise TypeError("save_pickle must be a boolean")
        if verbose not in [0, 1, 2]:
            raise ValueError("verbose must be 0, 1 or 2")

        self.param     = param
        self.verbose   = verbose
        self._seed_gen = SeedGenerator(sKey)

        # Create HistoryTracker only if save_pickle is True
        self.save_pickle = save_pickle
        self._history    = HistoryTracker() if save_pickle else None
        
        # Configuration du logger selon verbose
        self._set_log_level()

        if self.verbose >= 1:
            logger.info(f"[UPKF] Init with sKey={sKey}, verbose={verbose}, save_pickle={save_pickle}")


    # ------------------------------------------------------------------
    # Gestion du logging selon le niveau de verbosité
    # ------------------------------------------------------------------
    def _set_log_level(self):
        if self.verbose == 0:
            logger.setLevel(logging.WARNING)
        elif self.verbose == 1:
            logger.setLevel(logging.INFO)
        else:
            logger.setLevel(logging.DEBUG)

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
    def _data_generation(self, N: Optional[int] = None) -> Generator[Tuple[int, np.ndarray], None, None]:
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
    # Vérification de cohérence
    # ------------------------------------------------------------------
    def _check_consistency(self, **kwargs):
        """Check the consistency of the matrices (Positive Semi-Definite)."""
        
        tol = 1e-12
        for name, M in kwargs.items():
            if not np.allclose(M, M.T, atol=tol):
                logger.warning(f"⚠️ {name} matrix is not symmetrical")
            eigvals = np.linalg.eigvals(M)
            if np.any(eigvals < -tol):
                logger.warning(f"⚠️ {name} matrix is not positive semi-definite (min eig = {eigvals.min():.3e})")
            logger.debug(f"Eig of {name} matrix: {eigvals}")

    def _check_equality(self, **kwargs):
        """
        Check that all matrices passed in **kwargs are pairwise identical.
        """

        # Aucun argument fourni → on avertit
        if len(kwargs) < 2:
            logger.warning("⚠️ _check_equality : nee 2 matrices at least.")
            return

        # Liste des noms et matrices
        names = list(kwargs.keys())
        matrices = list(kwargs.values())

        # Vérifie les dimensions d'abord
        shapes = [m.shape for m in matrices]
        if len(set(shapes)) != 1:
            logger.warning(f"⚠️ Matrices are not identically shaped ! {shapes}")
            return

        # Vérification 2 à 2
        ref       = matrices[0]
        ref_name  = names[0]
        tol       = 1e-10
        all_equal = True

        for name, M in zip(names[1:], matrices[1:]):
            if not np.allclose(ref, M, atol=tol, rtol=tol):
                diff_norm = np.linalg.norm(ref - M)
                logger.warning(f"⚠️ Les matrices '{ref_name}' et '{name}' diffèrent (‖Δ‖={diff_norm:.3e}).")
                all_equal = False
            # else:
            #     logger.info(f"✅ '{ref_name}' et '{name}' sont identiques (tol={tol}).")

        if not all_equal:
            logger.warning("⚠️ Certaines matrices diffèrent — voir messages ci-dessus.")
            input('Waiting for you!')
        # else:
        #     logger.info(f"✅ Toutes les matrices ({', '.join(names)}) sont identiques.")

    def process_upkf(self, N=None, data_generator=None):
        """
        Generator of UPKF filter.
        It makes use of data generator called _data_generation().
        """
        
        if not ((isinstance(N, int) and N > 0) or N is None):
            raise ValueError("sKey must be None or a number >0")
        
        # Data generator
        generator = data_generator if data_generator is not None else self._data_generation()

        # Short-cuts
        A, mQ             = self.param.A, self.param.mQ
        dimx, dimy, dimxy = self.param.dim_x, self.param.dim_y, self.param.dim_xy

        # The first
        ###################
       
        # First generated data sample
        k, (xkp1, ykp1) = next(generator) # les parenthèses servent à déballer la liste de 2 élements

        # Filtering of the first sample
        temp          = self.param.b.T @ np.linalg.inv(self.param.syy)
        Xkp1_update   = temp @ ykp1
        PXXkp1_update = self.param.sxx - temp @ self.param.b

        # Check if cov matrices are indeed cov matrices!
        self._check_consistency(PXXkp1_update=PXXkp1_update)

        # Store if save_pickle==True
        if self.save_pickle and self._history is not None:
            self._history.record(   iter                 = k,
                                    xkp1                 = xkp1.copy(),
                                    ykp1                 = ykp1.copy(),
                                    Xkp1_predict         = np.zeros(shape=(dimx, 1)),      # il n'y a pas de prédiction pour la premier
                                    Pkp1_predict         = np.eye(dimx),                   # il n'y a pas de prédiction pour la premier
                                    ikp1                 = np.zeros(shape=(dimy, 1)),      # il n'y a pas de prédiction pour la premier
                                    Skp1                 = np.eye(dimy),                   # il n'y a pas de prédiction pour la premier
                                    Kkp1                 = np.zeros(shape=(dimx, dimy)),   # il n'y a pas de prédiction pour la premier
                                    Xkp1_update_math     = Xkp1_update.copy(),
                                    PXXkp1_update_math   = PXXkp1_update.copy(),
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
            M_top, M_bottom                = np.vsplit(Pkp1_predict, [dimx])
            PXXkp1_predict, PXYkp1_predict = np.hsplit(M_top,    [dimx])
            PYXkp1_predict, PYYkp1_predict = np.hsplit(M_bottom, [dimx])

            #######################################
            # Update with a new observation
            #######################################
            
            # Get new obervation from the data generator
            try:
                k, (xkp1, ykp1) = next(generator) # parenthesis is used to flatten the list of two elements
            except StopIteration:
                # generator qui fournit les données est terminé, on arrête alors process_upkf
                return

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
            # Updating expectation and variance, and variance in Joseph form
            Xkp1_update   = Xkp1_predict + Kkp1 @ ikp1
            PXXkp1_update = PXXkp1_predict - Kkp1 @ PYXkp1_predict
            # Dans la forme de Joseph, j'utilise les sous matrices de mQ, qui ne sont pas des matrices mais des ActiveView.
            # pour revenir à un forme np.ndarray, j'utilise l'opérateur value (méthode définie dans la classe ActiveView de ParamUPKF.py)
            PXXkp1_update_Joseph =  (self.param.A_xx.value - Kkp1 @ self.param.A_yx.value) @ PXXk_update @ (self.param.A_xx.value - Kkp1 @ self.param.A_yx.value).T \
                + self.param.Q_xx.value - Kkp1 @ self.param.Q_yx.value - self.param.Q_xy.value @ Kkp1.T + Kkp1 @ self.param.Q_yy.value @ Kkp1.T

            Xkp1_update_phys          = Xkp1_update.copy()
            PXXkp1_update_phys        = PXXkp1_update.copy()
            PXXkp1_update_Joseph_phys = PXXkp1_update_Joseph.copy()
            
            # Check if cov matrices are indeed cov matrices!
            self._check_consistency(Pkp1_predict         = Pkp1_predict, 
                                    Skp1                 = Skp1, 
                                    PXXkp1_update_math   = PXXkp1_update_math,
                                    PXXkp1_update_phys   = PXXkp1_update_phys,
                                    PXXkp1_update_Joseph = PXXkp1_update_Joseph)
            # Check if all cov matrices are identical
            self._check_equality(   PXXkp1_update_math   = PXXkp1_update_math,
                                    PXXkp1_update_phys   = PXXkp1_update_phys,
                                    PXXkp1_update_Joseph = PXXkp1_update_Joseph)
            
            # Check if all expectations vectors are identical
            self._check_equality(   Xkp1_update_math     = Xkp1_update_math,
                                    Xkp1_update_phys     = Xkp1_update_phys)

            # Store if save_pickle==True
            if self.save_pickle and self._history is not None:
                self._history.record(iter                 = k,
                                     xkp1                 = xkp1.copy(),
                                     ykp1                 = ykp1.copy(),
                                     Xkp1_predict         = Xkp1_predict,
                                     Pkp1_predict         = Pkp1_predict,
                                     ikp1                 = ikp1.copy(),
                                     Skp1                 = Skp1.copy(),
                                     Kkp1                 = Kkp1.copy(),
                                     Xkp1_update_math     = Xkp1_update_math,
                                     PXXkp1_update_math   = PXXkp1_update_math,
                                     Xkp1_update_phys     = Xkp1_update_phys,
                                     PXXkp1_update_phys   = PXXkp1_update_phys,
                                     PXXkp1_update_Joseph = PXXkp1_update_Joseph)

            yield xkp1, ykp1, Xkp1_update_math, Xkp1_update_phys

    def process_N_data(self, N, data_generator=None):
        return list(self.process_upkf(N=N, data_generator=data_generator))



if __name__ == "__main__":
    """
    Exemple d'utilisation du UPKF.
    Pour exécuter :
        python prg/UPKF.py
    """
    # ------------------------------------------------------------------
    # Constants
    # ------------------------------------------------------------------
    save_pickle = True
    verbose     = 0
    N           = 50
    
    # ------------------------------------------------------------------
    # Output repo for data, traces and plots
    # ------------------------------------------------------------------
    base_dir     = os.path.join(".",      "dataGenerated")
    tracker_dir  = os.path.join(base_dir, "historyTracker")
    datafile_dir = os.path.join(base_dir, "datafile")
    graph_dir    = os.path.join(base_dir, "plot")
    os.makedirs(tracker_dir, exist_ok=True)
    os.makedirs(graph_dir,   exist_ok=True)

    # ------------------------------------------------------------------
    # Test parameters
    # ------------------------------------------------------------------
    from models.model_dimx1_dimy1 import model_dimx1_dimy1_from_Sigma
    dim_x, dim_y, sxx, syy, a, b, c, d, e = model_dimx1_dimy1_from_Sigma()
    param = ParamUPKF(dim_x, dim_y, verbose, sxx=sxx, syy=syy, a=a, b=b, c=c, d=d, e=e)
    if verbose > 0:
        param.summary()


    # ------------------------------------------------------------------
    # Let's go
    # ------------------------------------------------------------------

    print("\nUPKF filtering with data generated from a UPKF... ")
    sKey  = 41
    upkf_1 = PKF(param, sKey=sKey, save_pickle=save_pickle, verbose=verbose)
    # Call with the default data simulator generator
    listeUPKF_1 = upkf_1.process_N_data(N=N)

    # Calcul du RMSE entre le simulé et l'estimation
    first_arrays  = np.vstack([t[0] for t in listeUPKF_1])
    third_arrays  = np.vstack([t[2] for t in listeUPKF_1])
    # Calcul du RMSE global
    print(f"RMSE (X, Esp[X]) : {rmse(first_arrays, third_arrays)}")
    
    if save_pickle and upkf_1.history is not None:
        df = upkf_1.history.as_dataframe()
        if verbose > 0:
            print("\nExtract of the resulting filtering with UPKF :")
            print(df.head())
            # print(df.info())

        # pickle storing and plots
        upkf_1.history.save_pickle(os.path.join(tracker_dir, f"history_run_upfk_1.pkl"))
        upkf_1.history.plot(list_param=["xkp1", "Xkp1_update"], \
                            list_label=["X - Ground Truth", "X - Filtered"], \
                            basename='upkf_1', \
                            show=False, base_dir=graph_dir)
        
    
    
    # datafile = 'data_dim2x2.parquet'
    # #datafile = 'data_dim1x1.csv'
    # print("\nUPKF filtering with data generated from a file... ")
    # upkf_2 = UPKF(param, save_pickle=save_pickle, verbose=verbose)
    # # Call with a fil as data generator
    # filename = os.path.join(datafile_dir, datafile)
    # listeUPKF_2 = upkf_2.process_N_data(N=None, data_generator=file_data_generator(filename, dim_x, verbose))
    # # print(f'listeUPKF_2={listeUPKF_2}')

    # if save_pickle and upkf_2.history is not None:
    #     df = upkf_2.history.as_dataframe()
    #     if verbose > 0:
    #         print("\nExtract of the resulting filtering with UPKF :")
    #         print(df.head())
    #         # print(df.info())

    #     # pickle storing and plots
    #     upkf_2.history.save_pickle(os.path.join(tracker_dir, f"history_run_upfk_2.pkl"))
    #     upkf_2.history.plot(list_param=["xkp1", "Xkp1_update"], \
    #                        list_label=["X - Ground Truth", "X - Filtered"], \
    #                        basename='upkf_2', \
    #                        show=False, base_dir=graph_dir)

