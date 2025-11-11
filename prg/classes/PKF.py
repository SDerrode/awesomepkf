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

# Linear models 
from models.linear import BaseModelLinear
# A few utils functions that are used several times
from others.Utils import mse, file_data_generator, check_consistency, check_equality
# Manage parameters for the PKF
from classes.ParamPKF import ParamPKF
# Keep trace of execution (all parameters at all iterations)
from classes.HistoryTracker import HistoryTracker
# To manage the seed for random generation
from classes.SeedGenerator import SeedGenerator

# ----------------------------------------------------------------------
# Configuration du logging global
# ----------------------------------------------------------------------
logging.basicConfig(format="[%(levelname)s] %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)


class PKF:
    """Implementation of PKF according to the mathematical and classical formulations."""

    def __init__(
        self,
        param: ParamPKF,
        sKey: Optional[int] = None,
        save_pickle: bool = False,
        verbose: int = 0):
        
        if not isinstance(param, ParamPKF):
            raise TypeError("param msut be an object from class ParamPKF")
        if not ((isinstance(sKey, int) and sKey > 0) or sKey is None):
            raise ValueError("sKey must be None or a number>0")
        if not isinstance(save_pickle, bool):
            raise TypeError("save_pickle must be a boolean")
        if verbose not in [0, 1, 2]:
            raise ValueError("verbose must be 0, 1 or 2")

        self.param     = param
        self.dt        = 1
        self.verbose   = verbose
        self._seed_gen = SeedGenerator(sKey)
        
        # Shortcuts
        self.dim_x, self.dim_y, self.dim_xy = param.dim_x, param.dim_y, param.dim_xy

        # Create HistoryTracker only if save_pickle is True
        self.save_pickle = save_pickle
        self._history    = HistoryTracker() if save_pickle else None
        
        # Configuration du logger selon verbose
        self._set_log_level()

        if self.verbose >= 1:
            logger.info(f"[PKF] Init with sKey={sKey}, verbose={verbose}, save_pickle={save_pickle}")

    # ------------------------------------------------------------------
    # Loger configuration according to verbose
    # ------------------------------------------------------------------
    def _set_log_level(self) -> None:
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
        Generator for the simulation of Z_{k+1} = A * Z_k + W_{k+1}, 
        with W_{k+1} ~ N(0, mQ) and Z_1 ~ N(0, Q1).
        This generator can be replaced by a some data acquired in real-time.
        """
        # Short-cuts
        z00, Pz00, g, A, mQ = self.param.z00, self.param.Pz00, self.param.g, self.param.A, self.param.mQ

        # The first
        k = 0
        Zkp1_simul = self._seed_gen.rng.multivariate_normal(mean=z00.T.flatten(), cov=Pz00).reshape(-1,1)
        # print('Zkp1_simul=', Zkp1_simul)
        # input('toptoptpoto')
        yield k, np.split(Zkp1_simul, [self.dim_x])
        

        # The next ones...
        zerosvector = np.zeros(shape=self.dim_xy)
        while N is None or k < N:
            k += 1
            # Zkp1_simul = A @ Zkp1_simul + self._seed_gen.rng.multivariate_normal(mean=zerosvector, cov=mQ).reshape(-1,1)
            Zkp1_simul = g(Zkp1_simul, self._seed_gen.rng.multivariate_normal(mean=zerosvector, cov=mQ).reshape(-1,1), self.dt)
            # print('Zkp1_simul=', Zkp1_simul)
            # print(np.split(Zkp1_simul, [self.dim_x]))
            # input('toptoptpoto')
            yield k, np.split(Zkp1_simul, [self.dim_x])


    def process_pkf(self, N: Optional[int] = None, data_generator: Optional[Generator] = None) -> Generator:
        """
        Generator of PKF filter (mathematic and physicist formulations).
        It makes use of data generator called data_generator().
        """
        
        if __debug__:
            if not ((isinstance(N, int) and N > 0) or N is None):
                raise ValueError("sKey must be None or a number >0")
        
        # Data generator
        generator = data_generator if data_generator is not None else self._data_generation()

        # Short-cuts
        g, A, mQ = self.param.g, self.param.A, self.param.mQ

        # The first
        ###################
        k, (xkp1, ykp1) = next(generator) # parenthesis are used to flatten the list of two items
        temp            = self.param.Pz00[0:self.dim_x, self.dim_x:] @ np.linalg.inv(self.param.Pz00[self.dim_x:, self.dim_x:])
        Xkp1_update     = temp @ ykp1
        PXXkp1_update   = self.param.Pz00[0:self.dim_x, 0:self.dim_x] - temp @ self.param.Pz00[self.dim_x:, 0:self.dim_x]
        check_consistency(PXXkp1_update=PXXkp1_update)

        Xkp1_predict = np.zeros(shape=(self.dim_x, 1))
        if self.save_pickle and self._history is not None:
            self._history.record(   iter                 = k,
                                    xkp1                 = xkp1.copy(),
                                    ykp1                 = ykp1.copy(),
                                    Xkp1_predict         = Xkp1_predict.copy(),              # No prediction for the first
                                    Pkp1_predict         = np.eye(self.dim_x),               # No prediction for the first
                                    ikp1                 = np.zeros(shape=(self.dim_y, 1)),           # na
                                    Skp1                 = np.eye(self.dim_y),                        # na
                                    Kkp1                 = np.zeros(shape=(self.dim_x, self.dim_y)),  # ina
                                    Xkp1_update_math     = Xkp1_update.copy(),
                                    PXXkp1_update_math   = PXXkp1_update.copy(),
                                    Xkp1_update_phys     = Xkp1_update.copy(),
                                    PXXkp1_update_phys   = PXXkp1_update.copy(),
                                    PXXkp1_update_Joseph = PXXkp1_update.copy())
        
        yield xkp1, ykp1, Xkp1_predict, Xkp1_update, Xkp1_update # the phys. and math. Xkp1_update are the same

        ###################
        # The next ones

        temp2 = np.zeros(shape=(self.dim_xy, self.dim_xy))
        while N is None or k < N:
            
            # Required for Joseph form
            PXXk_update = PXXkp1_update.copy()

            #######################################
            # Prediction
            #######################################
            temp1 = np.vstack((Xkp1_update, ykp1)) # here ykp1 still gives the previous : it is yk indeed!
            temp2[0:self.dim_x, 0:self.dim_x] = PXXkp1_update

            # Prediction
            # Zkp1_predict = A @ temp1
            Zkp1_predict = g(temp1, np.zeros((self.dim_xy, 1)), self.dt)
            # print(f'Zkp1_predict={Zkp1_predict}')
            # input('tyutyutyut')
            Xkp1_predict, Ykp1_predict = np.split(Zkp1_predict, [self.dim_x]) # 
            Pkp1_predict               = A @ temp2 @ A.T + mQ
            # Cutting Pkp1 into 4 blocks
            M_top, M_bottom                = np.vsplit(Pkp1_predict, [self.dim_x])
            PXXkp1_predict, PXYkp1_predict = np.hsplit(M_top,        [self.dim_x])
            PYXkp1_predict, PYYkp1_predict = np.hsplit(M_bottom,     [self.dim_x])

            #######################################
            # Update with a new observation
            #######################################
            
            # Get new observation from the data generator
            try:
                k, (xkp1, ykp1) = next(generator) # parenthesis are used to flatten the list of two items
            except StopIteration:
                # return # we stop as the data generator is stopped itself
                return

            # Updating with mathematical formulation
            ###############################################
            accel         = PXYkp1_predict @ np.linalg.inv(PYYkp1_predict)
            Xkp1_update   = Xkp1_predict   + accel @ (ykp1 - Ykp1_predict)
            PXXkp1_update = PXXkp1_predict - accel @ PYXkp1_predict
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
            # Dans la forme de Joseph, j'utilise les sous-matrices de mQ, qui ne sont pas des matrices mais des ActiveView.
            # pour revenir à un forme np.ndarray, j'utilise l'opérateur value (méthode définie dans la classe ActiveView )
            PXXkp1_update_Joseph =  (self.param.A_xx.value - Kkp1 @ self.param.A_yx.value) @ PXXk_update @ (self.param.A_xx.value - Kkp1 @ self.param.A_yx.value).T \
                + self.param.mQ_xx.value - Kkp1 @ self.param.mQ_yx.value - self.param.mQ_xy.value @ Kkp1.T + Kkp1 @ self.param.mQ_yy.value @ Kkp1.T

            Xkp1_update_phys   = Xkp1_update.copy()
            PXXkp1_update_phys = PXXkp1_update.copy()

            # Check if cov matrices are indeed cov matrices!
            check_consistency(Pkp1_predict         = Pkp1_predict,
                              Skp1                 = Skp1,
                              PXXkp1_update_math   = PXXkp1_update_math,
                              PXXkp1_update_phys   = PXXkp1_update_phys,
                              PXXkp1_update_Joseph = PXXkp1_update_Joseph)
            # Check if all cov matrices are identical
            check_equality(   PXXkp1_update_math   = PXXkp1_update_math,
                              PXXkp1_update_phys   = PXXkp1_update_phys,
                              PXXkp1_update_Joseph = PXXkp1_update_Joseph)

            # Check if all expectations vectors are identical
            check_equality(   Xkp1_update_math     = Xkp1_update_math,
                              Xkp1_update_phys     = Xkp1_update_phys)

            # Store if save_pickle==True
            if self.save_pickle and self._history is not None:
                self._history.record(iter                 = k,
                                     xkp1                 = xkp1.copy(),
                                     ykp1                 = ykp1.copy(),
                                     Xkp1_predict         = Xkp1_predict,
                                     PXXkp1_predict       = PXXkp1_predict,
                                     ikp1                 = ikp1.copy(),
                                     Skp1                 = Skp1.copy(),
                                     Kkp1                 = Kkp1.copy(),
                                     Xkp1_update_math     = Xkp1_update_math,
                                     PXXkp1_update_math   = PXXkp1_update_math,
                                     Xkp1_update_phys     = Xkp1_update_phys,
                                     PXXkp1_update_phys   = PXXkp1_update_phys,
                                     PXXkp1_update_Joseph = PXXkp1_update_Joseph)

            yield xkp1, ykp1, Xkp1_predict, Xkp1_update_math, Xkp1_update_phys

    def process_N_data(self, N, data_generator=None):
        return list(self.process_pkf(N=N, data_generator=data_generator))
    
    def simulate_N_data(self, N):
        return list(self._data_generation(N))

