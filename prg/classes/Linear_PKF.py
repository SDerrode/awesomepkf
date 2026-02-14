#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Module PKF #########################################################
####################################################################
Implémente un filtre de Kalman couple (PKF) 
  selon la formulation mathématique (Wojciech), ou
  selon la formulation physique (classique, avec expression du gain),
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
from rich import print
from rich.pretty import pprint

from scipy.linalg import cho_factor, cho_solve

import numpy as np

# A few utils functions that are used several times
from others.utils import diagnose_covariance, rich_show_fields
# Manage parameters for the PKF
from classes.ParamLinear import ParamLinear
# Keep trace of execution (all parameters at all iterations)
from classes.HistoryTracker import HistoryTracker
# To manage the seed for random generation
from classes.SeedGenerator import SeedGenerator

# ----------------------------------------------------------------------
# Configuration du logging global
# ----------------------------------------------------------------------
logging.basicConfig(format="[%(levelname)s] %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)


class Linear_PKF:
    """Implementation of PKF according to the mathematical and classical formulations."""

    def __init__(self, param: ParamLinear, sKey: Optional[int] = None, verbose: int = 0):

        if not isinstance(param, ParamLinear):
            raise TypeError("param must be an object from class ParamLinear")
        if not ((isinstance(sKey, int) and sKey > 0) or sKey is None):
            raise ValueError("sKey must be None or a number>0")
        if verbose not in [0, 1, 2]:
            raise ValueError("verbose must be 0, 1 or 2")

        self.param = param
        self.dt = 1
        self.verbose = verbose
        self._seed_gen = SeedGenerator(sKey)

        # Shortcuts
        self.dim_x, self.dim_y, self.dim_xy = param.dim_x, param.dim_y, param.dim_xy

        # Store data in the tracker
        self._history = HistoryTracker(self.verbose)

        # Configuration du logger selon verbose
        self._set_log_level()

        if self.verbose >= 1:
            logger.info(f"[PKF] Init with sKey={sKey}, verbose={verbose}")

    # ------------------------------------------------------------------
    # Loger configuration according to verbose
    # ------------------------------------------------------------------
    def _set_log_level(self) -> None:
        if self.verbose == 0 or self.verbose == 1:
            logger.setLevel(logging.CRITICAL + 1)
        elif self.verbose == 2:
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
        z00, Pz00, g, A, B, mQ = self.param.z00, self.param.Pz00, self.param.g, self.param.A, self.param.B, self.param.mQ

        Zkp1_simul = np.zeros(shape=(self.dim_xy, 1))

        # The first
        if augmented:
            Zkp1_simul[0:self.dim_x, 0] = self._seed_gen.rng.multivariate_normal(mean=z00[0:self.dim_x, 0], cov=Pz00[0:self.dim_x, 0:self.dim_x])
            Zkp1_simul[self.dim_x:,  0] = Zkp1_simul[self.dim_x - self.dim_y:self.dim_x, 0]
        else:
            Zkp1_simul[:, 0] = self._seed_gen.rng.multivariate_normal(mean=z00[:, 0], cov=Pz00)
        Xkp1_simul, Ykp1_simul = np.split(Zkp1_simul, [self.dim_x])

        k = 0
        yield k, Xkp1_simul, Ykp1_simul

        # The next ones...
        zerosvector_xy = np.zeros(shape=(self.dim_xy))
        zerosvector_x = np.zeros(shape=(self.dim_x))
        noise_z = np.zeros(shape=(self.dim_xy, 1))
        while N is None or k<N:
            if augmented:
                noise_z[0:self.dim_x, 0] = self._seed_gen.rng.multivariate_normal(mean=zerosvector_x, cov=mQ[0:self.dim_x, 0:self.dim_x])
                noise_z[self.dim_x:,  0] = noise_z[self.dim_x - self.dim_y:self.dim_x, 0]
            else:
                noise_z[:, 0] = self._seed_gen.rng.multivariate_normal(mean=zerosvector_xy, cov=mQ)
            Zkp1_simul = g(Zkp1_simul, noise_z, self.dt)
            Xkp1_simul, Ykp1_simul = np.split(Zkp1_simul, [self.dim_x])

            k += 1
            yield k, Xkp1_simul, Ykp1_simul

    def process_pkf(self, N: Optional[int] = None, data_generator: Optional[Generator] = None) -> Generator:
        """
        Generator of PKF filter.
        It makes use of data generator called data_generator().
        """

        if __debug__:
            if not ((isinstance(N, int) and N > 0) or N is None):
                raise ValueError("sKey must be None or a number >0")

        generator = data_generator if data_generator is not None else self._data_generation()

        # Short-cuts
        z00, Pz00, g, A, B, mQ, augmented = self.param._z00, self.param._Pz00, self.param.g, \
                                self.param.A, self.param.B, self.param.mQ, self.param.augmented
        AT = A.T
        BmQBT = B @ mQ @ B.T
        # print(f'A={A}')
        # print(f'B={B}')
        # print(f'mQ={mQ}')
        # input('popopooopo')

        # for speed
        eye_dim_y = np.eye(self.dim_y)
        eye_dim_x = np.eye(self.dim_x)

        # The first
        ###################
        k, xkp1, ykp1 = next(generator)

        # temp            = Pz00[0:self.dim_x, self.dim_x:] @ np.linalg.inv(Pz00[self.dim_x:, self.dim_x:])
        # Xkp1_update     = temp @ ykp1
        # PXXkp1_update   = Pz00[0:self.dim_x, 0:self.dim_x] - temp @ Pz00[self.dim_x:, 0:self.dim_x]
        Xkp1_update     = xkp1 #z00[0:self.dim_x]
        PXXkp1_update   = Pz00[0:self.dim_x, 0:self.dim_x]
        if not augmented:
            verdict, report = diagnose_covariance(PXXkp1_update)
            if not verdict:
                print(f'PXXkp1_update={PXXkp1_update}\nReport for PXXkp1_update - iteration k={k}:')
                print(report)
                input('attente')

        # Record data in the tracker
        Xkp1_predict = np.zeros(shape=(self.dim_x, 1))
        self._history.record(iter =k,
                             xkp1           = xkp1.copy() if xkp1 is not None else None,
                             ykp1           = ykp1.copy(),
                             Xkp1_predict   = Xkp1_predict.copy(),             # No prediction for the first
                             PXXkp1_predict = eye_dim_x,                       # No prediction for the first
                             ikp1           = np.zeros(shape=(self.dim_y, 1)),           # na
                             Skp1           = eye_dim_y,                                 # na
                             Kkp1           = np.zeros(shape=(self.dim_x, self.dim_y)),  # na
                             Xkp1_update    = Xkp1_update.copy(),
                             PXXkp1_update  = PXXkp1_update.copy()
                             )

        # last = self._history.last()
        # rich_show_fields(last, ["iter", "xkp1", "Xkp1_predict", "PXXkp1_predict", "ikp1", "Skp1", "Kkp1", "Xkp1_update", "PXXkp1_update"], title="Infos sélectionnées")
        # input('ATTENTE')

        yield k, xkp1, ykp1, Xkp1_predict, Xkp1_update

        ###################
        # The next ones
        
        accel_zero_xy_1 = np.zeros(shape=(self.dim_xy, 1))
        accel_xy_xy     = np.zeros(shape=(self.dim_xy, self.dim_xy))
        
        while N is None or k<N:

            #######################################
            # Prediction
            #######################################
            # here ykp1 still gives the previous : it is yk indeed!
            Xkp1_update_augmented = np.vstack([Xkp1_update, ykp1])

            # Prediction
            Zkp1_predict               = g(Xkp1_update_augmented, accel_zero_xy_1, self.dt)
            Xkp1_predict, Ykp1_predict = np.split(Zkp1_predict, [self.dim_x])
            accel_xy_xy[0:self.dim_x, 0:self.dim_x] = PXXkp1_update
            Pkp1_predict = A @ accel_xy_xy @ AT + BmQBT
            # print(f'Xkp1_update_augmented=\n{Xkp1_update_augmented}')
            # print(f'Zkp1_predict=\n{Zkp1_predict}')
            # print(f'accel_xy_xy=\n{accel_xy_xy}')
            # print(f'A @ accel_xy_xy={A @ accel_xy_xy}')
            # print(f'AT={AT}')
            # print(f'A @ accel_xy_xy @ AT={A @ accel_xy_xy @ AT}')
            # # print(f'BmQBT={BmQBT}')
            # print(f'Pkp1_predict=\n{Pkp1_predict}')
            # input('ATTENTE- po')

            if not augmented:
                verdict, report = diagnose_covariance(Pkp1_predict)
                if not verdict:
                    print(f'ICI - Pkp1_predict={Pkp1_predict}\nReport - iteration k={k}:')
                    print(report)
                    input('attente')

            # Cutting Pkp1 into 4 blocks
            M_top, M_bottom = np.vsplit(Pkp1_predict, [self.dim_x])
            PXXkp1_predict, PXYkp1_predict = np.hsplit(M_top,    [self.dim_x])
            PYXkp1_predict, PYYkp1_predict = np.hsplit(M_bottom, [self.dim_x])

            # New data
            try:
                k, xkp1, ykp1 = next(generator)
            except StopIteration:
                return  # we stop as the data generator is stopped itself

            # Updating with mathematical formulation
            ###############################################
            # accel = PXYkp1_predict @ np.linalg.inv(PYYkp1_predict)
            # print(f'accel={accel}')
            # Version robuste du calcul
            # c, low = cho_factor(PYYkp1_predict)
            # accel = PXYkp1_predict @ cho_solve((c, low), eye_dim_y)
            # print(f'accel={accel}')
            # print(f'ykp1={ykp1}')
            # print(f'Ykp1_predict={Ykp1_predict}')
            # print(f'ykp1 - Ykp1_predict={ykp1 - Ykp1_predict}')
            # print(ykp1, Ykp1_predict, ykp1 - Ykp1_predict)

            # # print(f'accel={accel}')
            # Xkp1_update   = Xkp1_predict   + accel @ (ykp1 - Ykp1_predict)
            # PXXkp1_update = PXXkp1_predict - accel @ PYXkp1_predict
            # print(f'Xkp1_update={Xkp1_update}')
            # print(f'PXXkp1_update={PXXkp1_update}')
            # input('attente')

            # Updating with physical formulation
            ###############################################
            # innovation (expectation and variance)
            # print(f'ykp1={ykp1}')
            # print(f'Ykp1_predict={Ykp1_predict}')
            # print(f'ikp1 = ykp1 - Ykp1_predict={ykp1 - Ykp1_predict}')

            ikp1   = ykp1 - Ykp1_predict
            Skp1   = PYYkp1_predict
            # Kalman gain - Version robuste du calcul
            c, low = cho_factor(Skp1)
            Kkp1   = PXYkp1_predict @ cho_solve((c, low), eye_dim_y)
            # print(f'Kkp1={Kkp1}')

            # Updating expectation and variance, and variance in Joseph form
            Xkp1_update   = Xkp1_predict + Kkp1 @ ikp1
            PXXkp1_update = PXXkp1_predict - Kkp1 @ PYXkp1_predict
            if not augmented:
                verdict, report = diagnose_covariance(PXXkp1_update)
                if not verdict:
                    print(f'PXXkp1_update={PXXkp1_update}\nReport - iteration k={k}:')
                    print(report)
                    input('attente')

            # Forme de Joseph
            temp = np.vstack((eye_dim_x, -Kkp1.T))
            PXXkp1_update_Joseph = temp.T @ Pkp1_predict @ temp
            if not augmented:
                verdict, report = diagnose_covariance(PXXkp1_update_Joseph)
                if not verdict:
                    print(f'PXXkp1_update_Joseph={PXXkp1_update_Joseph}\nReport - iteration k={k}:')
                    print(report)
                    input('attente')

            # Record data in the tracker
            self._history.record(iter           = k,
                                 xkp1           = xkp1.copy() if xkp1 is not None else None,
                                 ykp1           = ykp1.copy(),
                                 Xkp1_predict   = Xkp1_predict.copy(),
                                 PXXkp1_predict = PXXkp1_predict.copy(),
                                 ikp1           = ikp1.copy(),
                                 Skp1           = Skp1.copy(),
                                 Kkp1           = Kkp1.copy(),
                                 Xkp1_update    = Xkp1_update.copy(),
                                 PXXkp1_update  = PXXkp1_update_Joseph.copy(),  # PXXkp1_update.copy()
                                 )
            
            # Si on veut la forme robuste de la variance, on décommente
            PXXkp1_update = PXXkp1_update_Joseph

            # last = self._history.last()
            # rich_show_fields(last, ["iter", "xkp1", "Xkp1_predict", "PXXkp1_predict", "ikp1", "Skp1", "Kkp1", "Xkp1_update", "PXXkp1_update"], title="Infos sélectionnées")
            # input('ATTENTE')

            yield k, xkp1, ykp1, Xkp1_predict, Xkp1_update

    def process_N_data(self, N, data_generator=None):
        return list(self.process_pkf(N=N, data_generator=data_generator))

    def simulate_N_data(self, N):
        return list(self._data_generation(N))
