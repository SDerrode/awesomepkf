#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Modèles non linéaires (UPKF et EPKF)
"""

from __future__ import annotations


import logging
from typing import Generator, Optional

from scipy.linalg import cho_factor, cho_solve

import numpy as np

# A few utils functions that are used several times
from others.utils import diagnose_covariance, rich_show_fields #check_consistency, check_equality
# Keep trace of execution (all parameters at all iterations)
from classes.HistoryTracker import HistoryTracker
# To manage the seed for random generation
from classes.SeedGenerator import SeedGenerator

# ----------------------------------------------------------------------
# Configuration du logging global
# ----------------------------------------------------------------------
logging.basicConfig(format="[%(levelname)s] %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)

class NonLinear_PKF:
    """Base class for non linear filters (UPKF et EPKF)."""

    def __init__(self, param, sKey: Optional[int] = None, verbose: int = 0) -> None:

        if __debug__:
            if not ((isinstance(sKey, int) and sKey > 0) or sKey is None):
                raise ValueError("sKey must be None or a number>0")
            if verbose not in [0, 1, 2]:
                raise ValueError("verbose must be 0, 1 or 2")

        self.param     = param
        self.dt        = 1
        self.verbose   = verbose
        self._seed_gen = SeedGenerator(sKey)

        # Shortcuts
        self.dim_x, self.dim_y, self.dim_xy = self.param.dim_x, self.param.dim_y, self.param.dim_xy
        
        # short-cuts
        self.z00, self.Pz00, self.g, self.mQ, self.augmented = self.param._z00, self.param._Pz00, self.param.g, self.param.mQ, self.param.augmented

        # for speed
        self.eye_dim_y = np.eye(self.dim_y)
        self.eye_dim_x = np.eye(self.dim_x)

        # Create HistoryTracker
        self._history    = HistoryTracker(self.verbose)
        
        # Configuration du logger selon verbose
        self._set_log_level()

        if self.verbose >= 1:
            logger.info(f"[PKF] Init with sKey={sKey}, verbose={verbose}")

    # ------------------------------------------------------------------
    # Loger configuration according to verbose
    # ------------------------------------------------------------------
    def _set_log_level(self) -> None:

        if self.verbose == 0:
            logger.setLevel(logging.CRITICAL)
        elif self.verbose == 1:
            logger.setLevel(logging.WARNING)
        elif self.verbose == 2:
            logger.setLevel(logging.INFO)

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
    def _data_generation(self, N: Optional[int] = None) -> Generator[tuple[int, np.ndarray, np.ndarray], None, None]:
        """
        Generator for the simulation of Z_{k+1} = A * Z_k + W_{k+1},
        with W_{k+1} ~ N(0, mQ) and Z_1 ~ N(0, Q1).
        """
        # Short-cuts
        z00, Pz00, g, mQ = self.param._z00, self.param._Pz00, self.param.g, self.param.mQ
        
        Zkp1_simul = np.zeros(shape=(self.dim_xy, 1))

        # The first
        if self.param.augmented:
            Zkp1_simul[0:self.dim_x, 0] = self._seed_gen.rng.multivariate_normal(mean=z00[0:self.dim_x,0], cov=Pz00[0:self.dim_x, 0:self.dim_x])
            Zkp1_simul[self.dim_x:,  0] = Zkp1_simul[self.dim_x-self.dim_y:self.dim_x, 0]
        else:
            Zkp1_simul[:,0] = self._seed_gen.rng.multivariate_normal(mean=z00[:, 0], cov=Pz00)
        Xkp1_simul, Ykp1_simul = np.split(Zkp1_simul, [self.dim_x])
        # print(f'Zkp1_simul={Zkp1_simul}')

        k = 0
        yield k, Xkp1_simul, Ykp1_simul

        # The next ones...
        zerosvector_xy = np.zeros(shape=(self.dim_xy))
        zerosvector_x  = np.zeros(shape=(self.dim_x))
        noise_z        = np.zeros(shape=(self.dim_xy, 1))
        while N is None or k<N:
            
            if self.param.augmented:
                noise_z[0:self.dim_x, 0] = self._seed_gen.rng.multivariate_normal(mean=zerosvector_x, cov=mQ[0:self.dim_x, 0:self.dim_x])
                noise_z[self.dim_x:, 0]  = noise_z[self.dim_x-self.dim_y:self.dim_x, 0]
            else:
                noise_z[:, 0] = self._seed_gen.rng.multivariate_normal(mean=zerosvector_xy, cov=mQ)
           
            Zkp1_simul = g(Zkp1_simul, noise_z, self.dt)
            Xkp1_simul, Ykp1_simul = np.split(Zkp1_simul, [self.dim_x])
            
            k += 1
            yield k, Xkp1_simul, Ykp1_simul

    # ------------------------------------------------------------------
    # Factorizations
    # ------------------------------------------------------------------
    def _firstEstimate(self, generator):
        
        k, xkp1, ykp1 = next(generator)
        
        # temp            = self.Pz00[0:self.dim_x, self.dim_x:] @ np.linalg.inv(self.Pz00[self.dim_x:, self.dim_x:])
        # Xkp1_update     = temp @ ykp1
        # PXXkp1_update   = self.Pz00[0:self.dim_x, 0:self.dim_x] - temp @ self.Pz00[self.dim_x:, 0:self.dim_x]
        Xkp1_update     = xkp1 #z00[0:self.dim_x]
        PXXkp1_update   = self.Pz00[0:self.dim_x, 0:self.dim_x].copy()
        if not self.augmented:
            verdict, report = diagnose_covariance(PXXkp1_update)
            if not verdict:
                print(f'PXXkp1_update={PXXkp1_update}\nReport for PXXkp1_update - iteration k={k}:')
                rich_show_fields(report, ["is_symmetric", "cholesky_ok", "is_psd", "near_singular", "ill_conditioned", "numerically_singular"], title="")
                input('attente')

        # Record data in the tracker
        Xkp1_predict  = np.zeros(shape=(self.dim_x, 1))
        self._history.record(iter           = k,
                             xkp1           = xkp1.copy() if xkp1 is not None else None,
                             ykp1           = ykp1.copy(),
                             Xkp1_predict   = Xkp1_predict.copy(),
                             PXXkp1_predict = self.eye_dim_x,
                             ikp1           = np.zeros(shape=(self.dim_y, 1)),
                             Skp1           = self.eye_dim_y,
                             Kkp1           = np.zeros(shape=(self.dim_x, self.dim_y)),
                             Xkp1_update    = Xkp1_update.copy(),
                             PXXkp1_update  = PXXkp1_update.copy()
        )
        
        # last = self._history.last()
        # rich_show_fields(last, ["iter", "xkp1", "Xkp1_predict", "PXXkp1_predict", "ikp1", "Skp1", "Kkp1", "Xkp1_update", "PXXkp1_update"], title="")
        # input('ATTENTE')

        return k, xkp1, ykp1, Xkp1_predict, Xkp1_update, PXXkp1_update


    def _nextUpdating(self, k, xkp1, ykp1, Zkp1_predict, Pkp1_predict):
        
        Xkp1_predict, Ykp1_predict = np.split(Zkp1_predict, [self.dim_x])
        # Cutting Pkp1 into 4 blocks
        PXXkp1_predict = Pkp1_predict[:self.dim_x, :self.dim_x]
        PXYkp1_predict = Pkp1_predict[:self.dim_x, self.dim_x:]
        PYXkp1_predict = Pkp1_predict[self.dim_x:, :self.dim_x]
        PYYkp1_predict = Pkp1_predict[self.dim_x:, self.dim_x:]

        # Updating
        ###############################################
        ikp1   = ykp1 - Ykp1_predict
        Skp1 = PYYkp1_predict.copy()

        condS = np.linalg.cond(Skp1)
        if condS > 1e12:
            if self.verbose >= 2:
                logger.warning(f"Skp1 ill-conditioned (cond={condS:.2e})")
            Skp1 += 1e-10 * self.eye_dim_y
        try:
            c, low = cho_factor(Skp1)
            Kkp1   = PXYkp1_predict @ cho_solve((c, low), self.eye_dim_y)
        except np.linalg.LinAlgError as e:
            print(f'Skp1={Skp1}')
            input('ATTENTE')
        except ValueError as e:
            print("Erreur de valeur :", e)
            input('ATTENTE')
        # print(f'Kkp1={Kkp1}')
        
        Xkp1_update   = Xkp1_predict   + Kkp1 @ ikp1
        PXXkp1_update = PXXkp1_predict - Kkp1 @ PYXkp1_predict
        if not self.augmented:
            verdict, report = diagnose_covariance(PXXkp1_update)
            if not verdict:
                print(f'PXXkp1_update={PXXkp1_update}\nReport - iteration k={k}:')
                rich_show_fields(report, ["is_symmetric", "cholesky_ok", "is_psd", "near_singular", "ill_conditioned", "numerically_singular"], title="")
                input('attente')
        
        # Forme de Joseph
        temp = np.vstack((self.eye_dim_x, -Kkp1.T))
        PXXkp1_update_Joseph = temp.T @ Pkp1_predict @ temp
        if not self.augmented:
            verdict, report = diagnose_covariance(PXXkp1_update_Joseph)
            if not verdict:
                print(f'PXXkp1_update_Joseph={PXXkp1_update_Joseph}\nReport - iteration k={k}:')
                rich_show_fields(report, ["is_symmetric", "cholesky_ok", "is_psd", "near_singular", "ill_conditioned", "numerically_singular"], title="")
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
                             PXXkp1_update  = PXXkp1_update_Joseph.copy(), #PXXkp1_update.copy()
        )
        
        # Si on veut la forme robuste de la variance, on décommente
        PXXkp1_update = PXXkp1_update_Joseph

        # last = self._history.last()
        # rich_show_fields(last, ["iter", "xkp1", "Xkp1_predict", "PXXkp1_predict", "ikp1", "Skp1", "Kkp1", "Xkp1_update", "PXXkp1_update"], title="")
        # input('ATTENTE')
        
        return Xkp1_predict, Xkp1_update, PXXkp1_update

    def process_N_data(self, N: Optional[int], data_generator: Optional[Generator] = None) -> list:
        return list(self.process_nonlinearfilter(N=N, data_generator=data_generator))

    def simulate_N_data(self, N):
        return list(self._data_generation(N))

