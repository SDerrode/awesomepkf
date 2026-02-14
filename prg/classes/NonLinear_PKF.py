#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Modèles non linéaires (UPKF et EPKF)
"""

from __future__ import annotations

import os
import math
import logging
import warnings
from typing import Generator, Optional, Tuple

import numpy as np

# A few utils functions that are used several times
from others.utils import check_consistency, check_equality
# Manage parameters for non linear models
from classes.ParamNonLinear import ParamNonLinear
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
        if self.verbose==0 or self.verbose==1:
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
        # print(f'Zkp1_simul={Zkp1_simul}')

        k = 0
        yield k, np.split(Zkp1_simul, [self.dim_x])

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
            yield k, (Xkp1_simul, Ykp1_simul)


    def process_N_data(self, N: Optional[int], data_generator: Optional[Generator] = None) -> list:
        return list(self.process_nonlinearfilter(N=N, data_generator=data_generator))

    def simulate_N_data(self, N):
        return list(self._data_generation(N))

