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

    def __init__(
        self,
        param,
        sKey: Optional[int] = None,
        save_pickle: bool = False,
        verbose: int = 0
    ) -> None:

        if __debug__:
            # if not isinstance(param, ParamNonLinear):
            #     raise TypeError("param must be an object from class ParamNonLinear")
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
        self.dim_x, self.dim_y, self.dim_xy = self.param.dim_x, self.param.dim_y, self.param.dim_xy

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
        """
        # Short-cuts
        z00, Pz00, g, mQ = self.param._z00, self.param._Pz00, self.param.g, self.param.mQ

        # The first
        k = 0
        Zkp1_simul = self._seed_gen.rng.multivariate_normal(mean=z00.T.flatten(), cov=Pz00).reshape(-1,1)
        # print(f'Zkp1_simul={Zkp1_simul}')
        # exit(1)
        
        yield k, np.split(Zkp1_simul, [self.dim_x])

        # The next ones...
        zerosvector = np.zeros(self.dim_xy)
        while N is None or k+1 < N:
            k += 1
            Zkp1_simul = g(Zkp1_simul, self._seed_gen.rng.multivariate_normal(mean=zerosvector, cov=mQ).reshape(-1,1), self.dt)
            yield k, np.split(Zkp1_simul, [self.dim_x])

    def process_N_data(self, N: Optional[int], data_generator: Optional[Generator] = None) -> list:
        return list(self.process_nonlinearfilter(N=N, data_generator=data_generator))

    def simulate_N_data(self, N):
        return list(self._data_generation(N))

