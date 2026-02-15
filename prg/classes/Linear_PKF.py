#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Module LINEAR PKF ##################################################
####################################################################
Implémente un filtre de Kalman couple (PKF) 
Un exemple d'usage est donné dans le programme principal ci-dessous.
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

import numpy as np

# Base class for all PKF filter
from .PKF import PKF
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


class Linear_PKF(PKF):
    """PKF : base class for all filter classes."""

    def __init__(self, param: ParamLinear, sKey: Optional[int] = None, verbose: int = 0):

        if __debug__:
            if not isinstance(param, ParamLinear):
                raise TypeError("param must be an object from class ParamLinear")
        self.param     = param
        
        super().__init__(sKey, verbose)

    def process_filter(self, N: Optional[int] = None, data_generator: Optional[Generator] = None) -> Generator:
        """
        Generator of PKF filter.
        It makes use of data generator called data_generator().
        """

        if __debug__:
            if not ((isinstance(N, int) and N > 0) or N is None):
                raise ValueError("sKey must be None or a number >0")

        generator = data_generator if data_generator is not None else self._data_generation()

        # short-cuts supplementary
        A, B  = self.param.A, self.param.B
        AT    = A.T
        BmQBT = B @ self.mQ @ B.T

        # The first
        ##################################################################################################
        step = self._firstEstimate(generator)
        yield step.k, step.xkp1, step.ykp1, step.Xkp1_predict, step.Xkp1_update

        ###################
        # The next ones
        accel_xy_xy = self.zeros_dim_xy_xy.copy()
        Xkp1_update_augmented = self.zeros_dim_xy_1.copy()
        
        while N is None or step.k<N:

            #######################################
            # Prediction
            #######################################
            
            # here ykp1 still gives the previous : it is yk indeed!
            Xkp1_update_augmented[:self.dim_x] = step.Xkp1_update
            Xkp1_update_augmented[self.dim_x:] = step.ykp1

            # Prediction
            Zkp1_predict = self.g(Xkp1_update_augmented, self.zeros_dim_xy_1, self.dt)
            accel_xy_xy[0:self.dim_x, 0:self.dim_x] = step.PXXkp1_update
            Pkp1_predict = A @ accel_xy_xy @ AT + BmQBT
            self._test_CovMatrix(Pkp1_predict, step.k)

            # New data is arriving ##################################
            try:
                new_k, new_xkp1, new_ykp1 = next(generator)
            except StopIteration:
                return # we stop as the data generator is stopped itself

            # Updating ##############################################
            step = self._nextUpdating(new_k, new_xkp1, new_ykp1, Zkp1_predict, Pkp1_predict)
            yield step.k, step.xkp1, step.ykp1, step.Xkp1_predict, step.Xkp1_update
