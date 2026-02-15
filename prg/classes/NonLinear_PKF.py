#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Module NON LINEAR PKF ##################################################
####################################################################
Calsse de base pour UPKF et EPKF 
Un exemple d'usage est donné dans le programme principal ci-dessous.
####################################################################
"""

from __future__ import annotations


import logging
from typing import Generator, Optional

from scipy.linalg import cho_factor, cho_solve

import numpy as np

# Base class for all PKF filter
from .PKF import PKF
# Manage parameters for the PKF
from classes.ParamNonLinear import ParamNonLinear
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

class NonLinear_PKF(PKF):
    """Base class for non linear filters (UPKF et EPKF)."""

    def __init__(self, param: ParamNonLinear, sKey: Optional[int] = None, verbose: int = 0) -> None:

        if __debug__:
            if not isinstance(param, ParamNonLinear):
                raise TypeError("param must be an object from class ParamNonLinear")
        self.param     = param

        super().__init__(sKey, verbose)



