
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Module NON LINEAR PKF ##################################################
####################################################################
Classe de base pour UPKF et EPKF 
Un exemple d'usage est donné dans le programme principal ci-dessous.
####################################################################
"""

# from __future__ import annotations
from typing import Optional
# import logging
# from scipy.linalg import cho_factor, cho_solve
# import numpy as np

from .PKF import PKF
from classes.ParamNonLinear import ParamNonLinear
# from others.utils import diagnose_covariance, rich_show_fields
# from classes.HistoryTracker import HistoryTracker
# from classes.SeedGenerator import SeedGenerator


class NonLinear_PKF(PKF):
    """Base class for non-linear filters (UPKF and EPKF)."""

    def __init__(self, param: ParamNonLinear, sKey: Optional[int] = None, verbose: int = 0) -> None:
        self.param = param
        super().__init__(sKey, verbose)

        if __debug__:
            if not isinstance(param, ParamNonLinear):
                raise TypeError("param must be an object from class ParamNonLinear")
        

        # Logger info
        # self.logger.info(f"NonLinear_PKF instance created with verbose={verbose}")
