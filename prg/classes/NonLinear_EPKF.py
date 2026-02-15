#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
####################################################################
Extended Pairwise Kalman filter (EPKF) implementation
####################################################################
"""

from __future__ import annotations

import numpy as np
from rich import print

from .NonLinear_PKF import NonLinear_PKF, PKF
# A few utils functions that are used several times
from others.utils import diagnose_covariance, rich_show_fields

class NonLinear_EPKF(NonLinear_PKF):
    """Implementation of EPKF."""

    def __init__(self, param, sKey=None, verbose=0) -> None:
        super().__init__(param, sKey, verbose)

    def process_filter(self, N: Optional[int] = None, data_generator: Optional[Generator] = None) -> Generator:
        """
        Generator of EPKF filter using optional data generator.
        """
        if __debug__:
            if not ((isinstance(N, int) and N > 0) or N is None):
                raise ValueError("N must be None or a number >0")

        generator = data_generator if data_generator is not None else self._data_generation()
        
        # short-cuts supplementary
        self.jg = self.param.jacobiens_g

        # The first
        ##################################################################################################
        step = self._firstEstimate(generator)
        yield step.k, step.xkp1, step.ykp1, step.Xkp1_predict, step.Xkp1_update

        ##################################################################################################@
        # The next ones
        accel_xy_xy           = self.zeros_dim_xy_xy.copy()
        Xkp1_update_augmented = self.zeros_dim_xy_1.copy()
        
        while N is None or step.k<N:
            
            # here ykp1 still gives the previous : it is yk indeed!
            Xkp1_update_augmented[:self.dim_x] = step.Xkp1_update
            Xkp1_update_augmented[self.dim_x:] = step.ykp1
            
            # Prediction
            Zkp1_predict = self.g( Xkp1_update_augmented, self.zeros_dim_xy_1, self.dt)
            An, Bn       = self.jg(Xkp1_update_augmented, self.zeros_dim_xy_1, self.dt)
            accel_xy_xy.fill(0.0)
            accel_xy_xy[0:self.dim_x, 0:self.dim_x] = step.PXXkp1_update
            Pkp1_predict =  An @ accel_xy_xy @ An.T + Bn @ self.mQ @ Bn.T
            self._test_CovMatrix(Pkp1_predict, step.k)
 
            # New data is arriving ##################################
            try:
                new_k, new_xkp1, new_ykp1 = next(generator)
            except StopIteration:
                return # we stop as the data generator is stopped itself

            # Updating ##############################################
            step = self._nextUpdating(new_k, new_xkp1, new_ykp1, Zkp1_predict, Pkp1_predict)
            yield step.k, step.xkp1, step.ykp1, step.Xkp1_predict, step.Xkp1_update
