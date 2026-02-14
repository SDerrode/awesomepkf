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

from classes.NonLinear_PKF import NonLinear_PKF
# A few utils functions that are used several times
from others.utils import diagnose_covariance, rich_show_fields

class NonLinear_EPKF(NonLinear_PKF):
    """Implementation of EPKF."""

    def __init__(self, param, sKey=None, verbose=0) -> None:
        super().__init__(param, sKey, verbose)

    def process_nonlinearfilter(self, N=None, data_generator=None):
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
        k, xkp1, ykp1, Xkp1_predict, Xkp1_update, PXXkp1_update = self._firstEstimate(generator)
        yield k, xkp1, ykp1, Xkp1_predict, Xkp1_update

        ##################################################################################################@
        # The next ones
        
        accel_zero_xy_1 = np.zeros(shape=(self.dim_xy, 1))
        accel_xy_xy     = np.zeros(shape=(self.dim_xy, self.dim_xy))
        
        Xkp1_update_augmented = np.zeros((self.dim_xy, 1))
        while N is None or k<N:
            
            # here ykp1 still gives the previous : it is yk indeed!
            Xkp1_update_augmented[:self.dim_x] = Xkp1_update
            Xkp1_update_augmented[self.dim_x:] = ykp1
            
            # Prediction
            Zkp1_predict = self.g(Xkp1_update_augmented, accel_zero_xy_1, self.dt)
            An, Bn       = self.jg(Xkp1_update_augmented, accel_zero_xy_1, self.dt)
            accel_xy_xy.fill(0.0)
            accel_xy_xy[0:self.dim_x, 0:self.dim_x] = PXXkp1_update 
            
            Pkp1_predict =  An @ accel_xy_xy @ An.T + Bn @ self.mQ @ Bn.T
            if not self.augmented:
                verdict, report = diagnose_covariance(Pkp1_predict)
                if not verdict:
                    print(f'Pkp1_predict={Pkp1_predict}\nReport - iteration k={k}:')
                    rich_show_fields(report, ["is_symmetric", "cholesky_ok", "is_psd", "near_singular", "ill_conditioned", "numerically_singular"], title="")
                    input('attente')
            
 
            # New data is arriving ##################################
            try:
                k, xkp1, ykp1 = next(generator)
            except StopIteration:
                return # we stop as the data generator is stopped itself

            # updating ##############################################
            Xkp1_predict, Xkp1_update, PXXkp1_update = self._nextUpdating(k, xkp1, ykp1, Zkp1_predict, Pkp1_predict)
            yield k, xkp1, ykp1, Xkp1_predict, Xkp1_update
