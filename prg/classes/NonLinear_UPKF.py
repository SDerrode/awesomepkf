#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
####################################################################
Unscented Pairwise Kalman filter (UPKF) implementation
####################################################################
"""

from __future__ import annotations

import numpy as np
from rich import print

from classes.NonLinear_PKF import NonLinear_PKF
# A few utils functions that are used several times
from others.utils import diagnose_covariance, rich_show_fields

# Sigma points
from classes.SigmaPointsSet import SigmaPointsSet

class FilterConfig:
    def __init__(self, sigma_point_set_name, dim, param):
        self.sigma_point_set_name = sigma_point_set_name
        self.dim                  = dim
        self.param                = param

    def create_sigma_point_set(self) -> SigmaPointsSet:
        try:
            cls = SigmaPointsSet.registry[self.sigma_point_set_name]
        except KeyError:
            raise ValueError(
                f"Jeu de sigma-points inconnu '{self.sigma_point_set_name}'. "
                f"Disponibles : {list(SigmaPointsSet.registry.keys())}"
            )

        return cls(dim=self.dim, param=self.param)


class NonLinear_UPKF(NonLinear_PKF):
    """Implementation of UPKF."""

    def __init__(self, sigma_point_set_name, param, sKey=None, verbose=0):
        super().__init__(param, sKey, verbose)

        self.sigma_point_set_name = sigma_point_set_name

        cfg = FilterConfig(
            sigma_point_set_name = self.sigma_point_set_name,
            dim                  = 2*self.dim_xy,
            param                = self.param
        )

        self.sigma_point_set_obj = cfg.create_sigma_point_set()

    def create_sigma_point_set(self) -> SigmaPointsSet:
        try:
            cls = SigmaPointsSet.registry[self.sigma_point_set_name]
        except KeyError:
            raise ValueError(
                f"Jeu de sigma-points inconnu '{self.sigma_point_set_name}'. "
                f"Disponibles : {list(SigmaPointsSet.registry.keys())}"
            )

        return cls(dim=self.dim)

    def process_filter(self, N: Optional[int] = None, data_generator: Optional[Generator] = None) -> Generator:
        """
        Generator of UPKF filter using optional data generator.
        """
        if __debug__:
            if not ((isinstance(N, int) and N > 0) or N is None):
                raise ValueError("N must be None or a number >0")

        generator = data_generator if data_generator is not None else self._data_generation()

        # The first
        ##################################################################################################
        step = self._firstEstimate(generator)
        yield step.k, step.xkp1, step.ykp1, step.Xkp1_predict, step.Xkp1_update

        ###################
        # The next ones
        z  = np.zeros(shape=(2*self.dim_xy, 1))
        Pa = np.zeros(shape=(2*self.dim_xy, 2*self.dim_xy))
        Pa[self.dim_xy:, self.dim_xy:] = self.mQ
        Pkp1_predict = self.zeros_dim_xy_xy.copy()
        
        while N is None or step.k<N:
            
            # Sigma points et leur propagation par g
            z[0:self.dim_x]                = step.Xkp1_update
            z[self.dim_x:self.dim_xy]      = step.ykp1
            Pa[0:self.dim_x, 0:self.dim_x] = step.PXXkp1_update
            sigma = self.sigma_point_set_obj._sigma_point(z, Pa)
            # here ykp1 still gives the previous : it is yk indeed!
            sigma_propag = [self.g(*np.split(spoint, [self.dim_xy]), self.dt) for spoint in sigma]

            # Predicting ############################################
            Zkp1_predict = np.sum(self.sigma_point_set_obj.Wm[:, None, None] * sigma_propag, axis=0)
            
            # Remise à 0
            Pkp1_predict.fill(0.0)
            for i in range(self.sigma_point_set_obj.nbSigmaPoint):
                diff = sigma_propag[i] - Zkp1_predict
                Pkp1_predict += self.sigma_point_set_obj.Wc[i] * np.outer(diff, diff)
            self._test_CovMatrix(Pkp1_predict, step.k)
            
            # New data is arriving ##################################
            try:
                new_k, new_xkp1, new_ykp1 = next(generator)
            except StopIteration:
                return # we stop as the data generator is stopped itself

            # Updating ##############################################
            step = self._nextUpdating(new_k, new_xkp1, new_ykp1, Zkp1_predict, Pkp1_predict)
            yield step.k, step.xkp1, step.ykp1, step.Xkp1_predict, step.Xkp1_update
