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
from typing import Generator, Optional
import numpy as np
from scipy.linalg import LinAlgError

from .PKF import PKF
from others.utils import diagnose_covariance, rich_show_fields
from classes.ParamLinear import ParamLinear

class Linear_PKF(PKF):
    """PKF : Linear coupled Kalman filter."""

    def __init__(self, param: ParamLinear, sKey: Optional[int] = None, verbose: int = 0):
        if __debug__:
            if not isinstance(param, ParamLinear):
                raise TypeError("param must be an object from class ParamLinear")
        self.param = param
        super().__init__(sKey, verbose)
        

    def process_filter(self, N: Optional[int] = None,
                       data_generator: Optional[Generator[tuple[int, np.ndarray, np.ndarray], None, None]] = None
                      ) -> Generator[tuple[int, np.ndarray, np.ndarray, np.ndarray, np.ndarray], None, None]:
        """
        Generator for Linear PKF filter.
        Yields: k, xkp1, ykp1, Xkp1_predict, Xkp1_update
        """

        if __debug__:
            if not ((isinstance(N, int) and N > 0) or N is None):
                raise ValueError("N must be None or a number >0")

        # self.logger.info(f"Starting process_filter with N={N}")
        generator = data_generator if data_generator is not None else self._data_generation()

        # Short-cuts
        A, B  = self.param.A, self.param.B
        AT    = A.T
        BmQBT = B @ self.mQ @ B.T

        # ------------------------------------------------------------------
        # First step
        # ------------------------------------------------------------------
        step = self._firstEstimate(generator)
        if step.xkp1 is None: # Il n'y a pas de VT
            self.ground_truth = False

        yield step.k, step.xkp1, step.ykp1, step.Xkp1_predict, step.Xkp1_update

        # ------------------------------------------------------------------
        # Initialize temporary matrices
        # ------------------------------------------------------------------
        accel_xy_xy = self.zeros_dim_xy_xy.copy()
        Xkp1_update_augmented = self.zeros_dim_xy_1.copy()

        # ------------------------------------------------------------------
        # Next steps
        # ------------------------------------------------------------------
        while N is None or step.k < N:

            # self.logger.debug(f"Step {step.k}: starting prediction")
            # Assemble augmented state
            Xkp1_update_augmented[:self.dim_x] = step.Xkp1_update
            Xkp1_update_augmented[self.dim_x:] = step.ykp1

            # Prediction
            Zkp1_predict = self.g(Xkp1_update_augmented, self.zeros_dim_xy_1, self.dt)
            accel_xy_xy[0:self.dim_x, 0:self.dim_x] = step.PXXkp1_update
            Pkp1_predict = A @ accel_xy_xy @ AT + BmQBT
            # self.logger.debug(f"Step {step.k}: prediction done, testing covariance")
            self._test_CovMatrix(Pkp1_predict, step.k)

            # New data arrives
            try:
                new_k, new_xkp1, new_ykp1 = next(generator)
            except StopIteration:
                # self.logger.info("Data generator exhausted. Stopping filter.")
                return

            # Updating
            # self.logger.debug(f"Step {new_k}: starting update")
            try:
                step = self._nextUpdating(new_k, new_xkp1, new_ykp1, Zkp1_predict, Pkp1_predict)
            except LinAlgError:
                self.logger.error(f"Step {new_k}: LinAlgError during update")
                raise

            # self.logger.info(f"Step {step.k}: update computed successfully")
            yield step.k, step.xkp1, step.ykp1, step.Xkp1_predict, step.Xkp1_update

        # self.logger.info("process_filter completed.")
