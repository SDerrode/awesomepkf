#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
####################################################################
Unscented Pairwise Kalman filter (UPKF) implementation
####################################################################
"""

from __future__ import annotations

from typing import Generator, Optional, Tuple
import numpy as np
from rich import print

from classes.NonLinear_UPKF import NonLinear_UPKF
# A few utils functions that are used several times
from others.utils import check_consistency, diagnose_covariance#, check_equality

class NonLinear_CPKF(NonLinear_UPKF):
    """Implementation of CPKF."""

    def __init__(
        self,
        param,
        sKey: Optional[int] = None,
        save_pickle: bool = False,
        verbose: int = 0
    ) -> None:

        super().__init__(param, sKey, save_pickle, verbose)

        # Mean weights Wm, and correlation weights Wc
        self.Wm = np.full(2 * self.dim_x, 1. / (2. * self.dim_x))
        self.Wc = np.copy(self.Wm)

    def _sigma_points(self, x: np.ndarray, P: np.ndarray) -> np.ndarray:
        """Generate the 2*dim_x cubature points around x"""
        A = np.linalg.cholesky(P)
        sigma = []
        for i in range(self.dim_x):
            sigma.append(x + np.sqrt(self.dim_x) * A[:, i].reshape(-1,1))
            sigma.append(x - np.sqrt(self.dim_x) * A[:, i].reshape(-1,1))
        # print(f'x={x}')
        # print(f'sigma[0]={sigma[0]}')
        return np.array(sigma)
