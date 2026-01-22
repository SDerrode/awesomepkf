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

from classes.NonLinear_PKF import NonLinear_PKF
# A few utils functions that are used several times
from others.utils import check_consistency#, check_equality

class NonLinear_UPKF(NonLinear_PKF):
    """Implementation of UPKF."""

    def __init__(
        self,
        param,
        sKey: Optional[int] = None,
        save_pickle: bool = False,
        verbose: int = 0
    ) -> None:

        super().__init__(param, sKey, save_pickle, verbose)
        
        # print(self.param.alpha)
        # print(self.param.beta)
        # print(self.param.kappa)
        # print(self.param.lambda_)
        # print(self.param.gamma)
        

        # Mean weights Wm, and correlation weights Wc
        self.Wm = np.full(2 * self.dim_x + 1, 1. / (2. * (self.dim_x + self.param.lambda_)))
        self.Wc = np.copy(self.Wm)
        self.Wm[0] = self.param.lambda_ / (self.dim_x + self.param.lambda_)
        self.Wc[0] = self.param.lambda_ / (self.dim_x + self.param.lambda_) + (1. - self.param.alpha**2 + self.param.beta)
        
        # print(f'self.Wm={self.Wm}')
        # print(f'self.Wc={self.Wc}')
        # input('PARAM4')


    def _sigma_points(self, x: np.ndarray, P: np.ndarray) -> np.ndarray:
        """Generate the 2*dim_x+1 sigma points around x"""
        A = np.linalg.cholesky(P)
        sigma: list[np.ndarray] = [x]
        for i in range(self.dim_x):
            sigma.append(x + self.param.gamma * A[:, i].reshape(-1,1))
            sigma.append(x - self.param.gamma * A[:, i].reshape(-1,1))
        # print(f'x={x}')
        # print(f'sigma[0]={sigma[0]}')
        return np.array(sigma)

    def process_nonlinearfilter(self, N: Optional[int] = None, data_generator: Optional[Generator] = None) -> Generator:
        """
        Generator of UPKF filter using optional data generator.
        """
        if __debug__:
            if not ((isinstance(N, int) and N > 0) or N is None):
                raise ValueError("N must be None or a number >0")

        generator = data_generator if data_generator is not None else self._data_generation()
        # short-cuts
        Pz00, g, mQ = self.param._Pz00, self.param.g, self.param.mQ

        # The first
        ###################
        k, (xkp1, ykp1) = next(generator) # parenthesis are used to flatten the list of two items
        # temp          = Pz00[0:self.dim_x, self.dim_x:] @ np.linalg.inv(Pz00[self.dim_x:, self.dim_x:])
        # Xkp1_update   = temp @ ykp1
        # PXXkp1_update = Pz00[0:self.dim_x, 0:self.dim_x] - temp @ Pz00[self.dim_x:, 0:self.dim_x]
        Xkp1_update       = xkp1
        PXXkp1_update     = Pz00[0:self.dim_x, 0:self.dim_x] 
        # print(f'PXXkp1_update={PXXkp1_update}')
        # input('attente')
        check_consistency(PXXkp1_update=PXXkp1_update)

        Xkp1_predict = np.zeros((self.dim_x, 1))
        if self.save_pickle and self._history is not None:
            self._history.record(iter          = k,
                                 xkp1          = xkp1.copy() if xkp1 is not None else None,
                                 ykp1          = ykp1.copy(),
                                 Xkp1_predict  = Xkp1_predict.copy(),
                                 PXXkp1_predict= np.eye(self.dim_x),
                                 Xkp1_update   = Xkp1_update.copy(),
                                 PXXkp1_update = PXXkp1_update.copy())

        yield k, xkp1, ykp1, Xkp1_predict, Xkp1_update

        ###################
        # The next ones

        while N is None or k < N:
            # Sigma points
            # print(f'Xkp1_update={Xkp1_update}')
            # print(f'PXXkp1_update={PXXkp1_update}')
            sigma = self._sigma_points(Xkp1_update, PXXkp1_update)
            # print(f'sigma[1]={sigma[1]}')
            
            sigma_propag = [g(np.vstack((e, ykp1)), np.zeros((self.dim_xy, 1)), self.dt) for e in sigma]  # here ykp1 still gives the previous : it is yk indeed!
            # print(f'sigma_propag[1]={sigma_propag[1]}')

            Zkp1_predict = np.sum(self.Wm[:, None, None] * sigma_propag, axis=0)
            Xkp1_predict, Ykp1_predict = np.split(Zkp1_predict, [self.dim_x])
            Pkp1_predict = mQ.copy()
            for i in range(2*self.dim_x+1):
                diff = sigma_propag[i] - Zkp1_predict
                Pkp1_predict += self.Wc[i] * np.outer(diff, diff)
            # print(f'Zkp1_predict={Zkp1_predict}')
            # print(f'Pkp1_predict={Pkp1_predict}')
            # input('tretretret')
            check_consistency(Pkp1_predict=Pkp1_predict)

            # Cutting Pkp1 into 4 blocks
            M_top, M_bottom                = np.vsplit(Pkp1_predict, [self.dim_x])
            PXXkp1_predict, PXYkp1_predict = np.hsplit(M_top,        [self.dim_x])
            PYXkp1_predict, PYYkp1_predict = np.hsplit(M_bottom,     [self.dim_x])
            
            # print(f'PXYkp1_predict={PXYkp1_predict}')
            # print(f'PYYkp1_predict={PYYkp1_predict}')
            # input('pause')

            try:
                k, (xkp1, ykp1) = next(generator)
            except StopIteration:
                return # we stop as the data generator is stopped itself

            accel         = PXYkp1_predict @ np.linalg.inv(PYYkp1_predict)
            Xkp1_update   = Xkp1_predict   + accel @ (ykp1 - Ykp1_predict)
            # forme non robuste numériquement
            # PXXkp1_update = PXXkp1_predict - accel @ PYXkp1_predict
            # print(f'Xkp1_update={Xkp1_update}')

            # Forme de joseph, robuste numériquement
            # PXXkp1_update = PXXkp1_predict - accel @ PYXkp1_predict
            # print(f'PXXkp1_update={PXXkp1_update}')
            S             = PYYkp1_predict
            K             = PXYkp1_predict @ np.linalg.inv(S)
            # PXXkp1_update = PXXkp1_predict - K @ S @ K.T
            # print(f'PXXkp1_predict={PXXkp1_predict}')
            # print(f'K @ S @ K.T={K @ S @ K.T}')
            # print(f'PXXkp1_update={PXXkp1_update}')
            PXXkp1_update = PXXkp1_predict - K @ PXYkp1_predict.T - PXYkp1_predict @ K.T + K @ S @ K.T
            # print(f'PXXkp1_update={PXXkp1_update}')
            # input('attente')

            check_consistency(PXXkp1_update=PXXkp1_update)

            if self.save_pickle and self._history is not None:
                self._history.record(iter           = k,
                                     xkp1           = xkp1.copy() if xkp1 is not None else None,
                                     ykp1           = ykp1.copy(),
                                     Xkp1_predict   = Xkp1_predict.copy(),
                                     PXXkp1_predict = PXXkp1_predict.copy(),
                                     Xkp1_update    = Xkp1_update.copy(),
                                     PXXkp1_update  = PXXkp1_update.copy())

            yield k, xkp1, ykp1, Xkp1_predict, Xkp1_update
