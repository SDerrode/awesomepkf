#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
####################################################################
Extended Pairwise Kalman filter (EPKF) implementation
####################################################################
"""

from __future__ import annotations

from typing import Generator, Optional, Tuple
import numpy as np
from rich import print

from classes.NonLinear_PKF import NonLinear_PKF
# A few utils functions that are used several times
from others.utils import check_consistency, diagnose_covariance#, check_equality

class NonLinear_EPKF(NonLinear_PKF):
    """Implementation of EPKF."""

    def __init__(
        self,
        param,
        sKey: Optional[int] = None,
        save_pickle: bool = False,
        verbose: int = 0
    ) -> None:

        super().__init__(param, sKey, save_pickle, verbose)

    def process_nonlinearfilter(self, N: Optional[int] = None, data_generator: Optional[Generator] = None) -> Generator:
        """
        Generator of EPKF filter using optional data generator.
        """
        if __debug__:
            if not ((isinstance(N, int) and N > 0) or N is None):
                raise ValueError("N must be None or a number >0")

        generator = data_generator if data_generator is not None else self._data_generation()
        
        # short-cuts
        g, jg, mQ = self.param.g, self.param.jacobiens_g, self.param.mQ

        # The first
        ###################
        k, (xkp1, ykp1) = next(generator) # parenthesis are used to flatten the list of two items
        # temp            = self.param.Pz00[0:self.dim_x, self.dim_x:] @ np.linalg.inv(self.param.Pz00[self.dim_x:, self.dim_x:])
        # Xkp1_update     = temp @ ykp1
        # PXXkp1_update   = self.param.Pz00[0:self.dim_x, 0:self.dim_x] - temp @ self.param.Pz00[self.dim_x:, 0:self.dim_x]
        Xkp1_update       = xkp1
        PXXkp1_update     = self.param.Pz00[0:self.dim_x, 0:self.dim_x] 
        # print(f'PXXkp1_update={PXXkp1_update}')
        # input('attente')
        # check_consistency(PXXkp1_update=PXXkp1_update)
        verdict, report = diagnose_covariance(PXXkp1_update)
        if verdict != None:
            print(f'PXXkp1_update={PXXkp1_update}')
            print(f'report for PXXkp1_update - iteration k={k}:')
            print(report)
            input('attente')

        Xkp1_predict = np.zeros((self.dim_x, 1))
        if self.save_pickle and self._history is not None:
            self._history.record(iter          = k,
                                 xkp1          = xkp1.copy() if xkp1 is not None else None,
                                 ykp1          = ykp1.copy(),
                                 ikp1          = np.zeros(shape=(self.dim_y)),
                                 Skp1          = np.eye(self.dim_y),
                                 Kkp1          = np.zeros(shape=(self.dim_x, self.dim_y)),
                                 Xkp1_predict  = Xkp1_predict.copy(),
                                 PXXkp1_predict= np.eye(self.dim_x),
                                 Xkp1_update   = Xkp1_update.copy(),
                                 PXXkp1_update = PXXkp1_update.copy())

        yield k, xkp1, ykp1, Xkp1_predict, Xkp1_update

        ###################
        # The next ones
        accel_zero_xy  = np.zeros(shape=(self.dim_xy, 1))
        accel_zero_x_y = np.zeros((self.dim_x, self.dim_y))
        accel_zero_y_x = np.zeros((self.dim_y, self.dim_x))
        accel_zero_y_y = np.zeros((self.dim_y, self.dim_y))
        while N is None or k < N:
            
            # Required for Joseph form
            PXXk_update = PXXkp1_update.copy()
            
            Xkp1_update_augmented      = np.vstack([Xkp1_update, ykp1])
            Zkp1_predict               = g( Xkp1_update_augmented, accel_zero_xy, self.dt)
            Xkp1_predict, Ykp1_predict = np.split(Zkp1_predict, [self.dim_x])
            An, Bn                     = jg(Xkp1_update_augmented, accel_zero_xy, self.dt)
            Pkp1_predict               = Bn @ mQ @ Bn.T + An @ np.block(
                                                [[PXXkp1_update,  accel_zero_x_y],
                                                [accel_zero_y_x, accel_zero_y_y]]
                                            ) @ An.T
            # print(f'Zkp1_predict={Zkp1_predict}')
            # print(f'Pkp1_predict={Pkp1_predict}')
            # check_consistency(Pkp1_predict=Pkp1_predict)
            # Il ne faut pas dignostiquer la matrice car elle est singulière, 
            # mais cela n'est pas grave pour le reste des calculs
            # verdict, report = diagnose_covariance(Pkp1_predict)
            # if verdict != None:
            #     print(f'Pkp1_predict={Pkp1_predict}')
            #     print(f'report for Pkp1_predict - iteration k={k}:')
            #     print(report)
            #     input('attente')

            # Cutting Pkp1 into 4 blocks
            M_top, M_bottom                = np.vsplit(Pkp1_predict, [self.dim_x])
            PXXkp1_predict, PXYkp1_predict = np.hsplit(M_top,        [self.dim_x])
            PYXkp1_predict, PYYkp1_predict = np.hsplit(M_bottom,     [self.dim_x])
            verdict, report = diagnose_covariance(PXXkp1_predict)
            if verdict != None:
                print(f'PXXkp1_predict={PXXkp1_predict}')
                print(f'report for PXXkp1_predict - iteration k={k}:')
                print(report)
                input('attente')
            verdict, report = diagnose_covariance(PYYkp1_predict)
            if verdict != None:
                print(f'PYYkp1_predict={PYYkp1_predict}')
                print(f'report for PYYkp1_predict - iteration k={k}:')
                print(report)
                input('attente')
            
            # New data
            try:
                k, (xkp1, ykp1) = next(generator)
            except StopIteration:
                return # we stop as the data generator is stopped itself

            ikp1          = ykp1 - Ykp1_predict
            Skp1          = PYYkp1_predict
            Kkp1          = PXYkp1_predict @ np.linalg.inv(Skp1)
            Xkp1_update   = Xkp1_predict   + Kkp1 @ ikp1
            # print(f'Xkp1_update={Xkp1_update}')
            PXXkp1_update = PXXkp1_predict - Kkp1 @ PYXkp1_predict
            # print(f'PXXkp1_update={PXXkp1_update}')
            
            # PXXkp1_update = PXXkp1_predict - Kkp1 @ PXYkp1_predict.T
            # print(f'PXXkp1_update={PXXkp1_update}')
            # PXXkp1_update = PXXkp1_predict - Kkp1 @ S @ Kkp1.T
            # print(f'PXXkp1_update={PXXkp1_update}')
            # PXXkp1_update = PXXkp1_predict - Kkp1 @ PXYkp1_predict.T - PXYkp1_predict @ Kkp1.T + K @ Skp1 @ Kkp1.T
            # Q = Bn @ Bn.T
            # PXXkp1_update = (An[0:self.dim_x, 0:self.dim_x] - Kkp1 @ An[self.dim_x:self.dim_xy, 0:self.dim_x]) @ PXXk_update @ (An[0:self.dim_x, 0:self.dim_x] - K @ An[self.dim_x:self.dim_xy, 0:self.dim_x]).T \
            #     +Q[0:self.dim_x, 0:self.dim_x] - Kkp1 @ Q[0:self.dim_x, self.dim_x:self.dim_xy].T - Q[0:self.dim_x, self.dim_x:self.dim_xy] @ K.T + K @ Q[self.dim_x:self.dim_xy, self.dim_x:self.dim_xy] @ K.T
            # print(f'PXXkp1_update={PXXkp1_update}')
            # input('Attente')

            # Il ne faut pas dignostiquer la matrice car elle est singulière, 
            # mais cela n'est pas grave pour le reste des calculs
            # verdict, report = diagnose_covariance(PXXkp1_update)
            # if verdict != None:
            #     print(f'PXXkp1_update={PXXkp1_update}')
            #     print(f'report for PXXkp1_update - iteration k={k}:')
            #     print(report)
            #     input('attente')

            if self.save_pickle and self._history is not None:
                self._history.record(iter           = k,
                                     xkp1           = xkp1.copy() if xkp1 is not None else None,
                                     ykp1           = ykp1.copy(),
                                     Xkp1_predict   = Xkp1_predict,
                                     PXXkp1_predict = PXXkp1_predict.copy(),
                                     ikp1           = ikp1.copy(),
                                     Skp1           = Skp1.copy(),
                                     Kkp1           = Kkp1.copy(),
                                     Xkp1_update    = Xkp1_update.copy(),
                                     PXXkp1_update  = PXXkp1_update.copy())

            yield k, xkp1, ykp1, Xkp1_predict, Xkp1_update
