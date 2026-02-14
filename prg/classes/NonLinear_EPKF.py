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

from scipy.linalg import cho_factor, cho_solve

from classes.NonLinear_PKF import NonLinear_PKF
# A few utils functions that are used several times
from others.utils import diagnose_covariance, rich_show_fields

class NonLinear_EPKF(NonLinear_PKF):
    """Implementation of EPKF."""

    def __init__(self, param, sKey: Optional[int] = None, verbose: int = 0) -> None:
        super().__init__(param, sKey, verbose)

    def process_nonlinearfilter(self, N: Optional[int] = None, data_generator: Optional[Generator] = None) -> Generator:
        """
        Generator of EPKF filter using optional data generator.
        """
        if __debug__:
            if not ((isinstance(N, int) and N > 0) or N is None):
                raise ValueError("N must be None or a number >0")

        generator = data_generator if data_generator is not None else self._data_generation()
        
        # short-cuts
        z00, Pz00, g, jg, mQ, augmented = self.param._z00, self.param._Pz00, self.param.g, self.param.jacobiens_g, self.param.mQ, self.param.augmented
        
        # for speed
        eye_dim_y = np.eye(self.dim_y)
        eye_dim_x = np.eye(self.dim_x)

        # The first
        ##################################################################################################@
        k, xkp1, ykp1 = next(generator)
        
        # temp            = Pz00[0:self.dim_x, self.dim_x:] @ np.linalg.inv(Pz00[self.dim_x:, self.dim_x:])
        # Xkp1_update     = temp @ ykp1
        # PXXkp1_update   = Pz00[0:self.dim_x, 0:self.dim_x] - temp @ Pz00[self.dim_x:, 0:self.dim_x]
        Xkp1_update     = xkp1 #z00[0:self.dim_x]
        PXXkp1_update   = Pz00[0:self.dim_x, 0:self.dim_x]
        if not augmented:
            verdict, report = diagnose_covariance(PXXkp1_update)
            if not verdict:
                print(f'PXXkp1_update={PXXkp1_update}\nReport for PXXkp1_update - iteration k={k}:')
                rich_show_fields(report, ["is_symmetric", "cholesky_ok", "is_psd", "near_singular", "ill_conditioned", "numerically_singular"], title="")
                input('attente')

        # Record data in the tracker
        Xkp1_predict  = np.zeros(shape=(self.dim_x, 1))
        self._history.record(iter           = k,
                             xkp1           = xkp1.copy() if xkp1 is not None else None,
                             ykp1           = ykp1.copy(),
                             Xkp1_predict   = Xkp1_predict.copy(),
                             PXXkp1_predict = eye_dim_x,
                             ikp1           = np.zeros(shape=(self.dim_y, 1)),
                             Skp1           = eye_dim_y,
                             Kkp1           = np.zeros(shape=(self.dim_x, self.dim_y)),
                             Xkp1_update    = Xkp1_update.copy(),
                             PXXkp1_update  = PXXkp1_update.copy()
        )
        # last = self._history.last()
        # rich_show_fields(last, ["iter", "xkp1", "Xkp1_predict", "PXXkp1_predict", "ikp1", "Skp1", "Kkp1", "Xkp1_update", "PXXkp1_update"], title="")
        # input('ATTENTE')

        yield k, xkp1, ykp1, Xkp1_predict, Xkp1_update

        ##################################################################################################@
        # The next ones
        
        accel_zero_xy_1 = np.zeros(shape=(self.dim_xy, 1))
        accel_zero_x_y  = np.zeros(shape=(self.dim_x,  self.dim_y))
        accel_zero_y_x  = np.zeros(shape=(self.dim_y,  self.dim_x))
        accel_zero_y_y  = np.zeros(shape=(self.dim_y,  self.dim_y))
        accel_xy_xy     = np.zeros(shape=(self.dim_xy, self.dim_xy))
        
        while N is None or k<N:
            # here ykp1 still gives the previous : it is yk indeed!
            Xkp1_update_augmented      = np.vstack([Xkp1_update, ykp1])
            
            # Prediction
            Zkp1_predict               = g(Xkp1_update_augmented, accel_zero_xy_1, self.dt)
            Xkp1_predict, Ykp1_predict = np.split(Zkp1_predict, [self.dim_x])
            An, Bn                     = jg(Xkp1_update_augmented, accel_zero_xy_1, self.dt)
            accel_xy_xy[0:self.dim_x, 0:self.dim_x] = PXXkp1_update
            Pkp1_predict               =  An @ accel_xy_xy @ An.T + Bn @ mQ @ Bn.T
            
            if not augmented:
                verdict, report = diagnose_covariance(Pkp1_predict)
                if not verdict:
                    print(f'Pkp1_predict={Pkp1_predict}\nReport - iteration k={k}:')
                    rich_show_fields(report, ["is_symmetric", "cholesky_ok", "is_psd", "near_singular", "ill_conditioned", "numerically_singular"], title="")
                    input('attente')
            
            # Cutting Pkp1 into 4 blocks
            M_top, M_bottom                = np.vsplit(Pkp1_predict, [self.dim_x])
            PXXkp1_predict, PXYkp1_predict = np.hsplit(M_top,        [self.dim_x])
            PYXkp1_predict, PYYkp1_predict = np.hsplit(M_bottom,     [self.dim_x])

            # New data is arriving
            try:
                k, xkp1, ykp1 = next(generator)
            except StopIteration:
                return # we stop as the data generator is stopped itself

            # Updating
            ###############################################
            ikp1   = ykp1 - Ykp1_predict
            Skp1   = PYYkp1_predict
            # Kkp1   = PXYkp1_predict @ np.linalg.inv(Skp1)
            # print(f'Kkp1={Kkp1}')
            # Version robuste du calcul
            c, low = cho_factor(Skp1)
            Kkp1 = PXYkp1_predict @ cho_solve((c, low), eye_dim_y)
            # print(f'Kkp1={Kkp1}')

            Xkp1_update   = Xkp1_predict   + Kkp1 @ ikp1
            PXXkp1_update = PXXkp1_predict - Kkp1 @ PYXkp1_predict
            
            if not augmented:
                verdict, report = diagnose_covariance(PXXkp1_update)
                if not verdict:
                    print(f'PXXkp1_update={PXXkp1_update}\nReport - iteration k={k}:')
                    rich_show_fields(report, ["is_symmetric", "cholesky_ok", "is_psd", "near_singular", "ill_conditioned", "numerically_singular"], title="")
                    input('attente')
    
            # print(f'Xkp1_update  ={Xkp1_update}')
            # print(f'PXXkp1_update={PXXkp1_update}')
            
            # PXXkp1_update = PXXkp1_predict - Kkp1 @ PXYkp1_predict.T
            # print(f'PXXkp1_update={PXXkp1_update}')
            # PXXkp1_update = PXXkp1_predict - Kkp1 @ S @ Kkp1.T
            # print(f'PXXkp1_update={PXXkp1_update}')
            # PXXkp1_update = PXXkp1_predict - Kkp1 @ PXYkp1_predict.T - PXYkp1_predict @ Kkp1.T + K @ Skp1 @ Kkp1.T
            
            # Forme de Joseph
            # Q = Bn @ Bn.T
            # PXXkp1_update_Joseph = (An[0:self.dim_x, 0:self.dim_x] - Kkp1 @ An[self.dim_x:self.dim_xy, 0:self.dim_x]) @ PXXk_update @ (An[0:self.dim_x, 0:self.dim_x] - K @ An[self.dim_x:self.dim_xy, 0:self.dim_x]).T \
            #     +Q[0:self.dim_x, 0:self.dim_x] - Kkp1 @ Q[0:self.dim_x, self.dim_x:self.dim_xy].T - Q[0:self.dim_x, self.dim_x:self.dim_xy] @ K.T + K @ Q[self.dim_x:self.dim_xy, self.dim_x:self.dim_xy] @ K.T
            #print(f'PXXkp1_update_Joseph={PXXkp1_update_Joseph}')
            temp = np.vstack((eye_dim_x, -Kkp1.T))
            PXXkp1_update_Joseph = temp.T @ Pkp1_predict @ temp
            # print(f'PXXkp1_update_Joseph={PXXkp1_update_Joseph}')
            # input('attente')
            if not augmented:
                verdict, report = diagnose_covariance(PXXkp1_update_Joseph)
                if not verdict:
                    print(f'PXXkp1_update_Joseph={PXXkp1_update_Joseph}\nReport - iteration k={k}:')
                    rich_show_fields(report, ["is_symmetric", "cholesky_ok", "is_psd", "near_singular", "ill_conditioned", "numerically_singular"], title="")
                    input('attente')

            # Record data in the tracker
            self._history.record(iter           = k,
                                 xkp1           = xkp1.copy() if xkp1 is not None else None,
                                 ykp1           = ykp1.copy(),
                                 Xkp1_predict   = Xkp1_predict,
                                 PXXkp1_predict = PXXkp1_predict.copy(),
                                 ikp1           = ikp1.copy(),
                                 Skp1           = Skp1.copy(),
                                 Kkp1           = Kkp1.copy(),
                                 Xkp1_update    = Xkp1_update.copy(),
                                 PXXkp1_update  = PXXkp1_update_Joseph.copy(), #PXXkp1_update.copy())
            )
            
            # Si on veut la forme robuste de la variance, on décommente
            PXXkp1_update = PXXkp1_update_Joseph

            # last = self._history.last()
            # rich_show_fields(last, ["iter", "xkp1", "Xkp1_predict", "PXXkp1_predict", "ikp1", "Skp1", "Kkp1", "Xkp1_update", "PXXkp1_update"], title="")
            # input('ATTENTE')

            yield k, xkp1, ykp1, Xkp1_predict, Xkp1_update
