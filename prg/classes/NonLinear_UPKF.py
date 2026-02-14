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

from scipy.linalg import cho_factor, cho_solve

from classes.NonLinear_PKF import NonLinear_PKF
# A few utils functions that are used several times
from others.utils import check_consistency, diagnose_covariance#, check_equality

# Sigma points
from classes.SigmaPointsSet import SigmaPointsSet

class FilterConfig:
    def __init__(self, sigma_point_set, dim_x, param):
        self.sigma_point_set = sigma_point_set
        self.dim_x = dim_x
        self.param = param

    def create_sigma_point_set(self) -> SigmaPointsSet:
        try:
            cls = SigmaPointsSet.registry[self.sigma_point_set]
        except KeyError:
            raise ValueError(
                f"Jeu de sigma-points inconnu '{self.sigma_point_set}'. "
                f"Disponibles : {list(SigmaPointsSet.registry.keys())}"
            )

        return cls(dim_x=self.dim_x, param=self.param)


class NonLinear_UPKF(NonLinear_PKF):
    """Implementation of UPKF."""

    def __init__(self, sigma_point_set: str, param, sKey: Optional[int] = None, verbose: int = 0) -> None:
        super().__init__(param, sKey, verbose)

        self.sigma_point_set = sigma_point_set

        cfg = FilterConfig(
            sigma_point_set = self.sigma_point_set,
            dim_x           = self.param.dim_x,
            param           = self.param
        )

        self.sigma_point_set_obj = cfg.create_sigma_point_set()


    def create_sigma_point_set(self) -> SigmaPointsSet:
        try:
            cls = SigmaPointsSet.registry[self.sigma_point_set]
        except KeyError:
            raise ValueError(
                f"Jeu de sigma-points inconnu '{self.sigma_point_set}'. "
                f"Disponibles : {list(SigmaPointsSet.registry.keys())}"
            )

        return cls(dim_x=self.dim_x)

    def process_nonlinearfilter(self, N=None, data_generator=None):
        """
        Generator of UPKF filter using optional data generator.
        """
        if __debug__:
            if not ((isinstance(N, int) and N > 0) or N is None):
                raise ValueError("N must be None or a number >0")

        generator = data_generator if data_generator is not None else self._data_generation()
        
        # short-cuts
        Pz00, g, mQ, augmented = self.param._Pz00, self.param.g, self.param.mQ, self.param.augmented
        
        # for speed
        eye_dim_y = np.eye(self.dim_y)
        eye_dim_x = np.eye(self.dim_x)

        # The first
        ###################
        # parenthesis are used to flatten the list of two items
        k, (xkp1, ykp1) = next(generator) # parenthesis are used to flatten the list of two items

        # temp          = Pz00[0:self.dim_x, self.dim_x:] @ np.linalg.inv(Pz00[self.dim_x:, self.dim_x:])
        # Xkp1_update   = temp @ ykp1
        # PXXkp1_update = Pz00[0:self.dim_x, 0:self.dim_x] - temp @ Pz00[self.dim_x:, 0:self.dim_x]
        Xkp1_update       = xkp1 #z00[0:self.dim_x]
        PXXkp1_update     = Pz00[0:self.dim_x, 0:self.dim_x] 
        verdict, report = diagnose_covariance(PXXkp1_update)
        if not augmented:
            verdict, report = diagnose_covariance(PXXkp1_update)
            if not verdict:
                print(f'PXXkp1_update={PXXkp1_update}\nReport for PXXkp1_update - iteration k={k}:')
                print(report)
                input('attente')

        # Record data in the tracker
        Xkp1_predict = np.zeros((self.dim_x, 1))
        self._history.record(iter          = k,
                             xkp1          = xkp1.copy() if xkp1 is not None else None,
                             ykp1          = ykp1.copy(),
                             Xkp1_predict  = Xkp1_predict.copy(),
                             PXXkp1_predict= eye_dim_x,
                             ikp1          = np.zeros(shape=(self.dim_y)),
                             Skp1          = eye_dim_y,
                             Kkp1          = np.zeros(shape=(self.dim_x, self.dim_y)),
                             Xkp1_update   = Xkp1_update.copy(),
                             PXXkp1_update = PXXkp1_update.copy()
        )
        
        # last = self._history.last()
        # rich_show_fields(last, ["xkp1", "Xkp1_predict", "PXXkp1_predict", "ikp1", "Skp1", "Kkp1", "Xkp1_update", "PXXkp1_update"], title="Infos sélectionnées")
        # input('ATTENTE')

        yield k, xkp1, ykp1, Xkp1_predict, Xkp1_update

        ###################
        # The next ones
        
        accel_zero = np.zeros(shape=(self.dim_xy, 1))
        
        while N is None or k<N:
            # Sigma points
            sigma = self.sigma_point_set_obj._sigma_point(Xkp1_update, PXXkp1_update)
            sigma_propag = [g(np.vstack((e, ykp1)), accel_zero, self.dt) for e in sigma]  # here ykp1 still gives the previous : it is yk indeed!
            # print(f'sigma_propag[1]={sigma_propag[1]}')

            # Prediction
            Zkp1_predict               = np.sum(self.sigma_point_set_obj.Wm[:, None, None] * sigma_propag, axis=0)
            Xkp1_predict, Ykp1_predict = np.split(Zkp1_predict, [self.dim_x])
            Pkp1_predict = mQ.copy()
            for i in range(self.sigma_point_set_obj.nbSigmaPoint):
                diff = sigma_propag[i] - Zkp1_predict
                Pkp1_predict += self.sigma_point_set_obj.Wc[i] * np.outer(diff, diff)
            
            if not augmented:
                verdict, report = diagnose_covariance(Pkp1_predict)
                if not verdict:
                    print(f'ICI - Pkp1_predict={Pkp1_predict}\nReport - iteration k={k}:')
                    print(report)
                    input('attente')

            # Cutting Pkp1 into 4 blocks
            M_top, M_bottom                = np.vsplit(Pkp1_predict, [self.dim_x])
            PXXkp1_predict, PXYkp1_predict = np.hsplit(M_top,        [self.dim_x])
            PYXkp1_predict, PYYkp1_predict = np.hsplit(M_bottom,     [self.dim_x])

            # New data
            try:
                # parenthesis are used to flatten the list of two items
                k, (xkp1, ykp1) = next(generator)
            except StopIteration:
                return # we stop as the data generator is stopped itself

            # Updating
            ikp1        = ykp1 - Ykp1_predict
            Skp1        = PYYkp1_predict
            # Kalman gain - Version robuste du calcul
            c, low = cho_factor(Skp1)
            Kkp1 = PXYkp1_predict @ cho_solve((c, low), eye_dim_y)
            # print(f'Kkp1={Kkp1}')
            Xkp1_update   = Xkp1_predict   + Kkp1 @ ikp1
            PXXkp1_update = PXXkp1_predict - Kkp1 @ PXYkp1_predict.T - PXYkp1_predict @ Kkp1.T + Kkp1 @ Skp1 @ Kkp1.T
            
            if not augmented:
                verdict, report = diagnose_covariance(PXXkp1_update)
                if not verdict:
                    print(f'PXXkp1_update={PXXkp1_update}\nReport - iteration k={k}:')
                    print(report)
                    input('attente')
            
            # Forme de Joseph
            temp = np.vstack((eye_dim_x, -Kkp1.T))
            PXXkp1_update_Joseph = temp.T @ Pkp1_predict @ temp
            
            if not augmented:
                verdict, report = diagnose_covariance(PXXkp1_update_Joseph)
                if not verdict:
                    print(f'PXXkp1_update_Joseph={PXXkp1_update_Joseph}\nReport - iteration k={k}:')
                    print(report)
                    input('attente')
                
            # Record data in the tracker
            self._history.record(iter           = k,
                                 xkp1           = xkp1.copy() if xkp1 is not None else None,
                                 ykp1           = ykp1.copy(),
                                 Xkp1_predict   = Xkp1_predict.copy(),
                                 PXXkp1_predict = PXXkp1_predict.copy(),
                                 ikp1           = ikp1.copy(),
                                 Skp1           = Skp1.copy(),
                                 Kkp1           = Kkp1.copy(),
                                 Xkp1_update    = Xkp1_update.copy(),
                                 PXXkp1_update  = PXXkp1_update_Joseph.copy(), #PXXkp1_update.copy()
            )
            
            # Si on veut la forme robuste de la variance, on décommente
            PXXkp1_update = PXXkp1_update_Joseph

            # last = self._history.last()
            # rich_show_fields(last, ["xkp1", "Xkp1_predict", "PXXkp1_predict", "ikp1", "Skp1", "Kkp1", "Xkp1_update", "PXXkp1_update"], title="Infos sélectionnées")
            # input('ATTENTE')

            yield k, xkp1, ykp1, Xkp1_predict, Xkp1_update
