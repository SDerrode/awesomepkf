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

    def __init__(self, sigma_point_set_name: str, param, sKey: Optional[int] = None, verbose: int = 0) -> None:
        super().__init__(param, sKey, verbose)

        self.sigma_point_set_name = sigma_point_set_name

        cfg = FilterConfig(
            sigma_point_set_name = self.sigma_point_set_name,
            dim                  = 2*self.param.dim_xy,
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

    def process_nonlinearfilter(self, N=None, data_generator=None):
        """
        Generator of UPKF filter using optional data generator.
        """
        if __debug__:
            if not ((isinstance(N, int) and N > 0) or N is None):
                raise ValueError("N must be None or a number >0")

        generator = data_generator if data_generator is not None else self._data_generation()
        
        # short-cuts
        z00, Pz00, g, mQ, augmented = self.param._z00, self.param._Pz00, self.param.g, self.param.mQ, self.param.augmented
        
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

        ###################
        # The next ones

        z  = np.zeros(shape=(2*self.dim_xy, 1))
        Pa = np.zeros(shape=(2*self.dim_xy, 2*self.dim_xy))
        Pa[self.dim_xy:, self.dim_xy:] = mQ
        accel_zero = np.zeros(shape=(self.dim_xy, 1))
        while N is None or k<N:
            # Sigma points et leur propagation par g
            z[0:self.dim_x]           = Xkp1_update
            z[self.dim_x:self.dim_xy] = ykp1
            # print(f'z=\n{z}')
            Pa[0:self.dim_x, 0:self.dim_x] = PXXkp1_update
            # print(f'Pa=\n{Pa}')
            sigma = self.sigma_point_set_obj._sigma_point(z, Pa)
            # print(f'sigma=\n{sigma}')
            sigma_propag = [g(*np.split(spoint, [self.dim_xy]), self.dt) for spoint in sigma]  # here ykp1 still gives the previous : it is yk indeed!
            # print(f'sigma_propag[0]={sigma_propag[0]}')
            # print(f'sigma_propag[1]={sigma_propag[1]}')
            # exit(1)

            # Prediction
            Zkp1_predict               = np.sum(self.sigma_point_set_obj.Wm[:, None, None] * sigma_propag, axis=0)
            Xkp1_predict, Ykp1_predict = np.split(Zkp1_predict, [self.dim_x])
            Pkp1_predict = np.zeros(shape=(self.dim_xy, self.dim_xy))
            for i in range(self.sigma_point_set_obj.nbSigmaPoint):
                diff = sigma_propag[i] - Zkp1_predict
                Pkp1_predict += self.sigma_point_set_obj.Wc[i] * np.outer(diff, diff)
            
            if not augmented:
                # print('TUTUTUTU')
                verdict, report = diagnose_covariance(Pkp1_predict)
                # print(f'TUTUTUTU verdict={verdict}')
                if not verdict:
                    print(f'ICICICICI Pkp1_predict={Pkp1_predict}\nReport - iteration k={k}:')
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
            # Kalman gain - Version robuste du calcul
            try:
                c, low = cho_factor(Skp1)
                Kkp1   = PXYkp1_predict @ cho_solve((c, low), eye_dim_y)
            except np.linalg.LinAlgError as e:
                print(f'Skp1={Skp1}')
                input('ATTENTE')
            except ValueError as e:
                print("Erreur de valeur :", e)
                input('ATTENTE')
            # print(f'Kkp1={Kkp1}')
            Xkp1_update   = Xkp1_predict   + Kkp1 @ ikp1
            PXXkp1_update = PXXkp1_predict - Kkp1 @ PXYkp1_predict.T
            
            if not augmented:
                # print('TITITIITITIT')
                verdict, report = diagnose_covariance(PXXkp1_update)
                # print(f'verdict TITITIITITIT={verdict}')
                if not verdict:
                    print(f'PXXkp1_update={PXXkp1_update}\nReport - iteration k={k}:')
                    rich_show_fields(report, ["is_symmetric", "cholesky_ok", "is_psd", "near_singular", "ill_conditioned", "numerically_singular"], title="")
                    input('attente')
            
            # Forme de Joseph
            temp = np.vstack((eye_dim_x, -Kkp1.T))
            PXXkp1_update_Joseph = temp.T @ Pkp1_predict @ temp
            
            if not augmented:
                # print('TOTOTOTOTOO')
                verdict, report = diagnose_covariance(PXXkp1_update_Joseph)
                # print(f'verdict TOTOTOTOTOO={verdict}')
                if not verdict:
                    print(f'PXXkp1_update_Joseph={PXXkp1_update_Joseph}\nReport - iteration k={k}:')
                    rich_show_fields(report, ["is_symmetric", "cholesky_ok", "is_psd", "near_singular", "ill_conditioned", "numerically_singular"], title="")
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
            # rich_show_fields(last, ["iter", "xkp1", "Xkp1_predict", "PXXkp1_predict", "ikp1", "Skp1", "Kkp1", "Xkp1_update", "PXXkp1_update"], title="")
            # input('ATTENTE')

            yield k, xkp1, ykp1, Xkp1_predict, Xkp1_update
