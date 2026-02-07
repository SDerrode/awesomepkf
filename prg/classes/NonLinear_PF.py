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

from filterpy.monte_carlo import systematic_resample

from scipy.linalg import cho_factor, cho_solve

from classes.NonLinear_PKF import NonLinear_PKF
# A few utils functions that are used several times
from others.utils import check_consistency, diagnose_covariance#, check_equality

class NonLinear_PF(NonLinear_PKF):
    """Implementation of PF."""

    def __init__(self, param,  nbParticles=1000, resample_threshold=0.5, sKey=None, save_pickle=False, verbose=0):
        super().__init__(param, sKey, save_pickle, verbose)
        self.nbParticles        = nbParticles
        self.resample_threshold = resample_threshold

    # ======================================================
    # Utilities
    # ======================================================

    @staticmethod
    def weighted_mean(X, w):
        return np.average(X, axis=0, weights=w)

    @staticmethod
    def weighted_cov(X, w, mean):
        P = np.zeros((X.shape[1], X.shape[1]))
        for i in range(X.shape[0]):
            dx = (X[i] - mean).reshape(-1, 1)
            P += w[i] * (dx @ dx.T)
        return P

    def process_nonlinearfilter(self, N=None, data_generator=None):
        if __debug__:
            if not ((isinstance(N, int) and N > 0) or N is None):
                raise ValueError("N must be None or a number >0")

        generator = data_generator if data_generator is not None else self._data_generation()
        
        # short-cuts
        g, mQ = self.param.g, self.param.mQ

        # ==========================
        # Init
        # ==========================
        
        # Initial particles: joint Gaussian prior
        Z_particles = np.random.multivariate_normal(
            mean = np.zeros(self.dim_xy),
            cov  = self.param.Pz00,
            size = self.nbParticles
        )
        weights     = np.ones(self.nbParticles) / self.nbParticles

        # # ==========================
        # # Update estimate
        # # ==========================
        # Xkp1_update = self.weighted_mean(
        #     Z_particles[:, :self.dim_x], weights
        # ).reshape(-1, 1)
        # PXXkp1_update = self.weighted_cov(
        #     Z_particles[:, :self.dim_x], weights, Xkp1_update.ravel()
        # )
        # verdict, report = diagnose_covariance(PXXkp1_update)
        # if verdict is not None:
        #     print(f'PXXkp1_update={PXXkp1_update}\nReport for PXXkp1_update - iteration k={k}:')
        #     print(report)
        #     input('attente')

        # if self.save_pickle and self._history is not None:
        #     self._history.record(iter          = k,
        #                          xkp1          = xkp1.copy() if xkp1 is not None else None,
        #                          ykp1          = ykp1.copy(),
        #                          Xkp1_predict  = Xkp1_predict.copy(),
        #                          PXXkp1_predict= PXXkp1_predict.copy(),
        #                          Xkp1_update   = Xkp1_update.copy(),
        #                          PXXkp1_update = PXXkp1_update.copy(),
        #                          ESS           = self.nbParticles)

        # yield k, xkp1, ykp1, Xkp1_predict, Xkp1_update

        ##################################################################################################@
        # The next ones
        
        # accel_zero = np.zeros(shape=(self.dim_xy, 1))
        k=0
        while N is None or k < N:

            # ==========================
            # Predict
            # ==========================

            noise = np.random.multivariate_normal(
                mean = np.zeros(self.dim_xy),
                cov  = mQ,
                size = self.nbParticles
            )

            for i in range(self.nbParticles):
                Z_particles[i] = g(Z_particles[i].reshape(-1, 1), noise[i].reshape(-1, 1), self.dt).ravel()

            Zkp1_predict = self.weighted_mean(Z_particles, weights).reshape(-1, 1)
            Xkp1_predict, Ykp1_predict = np.split(Zkp1_predict, [self.dim_x])
            # print(f'Zkp1_predict={Zkp1_predict}')
            # print(f'Xkp1_predict={Xkp1_predict}')
            
            Pkp1_predict = self.weighted_cov(Z_particles, weights, Zkp1_predict.ravel())
            # print(f'Pkp1_predict={Pkp1_predict}'
            # Matrice singulière : ne pas tester
            # verdict, report = diagnose_covariance(Pkp1_predict)
            # if verdict is not None:
            #     print(f'Pkp1_predict={Pkp1_predict}\nReport for Pkp1_predict - iteration k={k}:')
            #     print(report)
            #     input('attente')

            # Cutting Pkp1 into 4 blocks
            M_top, M_bottom                = np.vsplit(Pkp1_predict, [self.dim_x])
            PXXkp1_predict, PXYkp1_predict = np.hsplit(M_top,        [self.dim_x])
            PYXkp1_predict, PYYkp1_predict = np.hsplit(M_bottom,     [self.dim_x])
            verdict, report = diagnose_covariance(PXXkp1_predict)
            if verdict is not None:
                print(f'PXXkp1_predict={PXXkp1_predict}\nReport for PXXkp1_predict - iteration k={k}:')
                print(report)
                input('attente')
            verdict, report = diagnose_covariance(PYYkp1_predict)
            if verdict is not None:
                print(f'PYYkp1_predict={PYYkp1_predict}\nReport for PYYkp1_predict - iteration k={k}:')
                print(report)
                input('attente')
            # print(f'PXXkp1_predict={PXXkp1_predict}')
            # print(f'PYYkp1_predict={PYYkp1_predict}')

            # ==========================
            # Observation
            # ==========================
            try:
                k, (xkp1, ykp1) = next(generator)
            except StopIteration:
                return # we stop as the data generator is stopped itself

            y_obs = ykp1.ravel()

            # ==========================
            # Update / weighting
            # =========================
            
            # Version robuste du calcul d'inversion de la cov d'innovation
            c, low = cho_factor(PYYkp1_predict)
            S_inv  = cho_solve((c, low), np.eye(self.dim_y))
            det_S  = np.linalg.det(PYYkp1_predict)
            norm   = 1.0 / np.sqrt((2 * np.pi) ** self.dim_y * det_S)

            for i in range(self.nbParticles):
                innov = y_obs - Z_particles[i, self.dim_x:]
                weights[i] *= norm * np.exp( -0.5 * innov.T @ S_inv @ innov )

            weights += 1e-300            # avoid round-off to zero
            weights /= np.sum(weights)   # normalize

            # ==========================
            # ESS + Resampling
            # ==========================

            ess = 1.0 / np.sum(weights ** 2)
            if ess < self.resample_threshold * self.nbParticles:
                idx         = systematic_resample(weights)
                Z_particles = Z_particles[idx]
                weights.fill(1.0 / self.nbParticles)
            # print(f'ess={ess}')

            # ==========================
            # Update estimate
            # ==========================

            Xkp1_update = self.weighted_mean(Z_particles[:, :self.dim_x], weights).reshape(-1, 1)
            PXXkp1_update = self.weighted_cov(Z_particles[:, :self.dim_x], weights, Xkp1_update.ravel())
            verdict, report = diagnose_covariance(PXXkp1_update)
            if verdict is not None:
                print(f'PXXkp1_update={PXXkp1_update}\nReport for PXXkp1_update - iteration k={k}:')
                print(report)
                input('attente')

            if self.save_pickle and self._history is not None:
                self._history.record(iter           = k,
                                     xkp1           = xkp1.copy() if xkp1 is not None else None,
                                     ykp1           = ykp1.copy(),
                                     Xkp1_predict   = Xkp1_predict,
                                     PXXkp1_predict = PXXkp1_predict.copy(),
                                     Xkp1_update    = Xkp1_update.copy(),
                                     PXXkp1_update  = PXXkp1_update.copy(),
                                     ESS            = ess)
            

            yield k, xkp1, ykp1, Xkp1_predict, Xkp1_update
