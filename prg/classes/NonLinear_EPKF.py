#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
####################################################################
Extended Pairwise Kalman filter (EPKF) implementation
####################################################################
"""

from __future__ import annotations
from typing import Generator, Optional
from scipy.linalg import LinAlgError
import numpy as np

from rich import print

from .NonLinear_PKF import NonLinear_PKF

class NonLinear_EPKF(NonLinear_PKF):
    """Implementation of EPKF."""

    def __init__(self, param: ParamNonLinear, ell: int = 1, sKey: Optional[int] = None, verbose: int = 0) -> None:
        
        super().__init__(param, sKey, verbose)
        
        if __debug__:
            assert ell>0 , "verbose must be 1 (classical EPKF) or above for IEPKF"
        self.ell = ell

    def process_filter(self, N: Optional[int] = None, data_generator: Optional[Generator] = None) \
                    -> Generator[tuple[int, np.ndarray, np.ndarray, np.ndarray, np.ndarray], None, None]:
        """
        Generator of EPKF filter using optional data generator.
        """
        if __debug__:
            if not ((isinstance(N, int) and N > 0) or N is None):
                raise ValueError("N must be None or a number >0")

        generator = data_generator if data_generator is not None else self._data_generation()
        
        # Additionnal short-cut
        self.jg = self.param.jacobiens_g

        # The first
        ##################################################################################################
        step = self._firstEstimate(generator)
        # print(f'step {step.k} - {hex(id(step))} - final =')
        # print(step)
        # input('ATTENTE')

        if step.xkp1 is None: # Il n'y a pas de VT (fro error computing and plotting)
            self.ground_truth = False

        yield step.k, step.xkp1, step.ykp1, step.Xkp1_predict, step.Xkp1_update

        ##################################################################################################@
        # The next ones
        accel_xy_xy = self.zeros_dim_xy_xy.copy()
        z_iterated  = np.zeros(shape=(self.dim_xy, 1))
        while N is None or step.k<N:

            # New data is arriving ##################################
            try:
                new_k, new_xkp1, new_ykp1 = next(generator)
            except StopIteration:
                return # we stop as the data generator is stopped itself
            
            # Initialisation des boucles pour l'Iterated EPKF
            z_iterated.fill(0.)
            z_iterated[:self.dim_x] = step.Xkp1_update
            z_iterated[self.dim_x:] = step.ykp1
            PXX_iterated            = step.PXXkp1_update
            
            store = False
            for l in range(self.ell):
                # print(f'iteration ell : {l}')
                
                Zl_predict = self.g(z_iterated, self.zeros_dim_xy_1, self.dt)

                Anl, Bnl = self.jg(z_iterated, self.zeros_dim_xy_1, self.dt)
                if Anl.shape != (self.dim_xy, self.dim_xy) or Bnl.shape != (self.dim_xy, self.dim_xy):
                    raise ValueError(f"Jacobian returned matrices of wrong shape: Anl={An.shape}, Bnl={Bn.shape}")
                accel_xy_xy[:self.dim_x, :self.dim_x] = PXX_iterated
                Pl_predict = Anl @ accel_xy_xy @ Anl.T + Bnl @ self.mQ @ Bnl.T
                self._test_CovMatrix(Pl_predict, step.k)
                
                # Updating ##############################################
                if l == self.ell-1: # dernier tour de boucle
                    store = True
                try:
                    step = self._nextUpdating(new_k, new_xkp1, new_ykp1, Zl_predict, Pl_predict, store)
                    # print(f"  iteration {l}: {hex(id(step))}")
                except LinAlgError:
                    self.logger.error(f"Step {new_k}, {l}: LinAlgError during update.")
                    raise
                # print(f'step {step.k} - {l} = \n')
                # print(step)
                # input('ATTENTE step')
            
                # print(f'step.Xkp1_update  ={step.Xkp1_update}')
                # print(f'step.PXXkp1_update={step.PXXkp1_update}')
            
                if l<=self.ell-1:
                    # MAJ pour la prochaine iteration
                    z_iterated[:self.dim_x] = step.Xkp1_update
                    PXX_iterated            = step.PXXkp1_update

            # print(f'step {step.k} - {hex(id(step))} - final =')
            # print(step)
            # input('ATTENTE step final')

            yield step.k, step.xkp1, step.ykp1, step.Xkp1_predict, step.Xkp1_update