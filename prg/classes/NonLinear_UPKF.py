#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
####################################################################
Unscented Pairwise Kalman filter (UPKF) implementation
####################################################################
"""

from __future__ import annotations  # Annotations de type
from typing import Generator, Optional
import numpy as np  # Utilisé partout
from scipy.linalg import LinAlgError  # Utilisé dans le try/except
from classes.PKF import PKF  # Classe parente
from classes.SigmaPointsSet import SigmaPointsSet  # Utilisé dans FilterConfig
from others.utils import symmetrize


class NonLinear_UPKF(PKF):
    """Implementation of UPKF."""

    def __init__(
        self,
        param: ParamLinear | ParamNonLinear,
        sigmaSet: str,
        sKey: Optional[int] = None,
        verbose: int = 0,
    ):
        super().__init__(param, sKey, verbose)

        # Dans NonLinear_UPKF.__init__, à la place de FilterConfig
        try:
            cls = SigmaPointsSet.registry[sigmaSet]
        except KeyError:
            raise ValueError(
                f"Jeu de sigma-points inconnu '{sigmaSet}'. "
                f"Disponibles : {list(SigmaPointsSet.registry.keys())}"
            )
        self.sigma_point_set_obj = cls(dim=2 * self.dim_xy, param=self.param)

    def process_filter(
        self,
        N: Optional[int] = None,
        data_generator: Optional[
            Generator[tuple[int, np.ndarray, np.ndarray], None, None]
        ] = None,
    ) -> Generator[
        tuple[int, np.ndarray, np.ndarray, np.ndarray, np.ndarray], None, None
    ]:
        """
        Generator of UPKF filter using optional data generator.
        """

        self._validate_N(N)

        generator = (
            data_generator if data_generator is not None else self._data_generation()
        )

        # The first
        ##################################################################################################
        step = self._firstEstimate(generator)
        if step.xkp1 is None:  # Il n'y a pas de VT
            self.ground_truth = False

        yield step.k, step.xkp1, step.ykp1, step.Xkp1_predict, step.Xkp1_update

        ###################
        # The next ones
        Xkp1_update_augmented = np.zeros((2 * self.dim_xy, 1))
        Pa_base = np.zeros((2 * self.dim_xy, 2 * self.dim_xy))
        Pa_base[self.dim_xy :, self.dim_xy :] = self.mQ
        Pkp1_predict = self.zeros_dim_xy_xy.copy()

        while N is None or step.k < N:

            # Sigma points et leur propagation par g
            Xkp1_update_augmented[: self.dim_x] = step.Xkp1_update
            Xkp1_update_augmented[self.dim_x : self.dim_xy] = step.ykp1
            Pa = Pa_base.copy()  # copier Pa_base plutôt que recréer la matrice entière
            Pa[: self.dim_x, : self.dim_x] = step.PXXkp1_update
            sigma = self.sigma_point_set_obj._sigma_point(Xkp1_update_augmented, Pa)
            # here ykp1 still gives the previous : it is yk indeed!
            sigma_propag = [
                self.g(*np.split(spoint, [self.dim_xy]), self.dt) for spoint in sigma
            ]

            # Predicting ############################################
            Zkp1_predict = np.sum(
                self.sigma_point_set_obj.Wm[:, None, None] * sigma_propag, axis=0
            )

            # Remise à 0
            Pkp1_predict.fill(0.0)
            diffs = np.array(sigma_propag) - Zkp1_predict  # (n, dim, 1)
            Pkp1_predict = symmetrize(
                np.einsum("i,ijk,ilk->jl", self.sigma_point_set_obj.Wc, diffs, diffs)
            )
            self._test_CovMatrix(Pkp1_predict, step.k)

            # New data is arriving ##################################
            try:
                new_k, new_xkp1, new_ykp1 = next(generator)
            except StopIteration:
                return  # we stop as the data generator is stopped itself

            # Updating ##############################################
            try:
                step = self._nextUpdating(
                    new_k, new_xkp1, new_ykp1, Zkp1_predict, Pkp1_predict
                )
            except LinAlgError:
                self.logger.error(f"Step {new_k}: LinAlgError during update")
                raise

            yield step.k, step.xkp1, step.ykp1, step.Xkp1_predict, step.Xkp1_update
