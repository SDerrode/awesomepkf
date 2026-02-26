#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
####################################################################
Parwise Particle Filter implementation
####################################################################
"""

from __future__ import annotations

# Stdlib
from dataclasses import replace
from typing import Generator

# Third-party
import numpy as np
from rich import print
from scipy.linalg import cholesky, inv

# Local
from classes.PKF import PKF
from classes.PKF import PKFStep
from others.numerics import EPS_ABS
from others.utils import rich_show_fields, symmetrize, check_eigvals


class NonLinear_PPF(PKF):
    """Implementation of PPF."""

    def __init__(
        self,
        param: ParamLinear | ParamNonLinear,
        nbParticles=300,
        resample_threshold=0.7,
        resample_method="stratified",
        sKey=None,
        verbose=0,
    ):

        super().__init__(param, sKey, verbose)

        self.nbParticles = nbParticles
        self.resample_threshold = resample_threshold
        self.resample_method = resample_method

        # dictionnaire des constantes
        self._cached = {}

    # =========================
    # RESAMPLING UNIFIÉ
    # =========================
    def resample(self, weights, method="stratified"):
        N = self.nbParticles
        cumulative_sum = np.cumsum(weights)
        cumulative_sum[-1] = 1.0

        if method == "multinomial":
            indexes = np.searchsorted(cumulative_sum, self._seed_gen.rng.random(N))
        elif method in ["systematic", "stratified"]:
            if method == "systematic":
                positions = (np.arange(N) + self._seed_gen.rng.random()) / N
            else:
                positions = (np.arange(N) + self._seed_gen.rng.random(N)) / N

            indexes = np.zeros(N, dtype=int)
            i = j = 0
            while i < N:
                if positions[i] < cumulative_sum[j]:
                    indexes[i] = j
                    i += 1
                else:
                    j += 1
        elif method == "residual":
            indexes = []
            num_copies = np.floor(N * weights).astype(int)
            for i in range(N):
                indexes += [i] * num_copies[i]
            residual = weights - num_copies / N
            # residual /= residual.sum()
            residual_sum = residual.sum()
            if residual_sum > EPS_ABS:
                residual /= residual_sum
            else:
                residual = np.ones(N) / N
            cumulative_sum_res = np.cumsum(residual)
            cumulative_sum_res[-1] = 1.0
            remaining = N - len(indexes)
            random_vals = self._seed_gen.rng.random(remaining)
            res_indexes = np.searchsorted(cumulative_sum_res, random_vals)
            indexes += list(res_indexes)
            indexes = np.array(indexes)
        else:
            raise ValueError(f"Unknown resampling method: {method}")

        return indexes

    def _precompute(self):
        Q = self.mQ[: self.dim_x, : self.dim_x]
        M = self.mQ[: self.dim_x, self.dim_x :]
        R = self.mQ[self.dim_x :, self.dim_x :]
        R_inv = inv(R)
        P_prime_x_base = Q - M @ R_inv @ M.T

        # Stabilisation PSD + Cholesky — calculés une seule fois
        eigvals, eigvecs = np.linalg.eigh(P_prime_x_base)
        check_eigvals(eigvals)
        P_prime_x = eigvecs @ np.diag(eigvals) @ eigvecs.T

        self._cached = dict(
            R_inv=R_inv,
            MRinv=M @ R_inv,
            L=cholesky(P_prime_x, lower=True),
            log_norm_const=-0.5
            * (self.dim_y * np.log(2 * np.pi) + np.linalg.slogdet(R)[1]),
        )

    def process_filter(self, N=None, data_generator=None):

        self._validate_N(N)

        generator = (
            data_generator if data_generator is not None else self._data_generation()
        )

        # Additional short-cuts
        # augmented = self.param.augmented

        # precalcul des constantes
        self._precompute()

        # Les particules et leur poids
        particles_current = self._seed_gen.rng.multivariate_normal(
            self.mz0[: self.dim_x].flatten(),
            self.Pz0[: self.dim_x, : self.dim_x],
            self.nbParticles,
        )
        particles_current = particles_current[..., np.newaxis]
        # particles_previous = np.zeros_like(particles_current)
        weights = np.full(self.nbParticles, 1.0 / self.nbParticles)

        # The first
        ##################################################################################################
        step_big = self._firstEstimate(generator)
        # On enleve les 3 champs qui n'ont pas d'interet dans cet algo
        step = replace(step_big, ikp1=None, Skp1=None, Kkp1=None)

        if step.xkp1 is None:  # Il n'y a pas de VT
            self.ground_truth = False

        yield step.k, step.xkp1, step.ykp1, step.Xkp1_predict, step.Xkp1_update

        ###################
        # The next ones

        while N is None or step.k < N:

            # Maj des particules
            particles_previous = particles_current.copy()

            # =========================
            # PREDICTION
            # =========================

            Xkp1_predict = np.mean(particles_current, axis=0)
            # print(f"Xkp1_predict={Xkp1_predict}")
            diff = particles_current - Xkp1_predict
            # PXXkp1_predict n'est pas utilisé
            PXXkp1_predict = symmetrize(
                np.einsum("tik,tjk->ij", diff, diff) / (self.nbParticles - 1)
            )
            # print(f"PXXkp1_predict={PXXkp1_predict}")
            self._test_CovMatrix(PXXkp1_predict, step.k)

            # New data is arriving ##################################
            try:
                new_k, new_xkp1, new_ykp1 = next(generator)
            except StopIteration:
                return  # we stop as the data generator is stopped itself

            # Updating ##############################################

            # tableau des moyennes x et y
            muxy = np.array(
                [
                    self.g(
                        np.vstack(
                            [p, step.ykp1]
                        ),  # ici step.ykp1 désigne le y précédent
                        self.zeros_dim_xy_1,
                        self.dt,
                    )
                    for p in particles_previous
                ]
            )

            # Calculer innovations pour toutes les particules pour la mise à jour des poids
            innovations = new_ykp1 - muxy[:, self.dim_x :, :]
            tmp = np.matmul(self._cached["R_inv"], innovations)  # (300,2,1)
            quad = np.matmul(
                innovations.transpose(0, 2, 1), tmp  # (300,1,2)  # (300,2,1)
            )
            exponents = -0.5 * quad.reshape(self.nbParticles)

            # Log-weights pour stabilité
            log_weights = (
                np.log(np.maximum(weights, EPS_ABS))
                + exponents
                + self._cached["log_norm_const"]
            )
            log_weights -= np.max(log_weights)
            weights = np.exp(log_weights)
            weights /= np.sum(weights)
            if __debug__:
                assert not np.any(
                    np.isnan(weights)
                ), f"NaN dans les poids au step {new_k}"
                assert np.isclose(
                    weights.sum(), 1.0, atol=1e-6
                ), f"Poids non normalisés : {weights.sum()}"

            # Mise à jour des particules — entièrement vectorisée
            # innovations_all = new_ykp1 - muxy[:, self.dim_x :, :]  # (N, dim_y, 1)
            mu_prime_x_all = (
                muxy[:, : self.dim_x, :] + self._cached["MRinv"] @ innovations
            )  # (N, dim_x, 1)
            noise = self._seed_gen.rng.standard_normal(
                (self.nbParticles, self.dim_x, 1)
            )
            particles_current = mu_prime_x_all + np.einsum(
                "ij,njk->nik", self._cached["L"], noise
            )

            particles_current_temp = particles_current.squeeze(-1)
            Xkp1_update = np.average(particles_current_temp, axis=0, weights=weights)[
                :, None
            ]
            # print(f"Xkp1_update={Xkp1_update}")
            dx = particles_current_temp - Xkp1_update.T
            PXXkp1_update = symmetrize((weights[:, None] * dx).T @ dx)
            # print(f"PXXkp1_update={PXXkp1_update}")

            step = PKFStep(
                k=new_k,
                xkp1=new_xkp1.copy() if new_xkp1 is not None else None,
                ykp1=new_ykp1.copy(),
                Xkp1_predict=Xkp1_predict.copy(),
                PXXkp1_predict=PXXkp1_predict.copy(),
                ikp1=None,
                Skp1=None,
                Kkp1=None,
                Xkp1_update=Xkp1_update.copy(),
                # PXXkp1_update  = PXXkp1_update.copy(),
                PXXkp1_update=PXXkp1_update.copy(),
            )

            # Ré-échantillonnage des particules
            ess = 1.0 / np.sum(weights**2)
            if ess < self.resample_threshold * self.nbParticles:
                indexes = self.resample(weights, self.resample_method)
                particles_current = particles_current[indexes]
                weights.fill(1.0 / self.nbParticles)

            if __debug__:
                assert not np.any(
                    np.isnan(particles_current)
                ), f"NaN dans les particules au step {new_k}"

            # Sauvegarde dans l'historique
            self.history.record(step)

            if self.verbose > 1:
                rich_show_fields(step, title=f"Step {step.k} Update")

            yield new_k, new_xkp1, new_ykp1, step.Xkp1_predict, step.Xkp1_update
