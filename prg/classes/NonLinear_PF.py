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
from scipy.linalg import solve, cholesky, inv, eigh

from classes.NonLinear_PKF import NonLinear_PKF
from classes.PKF import PKFStep

# A few utils functions that are used several times
from others.utils import diagnose_covariance, rich_show_fields
from others.numerics import EPS_ABS


class NonLinear_PF(NonLinear_PKF):
    """Implementation of PF."""

    def __init__(
        self,
        param: ParamLinear | ParamNonLinear,
        nbParticles=300,
        resample_threshold=0.5,
        sKey=None,
        verbose=0,
    ):

        super().__init__(param, sKey, verbose)

        self.nbParticles = nbParticles
        self.resample_threshold = resample_threshold

    # # ==========================
    # # ESS + Resampling
    # # ==========================

    # ess = 1.0 / np.sum(weights**2)
    # if ess < self.resample_threshold * self.nbParticles:
    #     idx = systematic_resample(weights)
    #     Z_particles = Z_particles[idx]
    #     weights.fill(1.0 / self.nbParticles)
    # # print(f'ess={ess}')

    # =========================
    # RESAMPLING UNIFIÉ
    # =========================
    def resample(self, weights, rng, method="stratified"):
        N = self.nbParticles
        cumulative_sum = np.cumsum(weights)
        # print(f"cumulative_sum={cumulative_sum[0:4]}")
        cumulative_sum[-1] = 1.0

        if method == "multinomial":
            indexes = np.searchsorted(cumulative_sum, rng.random(N))
        elif method in ["systematic", "stratified"]:
            if method == "systematic":
                positions = (np.arange(N) + rng.random()) / N
            else:
                positions = (np.arange(N) + rng.random(N)) / N
            # print(f"positions={positions[0:4]}")

            indexes = np.zeros(N, dtype=int)
            i = j = 0
            while i < N:
                if positions[i] < cumulative_sum[j]:
                    indexes[i] = j
                    i += 1
                else:
                    j += 1
            # print(f"i={i}, j={j}")
            # input("ATTENTE - debut resample")
        elif method == "residual":
            indexes = []
            num_copies = np.floor(N * self.weights).astype(int)
            for i in range(N):
                indexes += [i] * num_copies[i]
            residual = self.weights - num_copies / N
            residual /= residual.sum()
            cumulative_sum_res = np.cumsum(residual)
            cumulative_sum_res[-1] = 1.0
            remaining = N - len(indexes)
            random_vals = rng.random(remaining)
            res_indexes = np.searchsorted(cumulative_sum_res, random_vals)
            indexes += list(res_indexes)
            indexes = np.array(indexes)
        else:
            raise ValueError(f"Unknown resampling method: {method}")

        return indexes

    def process_filter(self, N=None, data_generator=None):
        if __debug__:
            if not ((isinstance(N, int) and N > 0) or N is None):
                raise ValueError("N must be None or a number >0")

        generator = (
            data_generator if data_generator is not None else self._data_generation()
        )

        seed = 303
        # seed_seq: np.random.SeedSequence = np.random.SeedSequence(seed)
        # rng: np.random.Generator = np.random.default_rng(seed_seq)
        rng: np.random.Generator = np.random.default_rng(seed)

        # Additionnal short-cuts
        g, mQ, augmented = self.param.g, self.param.mQ, self.param.augmented

        # Les particules et leur poids
        # print("mz0=", self.mz0[: self.dim_x].flatten())
        # print("Pz0=", self.Pz0[: self.dim_x, : self.dim_x])
        particles_courant = rng.multivariate_normal(
            self.mz0[: self.dim_x].flatten(),
            self.Pz0[: self.dim_x, : self.dim_x],
            self.nbParticles,
        )
        particles_courant = particles_courant[..., np.newaxis]
        # print(particles_courant)
        # print("EHEOHE4", particles_courant.shape)
        particles_precedent = np.zeros_like(particles_courant)
        weights = np.full(self.nbParticles, 1.0 / self.nbParticles)
        # print(particles_precedent)
        # print(particles_precedent.shape)
        # print(weights)
        # exit(1)
        # print(particles_courant[:3])
        # input("ATTENTE __init__")

        Q = self.mQ[: self.dim_x, : self.dim_x]
        M = self.mQ[: self.dim_x, self.dim_x :]
        MT = M.T
        R = self.mQ[self.dim_x :, self.dim_x :]
        R_inv = inv(R)
        sign, logdet = np.linalg.slogdet(R)
        log_norm_const = -0.5 * (self.dim_y * np.log(2 * np.pi) + logdet)

        # for speed
        # eye_dim_y = np.eye(self.dim_y)
        # eye_dim_x = np.eye(self.dim_x)

        # The first
        ##################################################################################################
        step = self._firstEstimate(generator)
        if step.xkp1 is None:  # Il n'y a pas de VT
            self.ground_truth = False

        # print(f"kp1={step.k}")
        # print(f"step.Xkp1_predict={step.Xkp1_predict}")
        # print(f"step.Xkp1_update={step.Xkp1_update}")
        # input("ATTENTE - kp1 = 0")

        yield step.k, step.xkp1, step.ykp1, step.Xkp1_predict, step.Xkp1_update

        ###################
        # The next ones

        while N is None or step.k < N:

            # Maj des particules
            particles_precedent = particles_courant.copy()
            # print(particles_courant[:3])
            # print(particles_precedent[:3])
            # input("ATTENTE - particles_precedent")

            # =========================
            # PREDICTION
            # =========================

            Xkp1_predict = np.mean(particles_courant, axis=0)
            # print(f"Xkp1_predict={Xkp1_predict}")
            diff = particles_courant - Xkp1_predict
            # print(f"diff={diff}")
            # Pkp1_predict = diff.T @ diff / (self.nbParticles - 1)
            PXXkp1_predict = np.einsum("tik,tjk->ij", diff, diff) / (
                self.nbParticles - 1
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
                    for p in particles_precedent
                ]
            )
            # print(f"muxy={muxy[:3]}")
            # input("ATTENTE")

            # Calculer innovations pour toutes les particules pour la mise à jour des poids
            innovations = new_ykp1 - muxy[:, self.dim_x :, :]
            tmp = np.matmul(R_inv, innovations)  # (300,2,1)
            quad = np.matmul(
                innovations.transpose(0, 2, 1), tmp  # (300,1,2)  # (300,2,1)
            )
            exponents = -0.5 * quad.squeeze()
            # print(f"exponents = {exponents.shape}")
            # print(f"exponents = {exponents[:3]}")
            # input("ATTENTE titut")

            # Log-weights pour stabilité
            log_weights = np.log(weights) + exponents + log_norm_const
            log_weights -= np.max(log_weights)
            weights = np.exp(log_weights)
            weights /= np.sum(weights)
            # print(f"weights = {weights[:3]}")

            # Mise à jour des particules
            for i in range(self.nbParticles):
                mu_prime_x = muxy[i, : self.dim_x] + M @ R_inv @ (
                    new_ykp1 - muxy[i, self.dim_x :]
                )
                P_prime_x = Q - M @ R_inv @ MT
                # print("mu_prime_x=", mu_prime_x)
                # print("P_prime_x=", P_prime_x)

                # Forcer PSD pour stabilité
                eigvals, eigvecs = np.linalg.eigh(P_prime_x)
                eigvals[eigvals < EPS_ABS] = EPS_ABS
                P_prime_x = eigvecs @ np.diag(eigvals) @ eigvecs.T
                # print("P_prime_x=", P_prime_x)

                # Tirage stable via Cholesky
                # Paramètres de la loi recherchée pour le tirage sont :
                L = cholesky(P_prime_x, lower=True)
                # print(f"L = {L}")
                particles_courant[i] = mu_prime_x + L @ rng.standard_normal(
                    self.dim_x
                ).reshape(-1, 1)
                # print(f"particles_courant[i]={particles_courant[i]}")
                # input("ATTENTE titut")

            particles_courant_temp = particles_courant.squeeze(-1)
            X = particles_courant_temp[:, : self.dim_x]
            Xkp1_update = np.average(X, axis=0, weights=weights)[:, None]
            # print(f"Xkp1_update={Xkp1_update}")
            dx = X - Xkp1_update.T  # (300,2)
            PXXkp1_update = (weights[:, None] * dx).T @ dx  # (2,2)
            # print(f"PXXkp1_update={PXXkp1_update}")

            step = PKFStep(
                k=new_k,
                xkp1=new_xkp1.copy() if new_xkp1 is not None else None,
                ykp1=new_ykp1.copy(),
                Xkp1_predict=Xkp1_predict.copy(),
                PXXkp1_predict=PXXkp1_predict.copy(),
                ikp1=self.zeros_dim_y_1.copy(),
                Skp1=self.eye_dim_y.copy(),
                Kkp1=self.zeros_dim_x_y.copy(),
                Xkp1_update=Xkp1_update.copy(),
                # PXXkp1_update  = PXXkp1_update.copy(),
                PXXkp1_update=PXXkp1_update.copy(),
            )

            # print(f"kp1={step.k}")
            # print(f"step.Xkp1_predict={step.Xkp1_predict}")
            # print(f"step.Xkp1_update={step.Xkp1_update}")
            # input(f"ATTENTE - kp1 = {step.k}")

            # Ré-échantillonnage des particules
            indexes = self.resample(weights, rng)
            particles_courant = particles_courant[indexes]
            weights.fill(1.0 / self.nbParticles)

            # print(f"particles_courant={particles_courant[:3]}")
            # input("ATTENTE - prg principal")

            # Sauvegarde dans l'historique
            self.history.record(step)

            if self.verbose > 1:
                rich_show_fields(step, title=f"Step {step.k} Update")

            # print("k=", step.k)
            # input("ATTENTE")

            yield new_k, new_xkp1, new_ykp1, step.Xkp1_predict, step.Xkp1_update
