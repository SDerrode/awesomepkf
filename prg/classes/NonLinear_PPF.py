#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
####################################################################
Pairwise Particle Filter implementation
####################################################################
"""

from __future__ import annotations

# Stdlib
from dataclasses import replace
from typing import Generator, Optional

# Third-party
import numpy as np
from rich import print
from scipy.linalg import cholesky, cho_factor, cho_solve

# Local
from prg.classes.SeedGenerator import SeedGenerator
from prg.classes.PKF import PKF, PKFStep
from prg.utils.numerics import EPS_ABS
from prg.utils.utils import rich_show_fields
from prg.classes.MatrixDiagnostics import CovarianceMatrix, InvertibleMatrix
from prg.utils.exceptions import (
    CovarianceError,
    FilterError,
    InvertibilityError,
    NumericalError,
    ParamError,
    StepValidationError,
)

import logging

logger = logging.getLogger(__name__)

__all__ = ["NonLinear_PPF"]


class NonLinear_PPF(PKF):
    """Filtre Particulaire Non Linéaire (Nonlinear Pairwise Particle Filter).

    Implémente un filtre particulaire pour les systèmes non linéaires à bruit
    additif gaussien, avec support optionnel des bruits corrélés (M ≠ 0) via
    la matrice de covariance jointe ``mQ``.

    Le modèle d'état considéré est :

    .. math::

        (X_n,Y_n)  = g_x(X_{n-1}, Y_{n-1}, V^x_n, V^y_n)

    avec :

    .. math::

        \\begin{pmatrix} V^x_n \\\\ V^y_n \\end{pmatrix}
        \\sim \\mathcal{N}\\!\\left(0,\\,
        \\begin{pmatrix} Q & M \\\\ M^\\top & R \\end{pmatrix}\\right)


    Parameters
    ----------
    param : ParamLinear | ParamNonLinear
        Paramètres du modèle (dimensions, matrices de bruit, conditions
        initiales, etc.).
    nbParticles : int, optional
        Nombre de particules. Par défaut 300.
    resample_threshold : float, optional
        Seuil relatif sur l'ESS déclenchant le rééchantillonnage,
        exprimé en fraction de ``nbParticles``. Par défaut 0.5.
    resample_method : str, optional
        Méthode de rééchantillonnage parmi ``'multinomial'``,
        ``'systematic'``, ``'stratified'`` (défaut) et ``'residual'``.
    sKey : optional
        Graine pour le générateur de nombres aléatoires.
    verbose : int, optional
        Niveau de verbosité (0 = silencieux). Par défaut 0.

    Attributes
    ----------
    nbParticles : int
        Nombre de particules.
    resample_threshold : float
        Seuil de rééchantillonnage.
    resample_method : str
        Méthode de rééchantillonnage.
    _cached : dict
        Constantes précalculées par ``_precompute()`` : ``R``, ``R_inv``,
        ``MRinv``, ``L`` (Cholesky de P'_x), ``log_norm_const``.
    """

    def __init__(
        self,
        param,
        nbParticles: int = 300,
        resample_threshold: float = 0.5,
        resample_method: str = "stratified",
        sKey=None,
        verbose: int = 0,
    ) -> None:
        super().__init__(param, sKey, verbose)

        self.nbParticles = nbParticles
        self.resample_threshold = resample_threshold
        self.resample_method = resample_method

        # Random number generator
        self.__randParticles = SeedGenerator()

        # Dictionnaire des constantes précalculées
        self._cached: dict = {}

    # =========================
    # RESAMPLING UNIFIÉ
    # =========================
    def resample(self, weights: np.ndarray, method: str = "stratified") -> np.ndarray:
        """Rééchantillonne les particules selon les poids normalisés.

        Parameters
        ----------
        weights : np.ndarray, shape (N,)
            Poids normalisés des particules (doivent sommer à 1).
        method : str, optional
            Méthode de rééchantillonnage :

            - ``'multinomial'``  : tirage multinomial indépendant.
            - ``'systematic'``   : rééchantillonnage systématique (un seul
              tirage aléatoire uniforme).
            - ``'stratified'``   : rééchantillonnage stratifié (un tirage
              par strate). Méthode par défaut.
            - ``'residual'``     : rééchantillonnage résiduel (copies
              déterministes + résidu stochastique).

        Returns
        -------
        indexes : np.ndarray of int, shape (N,)
            Indices des particules sélectionnées.

        Raises
        ------
        ParamError
            Si ``method`` n'est pas l'une des valeurs admises.
        """
        N = self.nbParticles
        cumulative_sum = np.cumsum(weights)
        cumulative_sum[-1] = 1.0

        if method == "multinomial":
            indexes = np.searchsorted(
                cumulative_sum, self.__randParticles.rng.random(N)
            )
        elif method in ["systematic", "stratified"]:
            if method == "systematic":
                positions = (np.arange(N) + self.__randParticles.rng.random()) / N
            else:
                positions = (np.arange(N) + self.__randParticles.rng.random(N)) / N

            indexes = np.zeros(N, dtype=int)
            i = j = 0
            while i < N:
                if positions[i] < cumulative_sum[j]:
                    indexes[i] = j
                    i += 1
                else:
                    j += 1

        elif method == "residual":
            indexes: list[int] = []
            num_copies = np.floor(N * weights).astype(int)
            for i in range(N):
                indexes += [i] * num_copies[i]
            residual = weights - num_copies / N
            residual_sum = residual.sum()
            if residual_sum > EPS_ABS:
                residual /= residual_sum
            else:
                residual = np.ones(N) / N
            cumulative_sum_res = np.cumsum(residual)
            cumulative_sum_res[-1] = 1.0
            remaining = N - len(indexes)
            random_vals = self.__randParticles.rng.random(remaining)
            res_indexes = np.searchsorted(cumulative_sum_res, random_vals)
            indexes += list(res_indexes)
            indexes = np.array(indexes)

        else:
            raise ParamError(
                f"Unknown resampling method: {method!r}. "
                f"Expected one of: 'multinomial', 'systematic', 'stratified', 'residual'."
            )

        return indexes

    def _precompute(self) -> None:
        """Précalcule et met en cache les constantes numériques du filtre.

        Extrait les blocs ``Q``, ``M``, ``R`` de la matrice de covariance
        jointe ``mQ``, puis calcule :

        - ``P'_x = Q - M @ R^{-1} @ M^T`` (complément de Schur de R dans Q).
        - ``L = cholesky(P'_x)`` pour le tirage des particules.
        - ``MRinv = M @ R^{-1}`` pour la correction par corrélation.
        - ``log_norm_const`` pour le calcul log-vraisemblance.

        Raises
        ------
        InvertibilityError
            Si ``R`` n'est pas inversible (diagnostic FAIL).
        CovarianceError
            Si ``P'_x`` n'est pas définie positive et que la régularisation
            de Tikhonov échoue.
        """
        Q = self.param.mQ[: self.dim_x, : self.dim_x]
        M = self.param.mQ[: self.dim_x, self.dim_x :]
        R = self.param.mQ[self.dim_x :, self.dim_x :]

        # --- Inversion de R avec diagnostic complet ---
        try:
            R_inv = InvertibleMatrix(R).inverse()
        except RuntimeError as e:
            raise InvertibilityError(
                "_precompute: R is not invertible — cannot continue.",
                matrix_name="R",
            ) from e

        # --- Complément de Schur ---
        P_prime_x_base = Q - M @ R_inv @ M.T

        # --- Validation et régularisation si nécessaire ---
        cov_diag = CovarianceMatrix(P_prime_x_base)
        report = cov_diag.check()

        if not report.is_ok and not report.is_valid:
            P_prime_x = cov_diag.regularized()
        else:
            P_prime_x = P_prime_x_base

        self._cached = dict(
            R=R,
            R_inv=R_inv,
            MRinv=M @ R_inv,
            L=cholesky(P_prime_x, lower=True),
            log_norm_const=-0.5
            * (self.dim_y * np.log(2 * np.pi) + np.linalg.slogdet(R)[1]),
        )

    @staticmethod
    def _safe_normalize_log_weights(log_weights: np.ndarray) -> np.ndarray:
        """Normalise les log-poids de façon numériquement stable.

        Gère : nan (overflow vraisemblance), tous à -inf (dégénérescence totale),
        underflow extrême après exp.
        """

        # nan → -inf (overflow dans le terme quadratique)
        nan_mask = np.isnan(log_weights)
        if nan_mask.any():
            logger.warning(
                f"Je rentre dans _safe_normalize_log_weights(...) - if nan_mask.any()"
            )
            log_weights = log_weights.copy()
            log_weights[nan_mask] = -np.inf

        # Dégénérescence totale : toutes les particules incompatibles → poids uniformes
        finite_mask = np.isfinite(log_weights)
        if not finite_mask.any():
            logger.warning(
                f"Je rentre dans _safe_normalize_log_weights(...) - if not finite_mask.any()"
            )
            return np.full(len(log_weights), 1.0 / len(log_weights))

        # log-sum-exp stable : soustraction du max fini uniquement
        max_lw = np.max(log_weights[finite_mask])
        log_weights = np.where(finite_mask, log_weights - max_lw, -np.inf)

        weights = np.exp(log_weights)
        total = weights.sum()

        # Underflow extrême après exp
        if not np.isfinite(total) or total <= 0.0:
            logger.warning(
                f"Je rentre dans _safe_normalize_log_weights(...) - if not np.isfinite(total) or total <= 0.0"
            )
            return np.full(len(log_weights), 1.0 / len(log_weights))

        return weights / total

    def process_filter(
        self,
        N: Optional[int] = None,
        data_generator: Optional[
            Generator[tuple[int, Optional[np.ndarray], np.ndarray], None, None]
        ] = None,
    ) -> Generator[
        tuple[int, Optional[np.ndarray], np.ndarray, np.ndarray, np.ndarray],
        None,
        None,
    ]:
        """Exécute le filtre particulaire et produit les estimées pas à pas.

        Parameters
        ----------
        N : int, optional
            Nombre maximal de pas de temps à traiter. Si ``None``, la boucle
            tourne jusqu'à épuisement du générateur de données.
        data_generator : Generator, optional
            Générateur externe fournissant des triplets
            ``(k, x_true, y_obs)``. Si ``None``, le générateur interne
            ``_data_generation()`` est utilisé.

        Yields
        ------
        k : int
            Indice temporel courant.
        xkp1 : np.ndarray or None
            Vérité terrain à l'instant ``k`` (``None`` si indisponible).
        ykp1 : np.ndarray, shape (dim_y, 1)
            Observation à l'instant ``k``.
        Xkp1_predict : np.ndarray, shape (dim_x, 1)
            Estimée a priori (avant observation).
        Xkp1_update : np.ndarray, shape (dim_x, 1)
            Estimée a posteriori (après observation).

        Raises
        ------
        ParamError
            Si ``N`` n'est pas un entier strictement positif ou ``None``.
        InvertibilityError
            Si ``R`` n'est pas inversible lors du précalcul.
        CovarianceError
            Si une matrice de covariance est invalide et non régularisable.
        StepValidationError
            Si la construction d'un ``PKFStep`` échoue.
        FilterError
            Si une erreur inattendue survient pendant le filtrage.
        """
        self._validate_N(N)

        generator = (
            data_generator if data_generator is not None else self._data_generation()
        )

        # Précalcul des constantes — lève InvertibilityError ou CovarianceError
        self._precompute()

        # Initialisation des particules et des poids
        particles_current: np.ndarray = self.__randParticles.rng.multivariate_normal(
            self.mz0[: self.dim_x].flatten(),
            self.Pz0[: self.dim_x, : self.dim_x],
            self.nbParticles,
        )
        particles_current = particles_current[..., np.newaxis]
        weights: np.ndarray = np.full(self.nbParticles, 1.0 / self.nbParticles)

        # --- First estimate -----------------------------------------------------------
        step = self._firstEstimate(generator)
        if step.xkp1 is None:
            self.ground_truth = False

        yield step.k, step.xkp1, step.ykp1, step.Xkp1_predict, step.Xkp1_update

        # --- Subsequent steps ---------------------------------------------------------
        while N is None or step.k < N:

            particles_previous = particles_current.copy()

            # =========================
            # PREDICTION
            # =========================
            Xkp1_predict: np.ndarray = np.mean(particles_current, axis=0)
            diff = particles_current - Xkp1_predict
            PXXkp1_predict: np.ndarray = np.einsum("tik,tjk->ij", diff, diff) / (
                self.nbParticles - 1
            )
            self._check_covariance(PXXkp1_predict, step.k, name="PXXkp1_predict")

            # Nouvelle observation
            try:
                new_k, new_xkp1, new_ykp1 = next(generator)
            except StopIteration:
                return

            # =========================
            # PROPAGATION via g (données scalaires)
            # =========================
            # muxy: np.ndarray = np.array(
            #     [
            #         self.param.g(
            #             np.vstack([p, step.ykp1]),
            #             self.zeros_dim_xy_1,
            #             self.dt,
            #         )
            #         for p in particles_previous
            #     ]
            # )

            # =========================
            # PROPAGATION via g (vectorisée)
            # =========================
            # particles_previous : (nbParticles, dim_x, 1)
            # step.ykp1 : (dim_y, 1)  →  à répliquer pour former z : (nbParticles, dim_xy, 1)
            ykp1_tiled = np.tile(step.ykp1, (self.nbParticles, 1, 1))  # (N, dim_y, 1)
            z_all = np.concatenate(
                [particles_previous, ykp1_tiled], axis=1
            )  # (N, dim_xy, 1)
            zeros_tiled = np.tile(
                self.zeros_dim_xy_1, (self.nbParticles, 1, 1)
            )  # (N, dim_xy, 1)

            muxy = self.param.g(z_all, zeros_tiled, self.dt)  # (N, dim_xy, 1)

            # DEBUG — vérification muxy
            if np.any(~np.isfinite(muxy)):
                bad = (~np.isfinite(muxy)).any(axis=(1, 2))
                print(
                    f"[DEBUG] Step {new_k}: muxy NaN/Inf dans {bad.sum()}/{self.nbParticles} particules"
                )
                print(
                    f"  particles_previous range: [{particles_previous.min():.3g}, {particles_previous.max():.3g}]"
                )
                print(f"  step.ykp1: {step.ykp1.flatten()}")

            # =========================
            # INNOVATION
            # =========================
            innovations: np.ndarray = new_ykp1 - muxy[:, self.dim_x :, :]

            # Mise à jour des poids par log-vraisemblance gaussienne
            tmp = np.matmul(self._cached["R_inv"], innovations)
            quad = np.matmul(innovations.transpose(0, 2, 1), tmp)
            exponents = -0.5 * quad.reshape(self.nbParticles)

            log_weights = (
                np.log(np.maximum(weights, EPS_ABS))
                + exponents
                + self._cached["log_norm_const"]
            )
            weights = self._safe_normalize_log_weights(log_weights)

            if __debug__:
                assert not np.any(
                    np.isnan(weights)
                ), f"NaN dans les poids au step {new_k}"
                assert np.isclose(
                    weights.sum(), 1.0, atol=1e-6
                ), f"Poids non normalisés : {weights.sum()}"

            # Mise à jour stochastique avec correction par corrélation
            mu_prime_x_all = (
                muxy[:, : self.dim_x, :] + self._cached["MRinv"] @ innovations
            )
            noise = self.__randParticles.rng.standard_normal(
                (self.nbParticles, self.dim_x, 1)
            )
            particles_current = mu_prime_x_all + np.einsum(
                "ij,njk->nik", self._cached["L"], noise
            )

            # Clipping des particules divergentes
            PARTICLE_CLIP = 1e6  # à ajuster selon l'échelle physique du modèle
            n_clipped = np.sum(
                ~np.isfinite(particles_current)
                | (np.abs(particles_current) > PARTICLE_CLIP)
            )
            particles_current = np.clip(
                np.where(np.isfinite(particles_current), particles_current, 0.0),
                -PARTICLE_CLIP,
                PARTICLE_CLIP,
            )

            # DEBUG — vérification particles_current
            if np.any(~np.isfinite(particles_current)):
                bad = (~np.isfinite(particles_current)).any(axis=(1, 2))
                print(
                    f"[DEBUG] Step {new_k}: particles_current NaN/Inf dans {bad.sum()}/{self.nbParticles} particules"
                )
                print(f"  mu_prime_x_all finite: {np.all(np.isfinite(mu_prime_x_all))}")
                print(
                    f"  mu_prime_x_all range:  [{np.nanmin(mu_prime_x_all):.3g}, {np.nanmax(mu_prime_x_all):.3g}]"
                )
                print(f"  MRinv:\n{self._cached['MRinv']}")
                print(
                    f"  innovations range: [{np.nanmin(innovations):.3g}, {np.nanmax(innovations):.3g}]"
                )
                print(f"  L (Cholesky):\n{self._cached['L']}")

            # =========================
            # ESTIMATION A POSTERIORI
            # =========================
            particles_current_temp: np.ndarray = particles_current.squeeze(-1)
            Xkp1_update: np.ndarray = np.average(
                particles_current_temp, axis=0, weights=weights
            )[:, None]

            dx = particles_current_temp - Xkp1_update.T
            PXXkp1_update = (weights[:, None] * dx).T @ dx
            self._check_covariance(PXXkp1_update, step.k, name="PXXkp1_update")

            ess_before_resample = 1.0 / np.sum(weights**2)
            max_innovation = np.abs(innovations).max()
            logger.debug(
                f"Step {new_k}: ESS={ess_before_resample:.1f}/{self.nbParticles}, "
                f"n_clipped={n_clipped}, "
                f"max_innov={max_innovation:.3g}, "
                f"Xupdate={Xkp1_update.flatten()}"
            )

            # =========================
            # RÉÉCHANTILLONNAGE
            # =========================
            ess: float = 1.0 / np.sum(weights**2)
            if ess < self.resample_threshold * self.nbParticles:
                indexes = self.resample(weights, self.resample_method)
                particles_current = particles_current[indexes]
                weights.fill(1.0 / self.nbParticles)

            if __debug__:
                assert not np.any(
                    np.isnan(particles_current)
                ), f"NaN dans les particules au step {new_k}"

            # =========================
            # ENREGISTREMENT
            # =========================
            try:
                step = PKFStep(
                    k=new_k,
                    xkp1=new_xkp1.copy() if new_xkp1 is not None else None,
                    ykp1=new_ykp1.copy(),
                    Xkp1_predict=Xkp1_predict.copy(),
                    PXXkp1_predict=PXXkp1_predict.copy(),
                    Xkp1_update=Xkp1_update.copy(),
                    PXXkp1_update=PXXkp1_update.copy(),
                )
            except (ValueError, Exception) as e:
                raise StepValidationError(
                    f"Step {new_k}: PKFStep construction failed in process_filter.",
                    step=new_k,
                ) from e

            self.history.record(step)

            if self.verbose > 1:
                rich_show_fields(step, title=f"Step {new_k} Update")

            yield new_k, new_xkp1, new_ykp1, step.Xkp1_predict, step.Xkp1_update
