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
        self.__randParticles = SeedGenerator(9)

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
        ValueError
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
            raise ValueError(f"Unknown resampling method: {method}")

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
        RuntimeError
            Si ``R`` n'est pas inversible (diagnostic FAIL).
        ValueError
            Si ``P'_x`` n'est pas définie positive et que la régularisation
            de Tikhonov échoue.
        """
        Q = self.param.mQ[: self.dim_x, : self.dim_x]
        M = self.param.mQ[: self.dim_x, self.dim_x :]
        R = self.param.mQ[self.dim_x :, self.dim_x :]

        # --- Inversion de R avec diagnostic complet ---
        try:
            R_inv = InvertibleMatrix(R).inverse()
        except Exception as e:
            input("ATTENTE _precompute")

        # --- Complément de Schur ---
        P_prime_x_base = Q - M @ R_inv @ M.T

        # --- Validation et régularisation si nécessaire ---
        cov_diag = CovarianceMatrix(P_prime_x_base)
        report = cov_diag.check()

        if not report.is_ok:
            self.logger.warning("_precompute: P'_x — %s", report.overall_status)
            if self.verbose > 1:
                self.logger.debug("_precompute: P'_x full diagnostic:\n%s", report)

            if not report.is_valid:
                self.logger.warning(
                    "_precompute: P'_x is invalid — attempting regularization."
                )
                try:
                    P_prime_x = cov_diag.regularized()
                    self.logger.warning("_precompute: P'_x regularized successfully.")
                except RuntimeError as e:
                    raise ValueError(
                        f"_precompute: P'_x is not positive definite and "
                        f"regularization failed — cannot continue.\n{e}"
                    ) from e
            else:
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

        À chaque pas de temps, la méthode effectue dans l'ordre :

        1. **Prédiction** : moyenne et covariance empiriques des particules.
        2. **Propagation** : application de la fonction non linéaire ``g``
           à chaque particule pour obtenir la prédiction jointe ``(X, Y)``.
        3. **Innovation** : écart entre l'observation reçue et la prédiction.
        4. **Mise à jour**  :

           - *Cas général* : gain de Kalman empirique, poids par
             vraisemblance gaussienne, mise à jour stochastique.

        5. **Estimation a posteriori** : moyenne et covariance pondérées
           (forme de Joseph pour la covariance).
        6. **Rééchantillonnage** si l'ESS passe sous le seuil.

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

        Notes
        -----
        Les assertions de débogage (``__debug__``) vérifient l'absence de
        NaN dans les poids et les particules, ainsi que la normalisation
        des poids.
        """

        self._validate_N(N)

        generator = (
            data_generator if data_generator is not None else self._data_generation()
        )

        # Précalcul des constantes
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

        # print(f"kp1={step.k}")
        # print(step.Xkp1_predict)
        # print(step.PXXkp1_predict)
        # print(step.Xkp1_update)
        # print(step.PXXkp1_update)
        # input("ATTENTE")

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
            # PROPAGATION via g
            # =========================
            muxy: np.ndarray = np.array(
                [
                    self.param.g(
                        np.vstack([p, step.ykp1]),
                        self.zeros_dim_xy_1,
                        self.dt,
                    )
                    for p in particles_previous
                ]
            )

            # =========================
            # PREDICTION JOINTE Z = (X, Y)
            # =========================
            # Zkp1_predict: np.ndarray = np.average(muxy, axis=0, weights=weights)
            # dz = muxy - Zkp1_predict[None, :, :]
            # Pkp1_predict: np.ndarray = np.einsum("i,ijk,ilk->jl", weights, dz, dz)
            # self._check_covariance(Pkp1_predict, step.k, name="Pkp1_predict")

            # # # Extraction des blocs de la covariance jointe
            # PYYkp1_predict: np.ndarray = Pkp1_predict[self.dim_x :, self.dim_x :]
            # PXYkp1_predict: np.ndarray = Pkp1_predict[: self.dim_x, self.dim_x :]

            # =========================
            # INNOVATION
            # =========================
            # Ykp1_predict: np.ndarray = Zkp1_predict[self.dim_x :]
            # ikp1: np.ndarray = new_ykp1 - Ykp1_predict
            innovations: np.ndarray = new_ykp1 - muxy[:, self.dim_x :, :]

            # # Cas général : R > 0
            # Skp1 = PYYkp1_predict + self._cached["R"]
            # # Validate innovation covariance before Cholesky solve
            # self._check_invertible(Skp1, step.k, name="Skp1")

            # try:
            #     c, low = cho_factor(Skp1)
            #     Kkp1: np.ndarray = PXYkp1_predict @ cho_solve((c, low), self.eye_dim_y)
            # except Exception as e:
            #     self.logger.error(
            #         "Step %d: LinAlgError/ValueError in cho_factor/solve: %s", step.k, e
            #     )
            #     raise

            # Mise à jour des poids par log-vraisemblance gaussienne
            tmp = np.matmul(self._cached["R_inv"], innovations)
            quad = np.matmul(innovations.transpose(0, 2, 1), tmp)
            exponents = -0.5 * quad.reshape(self.nbParticles)

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

            # Forme de Joseph — préserve la définition positive
            # Joseph_factor = np.vstack((self.eye_dim_x, -Kkp1.T))
            # PXXkp1_update = Joseph_factor.T @ Pkp1_predict @ Joseph_factor
            # self._check_covariance(PXXkp1_update, step.k, name="PXXkp1_update")

            # =========================
            # RÉÉCHANTILLONNAGE
            # =========================
            ess: float = 1.0 / np.sum(weights**2)
            if ess < self.resample_threshold * self.nbParticles:
                indexes = self.resample(
                    weights, "multinomial"
                )  # multinomial, systematic, stratified, residual, self.resample_method,
                particles_current = particles_current[indexes]
                weights.fill(1.0 / self.nbParticles)

            if __debug__:
                assert not np.any(
                    np.isnan(particles_current)
                ), f"NaN dans les particules au step {new_k}"

            # =========================
            # ENREGISTREMENT
            # =========================
            step = PKFStep(
                k=new_k,
                xkp1=new_xkp1.copy() if new_xkp1 is not None else None,
                ykp1=new_ykp1.copy(),
                Xkp1_predict=Xkp1_predict.copy(),
                PXXkp1_predict=PXXkp1_predict.copy(),
                # ikp1=ikp1.copy(),
                # Skp1=Skp1.copy(),
                # Kkp1=Kkp1.copy(),
                Xkp1_update=Xkp1_update.copy(),
                PXXkp1_update=PXXkp1_update.copy(),
            )

            # print(f"kp1={step.k}")
            # print(step.Xkp1_predict)
            # print(step.PXXkp1_predict)
            # print(step.Xkp1_update)
            # print(step.PXXkp1_update)
            # input("ATTENTE")

            self.history.record(step)

            if self.verbose > 1:
                rich_show_fields(step, title=f"Step {new_k} Update")

            yield new_k, new_xkp1, new_ykp1, step.Xkp1_predict, step.Xkp1_update
