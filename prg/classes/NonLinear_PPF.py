"""
####################################################################
Pairwise Particle Filter implementation
####################################################################
"""

from __future__ import annotations

import logging
from collections.abc import Generator

# Stdlib
# Third-party
import numpy as np
from scipy.linalg import cholesky

from prg.classes.matrix_diagnostics import CovarianceMatrix, InvertibleMatrix
from prg.classes.PKF import PKF, PKFStep

# Local
from prg.classes.SeedGenerator import SeedGenerator
from prg.utils.exceptions import (
    InvertibilityError,
    ParamError,
    StepValidationError,
)
from prg.utils.numerics import EPS_ABS
from prg.utils.utils import rich_show_fields

logger = logging.getLogger(__name__)

__all__ = ["NonLinear_PPF"]


class NonLinear_PPF(PKF):
    """Nonlinear Pairwise Particle Filter.

    Implements a particle filter for nonlinear systems with additive Gaussian
    noise, with optional support for correlated noises (M ≠ 0) via the joint
    covariance matrix ``mQ``.

    The state model considered is:

    .. math::

        (X_n,Y_n)  = g_x(X_{n-1}, Y_{n-1}, V^x_n, V^y_n)

    with:

    .. math::

        \\begin{pmatrix} V^x_n \\\\ V^y_n \\end{pmatrix}
        \\sim \\mathcal{N}\\!\\left(0,\\,
        \\begin{pmatrix} Q & M \\\\ M^\\top & R \\end{pmatrix}\\right)


    Parameters
    ----------
    param : ParamLinear | ParamNonLinear
        Model parameters (dimensions, noise matrices, initial conditions, etc.).
    n_particles : int, optional
        Number of particles. Default 300.
    resample_threshold : float, optional
        Relative threshold on the ESS triggering resampling,
        expressed as a fraction of ``n_particles``. Default 0.5.
    resample_method : str, optional
        Resampling method among ``'multinomial'``,
        ``'systematic'``, ``'stratified'`` (default) and ``'residual'``.
    sKey : optional
        Seed for the random number generator.
    verbose : int, optional
        Verbosity level (0 = silent). Default 0.

    Attributes
    ----------
    n_particles : int
        Number of particles.
    resample_threshold : float
        Resampling threshold.
    resample_method : str
        Resampling method.
    _cached : dict
        Constants pre-computed by ``_precompute()``: ``R``, ``R_inv``,
        ``MRinv``, ``L`` (Cholesky of P'_x), ``log_norm_const``.
    """

    def __init__(
        self,
        param,
        n_particles: int = 300,
        resample_threshold: float = 0.5,
        resample_method: str = "stratified",
        sKey=None,
        verbose: int = 0,
    ) -> None:
        super().__init__(param, sKey, verbose)

        self.n_particles = n_particles
        self.resample_threshold = resample_threshold
        self.resample_method = resample_method

        # Random number generator
        self.__randParticles = SeedGenerator()

        # Dictionary of pre-computed constants
        self._cached: dict = {}

    # =========================
    # UNIFIED RESAMPLING
    # =========================
    def resample(self, weights: np.ndarray, method: str = "stratified") -> np.ndarray:
        """Resamples particles according to normalised weights.

        Parameters
        ----------
        weights : np.ndarray, shape (N,)
            Normalised particle weights (must sum to 1).
        method : str, optional
            Resampling method:

            - ``'multinomial'``  : independent multinomial sampling.
            - ``'systematic'``   : systematic resampling (a single
              uniform random draw).
            - ``'stratified'``   : stratified resampling (one draw
              per stratum). Default method.
            - ``'residual'``     : residual resampling (deterministic
              copies + stochastic residual).

        Returns
        -------
        indexes : np.ndarray of int, shape (N,)
            Indices of the selected particles.

        Raises
        ------
        ParamError
            If ``method`` is not one of the accepted values.
        """
        N = self.n_particles
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
        """Pre-computes and caches the numerical constants of the filter.

        Extracts blocks ``Q``, ``M``, ``R`` from the joint covariance matrix
        ``mQ``, then computes:

        - ``P'_x = Q - M @ R^{-1} @ M^T`` (Schur complement of R in Q).
        - ``L = cholesky(P'_x)`` for particle sampling.
        - ``MRinv = M @ R^{-1}`` for correlation correction.
        - ``log_norm_const`` for the log-likelihood computation.

        Raises
        ------
        InvertibilityError
            If ``R`` is not invertible (diagnostic FAIL).
        CovarianceError
            If ``P'_x`` is not positive definite and Tikhonov regularisation
            fails.
        """
        Q = self.param.mQ[: self.dim_x, : self.dim_x]
        M = self.param.mQ[: self.dim_x, self.dim_x :]
        R = self.param.mQ[self.dim_x :, self.dim_x :]

        # --- Inversion of R with full diagnostic ---
        try:
            R_inv = InvertibleMatrix(R).inverse()
        except RuntimeError as e:
            raise InvertibilityError(
                "_precompute: R is not invertible — cannot continue.",
                matrix_name="R",
            ) from e

        # --- Schur complement ---
        P_prime_x_base = Q - M @ R_inv @ M.T

        # --- Validation and regularisation if needed ---
        cov_diag = CovarianceMatrix(P_prime_x_base)
        report = cov_diag.check()

        if not report.is_ok and not report.is_valid:
            P_prime_x = cov_diag.regularized()
        else:
            P_prime_x = P_prime_x_base

        self._cached = {
            "R": R,
            "R_inv": R_inv,
            "MRinv": M @ R_inv,
            "L": cholesky(P_prime_x, lower=True),
            "log_norm_const": -0.5
            * (self.dim_y * np.log(2 * np.pi) + np.linalg.slogdet(R)[1]),
        }

    @staticmethod
    def _safe_normalize_log_weights(log_weights: np.ndarray) -> np.ndarray:
        """Normalises log-weights in a numerically stable way.

        Handles: nan (likelihood overflow), all at -inf (total degeneracy),
        extreme underflow after exp.
        """

        # nan → -inf (overflow in the quadratic term)
        nan_mask = np.isnan(log_weights)
        if nan_mask.any():
            logger.warning(
                "Entering _safe_normalize_log_weights(...) - if nan_mask.any()"
            )
            log_weights = log_weights.copy()
            log_weights[nan_mask] = -np.inf

        # Total degeneracy: all particles incompatible → uniform weights
        finite_mask = np.isfinite(log_weights)
        if not finite_mask.any():
            logger.warning(
                "Entering _safe_normalize_log_weights(...) - if not finite_mask.any()"
            )
            return np.full(len(log_weights), 1.0 / len(log_weights))

        # Stable log-sum-exp: subtract only the finite max
        max_lw = np.max(log_weights[finite_mask])
        log_weights = np.where(finite_mask, log_weights - max_lw, -np.inf)

        weights = np.exp(log_weights)
        total = weights.sum()

        # Extreme underflow after exp
        if not np.isfinite(total) or total <= 0.0:
            logger.warning(
                "Entering _safe_normalize_log_weights(...) - if not np.isfinite(total) or total <= 0.0"
            )
            return np.full(len(log_weights), 1.0 / len(log_weights))

        return weights / total

    def process_filter(
        self,
        N: int | None = None,
        data_generator: Generator[tuple[int, np.ndarray | None, np.ndarray], None, None] | None = None,
    ) -> Generator[
        tuple[int, np.ndarray | None, np.ndarray, np.ndarray, np.ndarray],
        None,
        None,
    ]:
        """Runs the particle filter and yields estimates step by step.

        Parameters
        ----------
        N : int, optional
            Maximum number of time steps to process. If ``None``, the loop
            runs until the data generator is exhausted.
        data_generator : Generator, optional
            External generator providing triplets
            ``(k, x_true, y_obs)``. If ``None``, the internal generator
            ``_data_generation()`` is used.

        Yields
        ------
        k : int
            Current time index.
        xkp1 : np.ndarray or None
            Ground truth at time ``k`` (``None`` if unavailable).
        ykp1 : np.ndarray, shape (dim_y, 1)
            Observation at time ``k``.
        Xkp1_predict : np.ndarray, shape (dim_x, 1)
            Prior estimate (before observation).
        Xkp1_update : np.ndarray, shape (dim_x, 1)
            Posterior estimate (after observation).

        Raises
        ------
        ParamError
            If ``N`` is not a strictly positive integer or ``None``.
        InvertibilityError
            If ``R`` is not invertible during pre-computation.
        CovarianceError
            If a covariance matrix is invalid and cannot be regularised.
        StepValidationError
            If construction of a ``PKFStep`` fails.
        FilterError
            If an unexpected error occurs during filtering.
        """
        self._validate_N(N)
        self.history.clear()

        generator = (
            data_generator if data_generator is not None else self._data_generation()
        )

        # Pre-computation of constants — raises InvertibilityError or CovarianceError
        self._precompute()

        # Initialisation of particles and weights
        particles_current: np.ndarray = self.__randParticles.rng.multivariate_normal(
            self.mz0[: self.dim_x].flatten(),
            self.Pz0[: self.dim_x, : self.dim_x],
            self.n_particles,
        )
        particles_current = particles_current[..., np.newaxis]
        weights: np.ndarray = np.full(self.n_particles, 1.0 / self.n_particles)

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
                self.n_particles - 1
            )
            self._check_covariance(PXXkp1_predict, step.k, name="PXXkp1_predict")

            # Nouvelle observation
            try:
                new_k, new_xkp1, new_ykp1 = next(generator)
            except StopIteration:
                return

            # =========================
            # PROPAGATION via g (scalar data)
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
            # PROPAGATION via g (vectorised)
            # =========================
            # particles_previous : (n_particles, dim_x, 1)
            # step.ykp1 : (dim_y, 1)  →  to be tiled to form z : (n_particles, dim_xy, 1)
            ykp1_tiled = np.tile(step.ykp1, (self.n_particles, 1, 1))  # (N, dim_y, 1)
            z_all = np.concatenate(
                [particles_previous, ykp1_tiled], axis=1
            )  # (N, dim_xy, 1)
            zeros_tiled = np.tile(
                self.zeros_dim_xy_1, (self.n_particles, 1, 1)
            )  # (N, dim_xy, 1)

            muxy = self.param.g(z_all, zeros_tiled, self.dt)  # (N, dim_xy, 1)

            # DEBUG — muxy verification
            if np.any(~np.isfinite(muxy)):
                bad = (~np.isfinite(muxy)).any(axis=(1, 2))
                logger.debug(
                    "Step %d: muxy NaN/Inf in %d/%d particles",
                    new_k, bad.sum(), self.n_particles,
                )
                logger.debug(
                    "  particles_previous range: [%.3g, %.3g]",
                    particles_previous.min(), particles_previous.max(),
                )
                logger.debug("  step.ykp1: %s", step.ykp1.flatten())

            # =========================
            # INNOVATION
            # =========================
            innovations: np.ndarray = new_ykp1 - muxy[:, self.dim_x :, :]

            # Weight update via Gaussian log-likelihood
            tmp = np.matmul(self._cached["R_inv"], innovations)
            quad = np.matmul(innovations.transpose(0, 2, 1), tmp)
            exponents = -0.5 * quad.reshape(self.n_particles)

            log_weights = (
                np.log(np.maximum(weights, EPS_ABS))
                + exponents
                + self._cached["log_norm_const"]
            )
            weights = self._safe_normalize_log_weights(log_weights)

            if __debug__:
                assert not np.any(np.isnan(weights)), f"NaN in weights at step {new_k}"
                assert np.isclose(
                    weights.sum(), 1.0, atol=1e-6
                ), f"Weights not normalised: {weights.sum()}"

            # Stochastic update with correlation correction
            mu_prime_x_all = (
                muxy[:, : self.dim_x, :] + self._cached["MRinv"] @ innovations
            )
            noise = self.__randParticles.rng.standard_normal(
                (self.n_particles, self.dim_x, 1)
            )
            particles_current = mu_prime_x_all + np.einsum(
                "ij,njk->nik", self._cached["L"], noise
            )

            # Clipping of diverging particles
            PARTICLE_CLIP = (
                1e6  # to be adjusted according to the physical scale of the model
            )
            n_clipped = np.sum(
                ~np.isfinite(particles_current)
                | (np.abs(particles_current) > PARTICLE_CLIP)
            )
            particles_current = np.clip(
                np.where(np.isfinite(particles_current), particles_current, 0.0),
                -PARTICLE_CLIP,
                PARTICLE_CLIP,
            )

            # DEBUG — particles_current verification
            if np.any(~np.isfinite(particles_current)):
                bad = (~np.isfinite(particles_current)).any(axis=(1, 2))
                logger.debug(
                    "Step %d: particles_current NaN/Inf in %d/%d particles",
                    new_k, bad.sum(), self.n_particles,
                )
                logger.debug(
                    "  mu_prime_x_all finite: %s",
                    np.all(np.isfinite(mu_prime_x_all)),
                )
                logger.debug(
                    "  mu_prime_x_all range:  [%.3g, %.3g]",
                    np.nanmin(mu_prime_x_all), np.nanmax(mu_prime_x_all),
                )
                logger.debug("  MRinv:\n%s", self._cached['MRinv'])
                logger.debug(
                    "  innovations range: [%.3g, %.3g]",
                    np.nanmin(innovations), np.nanmax(innovations),
                )
                logger.debug("  L (Cholesky):\n%s", self._cached['L'])

            # =========================
            # POSTERIOR ESTIMATE — Rao-Blackwellised
            # =========================
            # We use the conditional means mu'_x (not the noisy particles)
            # to avoid the variance of P'_x being counted twice in PXXkp1_update.
            mu_prime_x_temp: np.ndarray = mu_prime_x_all.squeeze(-1)  # (N, dim_x)

            Xkp1_update: np.ndarray = np.average(
                mu_prime_x_temp, axis=0, weights=weights
            )[:, None]

            # Between-particle variance of the conditional means
            dx_mu = mu_prime_x_temp - Xkp1_update.T  # (N, dim_x)
            var_between = (weights[:, None] * dx_mu).T @ dx_mu  # (dim_x, dim_x)

            # Intra-particle variance = P'_x (constant, analytical)
            P_prime_x = self._cached["L"] @ self._cached["L"].T

            # ESS correction: ~ 1 if ESS is high, avoids over-estimation at low ESS
            _ess_correction = 1.0 - np.sum(weights**2)

            # Final (exact Rao-Blackwell formula)
            PXXkp1_update = var_between + P_prime_x

            self._check_covariance(PXXkp1_update, step.k, name="PXXkp1_update")

            ess_before_resample = 1.0 / np.sum(weights**2)
            max_innovation = np.abs(innovations).max()
            logger.debug(
                "Step %d: ESS=%.1f/%d, n_clipped=%d, max_innov=%.3g, Xupdate=%s",
                new_k,
                ess_before_resample,
                self.n_particles,
                n_clipped,
                max_innovation,
                Xkp1_update.flatten(),
            )

            # =========================
            # RESAMPLING
            # =========================
            ess: float = 1.0 / np.sum(weights**2)
            if ess < self.resample_threshold * self.n_particles:
                indexes = self.resample(weights, self.resample_method)
                particles_current = particles_current[indexes]
                weights.fill(1.0 / self.n_particles)

            if __debug__:
                assert not np.any(
                    np.isnan(particles_current)
                ), f"NaN in particles at step {new_k}"

            # =========================
            # RECORDING
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
