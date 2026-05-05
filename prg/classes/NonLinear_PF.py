"""
####################################################################
Bootstrap Particle Filter (PF) for classical (non-pairwise) models
####################################################################

Designed for models of the form:

    X_{k+1} = f(X_k, V^x_k)       V^x_k ~ N(0, Q)
    Y_{k+1} = h(X_{k+1}, V^y_k)   V^y_k ~ N(0, R)

where f and h are accessed via self.param.f and self.param.h.

The proposal is the prior transition density p(x_{k+1} | x_k),
and the likelihood is p(y_{k+1} | x_{k+1}) = N(y; h(x, 0), R).

Unlike NonLinear_PPF (pairwise), this filter does NOT require M != 0.
"""

from __future__ import annotations

import logging
from collections.abc import Generator

import numpy as np
from scipy.linalg import cholesky

from prg.classes.MatrixDiagnostics import CovarianceMatrix, InvertibleMatrix
from prg.classes.PKF import PKF, PKFStep
from prg.classes.SeedGenerator import SeedGenerator
from prg.utils.exceptions import (
    InvertibilityError,
    ParamError,
    StepValidationError,
)
from prg.utils.numerics import EPS_ABS
from prg.utils.utils import rich_show_fields

logger = logging.getLogger(__name__)

__all__ = ["NonLinear_PF"]


class NonLinear_PF(PKF):
    """Bootstrap Particle Filter for classical (non-pairwise) nonlinear models.

    Uses the prior transition density as proposal:

    .. math::

        x_i^{k+1} \\sim p(x | x_i^k) = f(x_i^k, v),\\quad v \\sim \\mathcal{N}(0, Q)

    Weights are updated with the observation likelihood:

    .. math::

        w_i \\propto p(y_{k+1} | x_i^{k+1}) = \\mathcal{N}\\!\\left(y;\\,
        h(x_i^{k+1}, 0),\\, R\\right)

    Parameters
    ----------
    param : ParamLinear | ParamNonLinear
        Model parameters. Must expose ``param.f``, ``param.h``,
        and ``param.mQ`` partitioned as diag(Q, R).
    n_particles : int, optional
        Number of particles. Default 300.
    resample_threshold : float, optional
        Relative ESS threshold triggering resampling, as a fraction
        of ``n_particles``. Default 0.5.
    resample_method : str, optional
        One of ``'multinomial'``, ``'systematic'``,
        ``'stratified'`` (default), ``'residual'``.
    sKey : optional
        Random seed for reproducibility.
    verbose : int, optional
        Verbosity level (0 = silent). Default 0.

    Attributes
    ----------
    n_particles : int
    resample_threshold : float
    resample_method : str
    _cached : dict
        Pre-computed constants: ``Q``, ``R``, ``R_inv``,
        ``L_Q`` (Cholesky of Q), ``log_norm_const``.
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

        if getattr(param, "pairwiseModel", False):
            raise ParamError(
                "NonLinear_PF does not support pairwise models (param.f is None). "
                "Use NonLinear_PPF instead."
            )

        self.n_particles = n_particles
        self.resample_threshold = resample_threshold
        self.resample_method = resample_method

        self.__randParticles = SeedGenerator()
        self._cached: dict = {}

    # =========================
    # RESAMPLING
    # =========================

    def resample(self, weights: np.ndarray, method: str = "stratified") -> np.ndarray:
        """Resample particles according to normalised weights.

        Parameters
        ----------
        weights : np.ndarray, shape (N,)
            Normalised particle weights (must sum to 1).
        method : str, optional
            Resampling method: ``'multinomial'``, ``'systematic'``,
            ``'stratified'`` (default), ``'residual'``.

        Returns
        -------
        indexes : np.ndarray of int, shape (N,)
            Indices of selected particles.

        Raises
        ------
        ParamError
            If ``method`` is unknown.
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

    # =========================
    # PRE-COMPUTATION
    # =========================

    def _precompute(self) -> None:
        """Pre-compute and cache filter constants.

        Extracts ``Q`` (process noise) and ``R`` (observation noise)
        from ``param.mQ``, then computes:

        - ``L_Q = cholesky(Q)`` for process noise sampling.
        - ``R_inv`` for the observation likelihood.
        - ``log_norm_const`` for the log-likelihood.

        Raises
        ------
        InvertibilityError
            If ``R`` is not invertible.
        CovarianceError
            If ``Q`` is not positive definite and regularisation fails.
        """
        Q = self.param.mQ[: self.dim_x, : self.dim_x]
        R = self.param.mQ[self.dim_x :, self.dim_x :]

        # --- Inversion of R ---
        try:
            R_inv = InvertibleMatrix(R).inverse()
        except RuntimeError as e:
            raise InvertibilityError(
                "_precompute: R is not invertible — cannot continue.",
                matrix_name="R",
            ) from e

        # --- Cholesky of Q for process noise sampling ---
        cov_diag = CovarianceMatrix(Q)
        report = cov_diag.check()
        if not report.is_ok and not report.is_valid:
            Q = cov_diag.regularized()

        self._cached = {
            "Q": Q,
            "R": R,
            "R_inv": R_inv,
            "L_Q": cholesky(Q, lower=True),
            "log_norm_const": -0.5
            * (self.dim_y * np.log(2 * np.pi) + np.linalg.slogdet(R)[1]),
        }

    # =========================
    # LOG-WEIGHT NORMALISATION
    # =========================

    @staticmethod
    def _safe_normalize_log_weights(log_weights: np.ndarray) -> np.ndarray:
        """Normalise log-weights in a numerically stable way.

        Handles NaN (likelihood overflow), total degeneracy (all -inf),
        and extreme underflow after exp.
        """
        nan_mask = np.isnan(log_weights)
        if nan_mask.any():
            logger.warning("_safe_normalize_log_weights: NaN detected in log_weights")
            log_weights = log_weights.copy()
            log_weights[nan_mask] = -np.inf

        finite_mask = np.isfinite(log_weights)
        if not finite_mask.any():
            logger.warning("_safe_normalize_log_weights: all weights -inf → uniform")
            return np.full(len(log_weights), 1.0 / len(log_weights))

        max_lw = np.max(log_weights[finite_mask])
        log_weights = np.where(finite_mask, log_weights - max_lw, -np.inf)

        weights = np.exp(log_weights)
        total = weights.sum()

        if not np.isfinite(total) or total <= 0.0:
            logger.warning("_safe_normalize_log_weights: underflow after exp → uniform")
            return np.full(len(log_weights), 1.0 / len(log_weights))

        return weights / total

    # =========================
    # MAIN FILTER LOOP
    # =========================

    def process_filter(
        self,
        N: int | None = None,
        data_generator: Generator[tuple[int, np.ndarray | None, np.ndarray], None, None] | None = None,
    ) -> Generator[
        tuple[int, np.ndarray | None, np.ndarray, np.ndarray, np.ndarray],
        None,
        None,
    ]:
        """Run the Bootstrap Particle Filter and yield estimates step by step.

        Parameters
        ----------
        N : int, optional
            Maximum number of time steps. If ``None``, runs until the
            data generator is exhausted.
        data_generator : Generator, optional
            External generator yielding ``(k, x_true, y_obs)`` triplets.
            If ``None``, the internal ``_data_generation()`` is used.

        Yields
        ------
        k : int
        xkp1 : np.ndarray or None
        ykp1 : np.ndarray, shape (dim_y, 1)
        Xkp1_predict : np.ndarray, shape (dim_x, 1)
        Xkp1_update : np.ndarray, shape (dim_x, 1)

        Raises
        ------
        ParamError
            If ``N`` is invalid.
        InvertibilityError
            If ``R`` is not invertible during pre-computation.
        CovarianceError
            If a covariance matrix is invalid and cannot be regularised.
        StepValidationError
            If ``PKFStep`` construction fails.
        FilterError
            If an unexpected error occurs during filtering.
        """
        self._validate_N(N)
        self.history.clear()

        generator = (
            data_generator if data_generator is not None else self._data_generation()
        )

        self._precompute()

        # --- Particle initialisation ---
        particles_current: np.ndarray = self.__randParticles.rng.multivariate_normal(
            self.mz0[: self.dim_x].flatten(),
            self.Pz0[: self.dim_x, : self.dim_x],
            self.n_particles,
        )[
            ..., np.newaxis
        ]  # (N, dim_x, 1)

        weights: np.ndarray = np.full(self.n_particles, 1.0 / self.n_particles)

        # --- First estimate (Gaussian conditioning on prior) ---
        step = self._firstEstimate(generator)
        if step.xkp1 is None:
            self.ground_truth = False

        yield step.k, step.xkp1, step.ykp1, step.Xkp1_predict, step.Xkp1_update

        # --- Main loop ---
        while N is None or step.k < N:

            # =========================
            # PREDICTION — prior estimate before propagation
            # =========================
            Xkp1_predict: np.ndarray = np.average(
                particles_current.squeeze(-1), axis=0, weights=weights
            )[:, None]
            dx_pred = particles_current.squeeze(-1) - Xkp1_predict.T
            PXXkp1_predict: np.ndarray = (weights[:, None] * dx_pred).T @ dx_pred
            self._check_covariance(PXXkp1_predict, step.k, name="PXXkp1_predict")

            # --- Fetch next observation ---
            try:
                new_k, new_xkp1, new_ykp1 = next(generator)
            except StopIteration:
                return

            # =========================
            # PROPAGATION — sample x_i ~ p(x_{k+1} | x_i^k)
            # x_{k+1} = f(x_k, L_Q * z),  z ~ N(0, I)
            # =========================
            noise = self.__randParticles.rng.standard_normal(
                (self.n_particles, self.dim_x, 1)
            )
            process_noise = np.einsum(
                "ij,njk->nik", self._cached["L_Q"], noise
            )  # (N, dim_x, 1)

            # Vectorised call to f — noise passed directly as second argument
            particles_propagated = self.param.f(
                particles_current, process_noise, self.dt
            )  # (N, dim_x, 1)

            if np.any(~np.isfinite(particles_propagated)):
                bad = (~np.isfinite(particles_propagated)).any(axis=(1, 2))
                logger.warning(
                    f"Step {new_k}: particles_propagated NaN/Inf "
                    f"in {bad.sum()}/{self.n_particles} particles"
                )

            # =========================
            # LIKELIHOOD WEIGHTING — p(y_{k+1} | x_i^{k+1}) = N(y; h(x_i, 0), R)
            # =========================
            zeros_obs_noise = np.tile(
                self.zeros_dim_xy_1[self.dim_x :], (self.n_particles, 1, 1)
            )  # (N, dim_y, 1)
            hx = self.param.h(
                particles_propagated, zeros_obs_noise, self.dt
            )  # (N, dim_y, 1)

            innovations: np.ndarray = new_ykp1 - hx  # (N, dim_y, 1)

            tmp = np.matmul(self._cached["R_inv"], innovations)  # (N, dim_y, 1)
            quad = np.matmul(innovations.transpose(0, 2, 1), tmp)  # (N, 1, 1)
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

            # Update particles
            particles_current = particles_propagated

            # Clip diverging particles
            PARTICLE_CLIP = 1e6
            n_clipped = np.sum(
                ~np.isfinite(particles_current)
                | (np.abs(particles_current) > PARTICLE_CLIP)
            )
            particles_current = np.clip(
                np.where(np.isfinite(particles_current), particles_current, 0.0),
                -PARTICLE_CLIP,
                PARTICLE_CLIP,
            )

            # =========================
            # POSTERIOR ESTIMATE
            # =========================
            particles_temp: np.ndarray = particles_current.squeeze(-1)  # (N, dim_x)
            Xkp1_update: np.ndarray = np.average(
                particles_temp, axis=0, weights=weights
            )[:, None]

            dx = particles_temp - Xkp1_update.T  # (N, dim_x)
            PXXkp1_update: np.ndarray = (weights[:, None] * dx).T @ dx
            self._check_covariance(PXXkp1_update, step.k, name="PXXkp1_update")

            ess_before_resample = 1.0 / np.sum(weights**2)
            logger.debug(
                f"Step {new_k}: ESS={ess_before_resample:.1f}/{self.n_particles}, "
                f"n_clipped={n_clipped}, "
                f"max_innov={np.abs(innovations).max():.3g}, "
                f"Xupdate={Xkp1_update.flatten()}"
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
            except Exception as e:
                raise StepValidationError(
                    f"Step {new_k}: PKFStep construction failed in process_filter.",
                    step=new_k,
                ) from e

            self.history.record(step)

            if self.verbose > 1:
                rich_show_fields(step, title=f"Step {new_k} Update")

            yield new_k, new_xkp1, new_ykp1, step.Xkp1_predict, step.Xkp1_update
