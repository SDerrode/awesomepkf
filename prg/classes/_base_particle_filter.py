"""
Common base for particle filters (NonLinear_PF, NonLinear_PPF).

Holds the parts that were duplicated verbatim between the two filters:
- ``__init__`` skeleton (n_particles, resampling settings, RNG, cache)
- ``resample()`` — 4 resampling schemes (multinomial, systematic,
  stratified, residual)
- ``_safe_normalize_log_weights()`` — numerically robust log-sum-exp

Subclasses must implement ``_precompute()`` and ``process_filter()``,
which are filter-specific (PF uses prior-as-proposal with diagonal
Q/R; PPF uses the conditional Schur-complement update).
"""

from __future__ import annotations

import logging

import numpy as np

from prg.classes.pkf import PKF
from prg.classes.seed_generator import SeedGenerator
from prg.utils.exceptions import ParamError
from prg.utils.numerics import EPS_ABS

logger = logging.getLogger(__name__)

__all__ = ["_BaseParticleFilter"]


class _BaseParticleFilter(PKF):
    """Abstract base for particle filters; not instantiated directly."""

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
        self._randParticles = SeedGenerator()
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
                cumulative_sum, self._randParticles.rng.random(N)
            )

        elif method in ("systematic", "stratified"):
            if method == "systematic":
                positions = (np.arange(N) + self._randParticles.rng.random()) / N
            else:
                positions = (np.arange(N) + self._randParticles.rng.random(N)) / N
            indexes = np.zeros(N, dtype=int)
            i = j = 0
            while i < N:
                if positions[i] < cumulative_sum[j]:
                    indexes[i] = j
                    i += 1
                else:
                    j += 1

        elif method == "residual":
            indexes_list: list[int] = []
            num_copies = np.floor(N * weights).astype(int)
            for i in range(N):
                indexes_list += [i] * num_copies[i]
            residual = weights - num_copies / N
            residual_sum = residual.sum()
            if residual_sum > EPS_ABS:
                residual /= residual_sum
            else:
                residual = np.ones(N) / N
            cumulative_sum_res = np.cumsum(residual)
            cumulative_sum_res[-1] = 1.0
            remaining = N - len(indexes_list)
            random_vals = self._randParticles.rng.random(remaining)
            res_indexes = np.searchsorted(cumulative_sum_res, random_vals)
            indexes_list += list(res_indexes)
            indexes = np.array(indexes_list)

        else:
            raise ParamError(
                f"Unknown resampling method: {method!r}. "
                f"Expected one of: 'multinomial', 'systematic', 'stratified', 'residual'."
            )

        return indexes

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
