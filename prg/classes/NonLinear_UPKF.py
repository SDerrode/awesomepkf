"""
####################################################################
Unscented Pairwise Kalman filter (UPKF) implementation
####################################################################
"""

from __future__ import annotations

from collections.abc import Generator

import numpy as np

from prg.classes.PKF import PKF
from prg.classes.SigmaPointsSet import SigmaPointsSet
from prg.utils.exceptions import (
    FilterError,
    InvertibilityError,
    NumericalError,
    ParamError,
)

__all__ = ["NonLinear_UPKF"]


class NonLinear_UPKF(PKF):
    """
    Unscented Pairwise Kalman Filter (UPKF).

    Extends :class:`PKF` by introducing the UPKF.
    """

    def __init__(
        self,
        param,
        sigmaSet: str,
        sKey: int | None = None,
        verbose: int = 0,
    ) -> None:
        """
        Initialise the UPKF filter.

        Parameters
        ----------
        param : ParamLinear | ParamNonLinear
            Model parameters.
        sigmaSet : str
            Key of the sigma-point set in ``SigmaPointsSet.registry``.
        sKey : int, optional
            Random seed for reproducibility.
        verbose : int, optional
            Verbosity level (default 0).

        Raises
        ------
        ParamError
            If ``sigmaSet`` is not a known key in the registry.
        """
        super().__init__(param, sKey, verbose)

        try:
            cls = SigmaPointsSet.registry[sigmaSet]
        except KeyError:
            raise ParamError(
                f"Jeu de sigma-points inconnu : {sigmaSet!r}. "
                f"Disponibles : {list(SigmaPointsSet.registry.keys())}."
            )

        self.sigma_point_set_obj = cls(
            dim=2 * self.dim_x + self.dim_y, param=self.param
        )

    def process_filter(
        self,
        N: int | None = None,
        data_generator: Generator[tuple[int, np.ndarray, np.ndarray], None, None] | None = None,
    ) -> Generator[tuple[int, np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
        """
        Runs the UPKF filter as a generator.

        Parameters
        ----------
        N : int, optional
            Maximum number of time steps. If ``None``, runs until the data generator is exhausted.
        data_generator : Generator, optional
            External data generator. If ``None``, the internal generator is used.

        Yields
        ------
        k : int
            Current time index.
        x_true : np.ndarray or None
            Ground truth at time ``k``.
        y_observed : np.ndarray
            Observation at time ``k``.
        X_predict : np.ndarray
            Prior estimate, shape ``(dim_x, 1)``.
        X_update : np.ndarray
            Posterior estimate, shape ``(dim_x, 1)``.

        Raises
        ------
        ParamError
            If ``N`` is not a strictly positive integer or ``None``
            (raised by :meth:`_validate_N` in the parent).
        InvertibilityError
            If the innovation covariance matrix ``Skp1`` is not
            invertible during the update step.
        NumericalError
            If the predicted covariance matrix ``Pkp1_predict`` is not
            valid (raised by :meth:`_check_covariance`).
        FilterError
            If an unexpected error occurs during the update step.
        """
        self._validate_N(N)
        self.history.clear()

        generator = (
            data_generator if data_generator is not None else self._data_generation()
        )

        # --- First estimate -----------------------------------------------------------
        step = self._firstEstimate(generator)
        if step.xkp1 is None:
            self.ground_truth = False

        yield step.k, step.xkp1, step.ykp1, step.Xkp1_predict, step.Xkp1_update

        # --- Subsequent steps ---------------------------------------------------------
        za = np.zeros((2 * self.dim_x + self.dim_y, 1))
        Pa_base = np.zeros((2 * self.dim_x + self.dim_y, 2 * self.dim_x + self.dim_y))
        Pa_base[self.dim_x :, self.dim_x :] = self.param.mQ
        Pkp1_predict = self.zeros_dim_xy_xy.copy()

        while N is None or step.k < N:

            # Sigma points and their propagation through g
            za[: self.dim_x] = step.Xkp1_update
            Pa = Pa_base.copy()
            Pa[: self.dim_x, : self.dim_x] = step.PXXkp1_update

            sigma_without_y = self.sigma_point_set_obj._sigma_point(za, Pa)

            # Vectorisation : stack (n_sigma, 2*dim_x + dim_y, 1)
            sigma_stack = np.array(sigma_without_y)  # (n_sigma, 2*dim_x+dim_y, 1)
            n_sigma = sigma_stack.shape[0]
            ykp1_tiled = np.tile(step.ykp1, (n_sigma, 1, 1))  # (n_sigma, dim_y, 1)

            # Insert ykp1 between the first dim_x elements and the rest (process noise)
            sigma_with_y = np.concatenate(
                [
                    sigma_stack[:, : self.dim_x, :],  # (n_sigma, dim_x, 1)
                    ykp1_tiled,  # (n_sigma, dim_y, 1)
                    sigma_stack[:, self.dim_x :, :],  # (n_sigma, dim_x, 1)
                ],
                axis=1,
            )  # (n_sigma, dim_xy + dim_x, 1)

            # Vectorised call to g — single batch call instead of n_sigma scalar calls
            z_batch, noise_batch = np.split(sigma_with_y, [self.dim_xy], axis=1)
            sigma_propag = self.param.g(
                z_batch, noise_batch, self.dt
            )  # (n_sigma, dim_xy, 1)

            # Prediction
            Zkp1_predict = np.sum(
                self.sigma_point_set_obj.Wm[:, None, None] * sigma_propag, axis=0
            )

            diffs = sigma_propag - Zkp1_predict  # (n_sigma, dim_xy, 1)
            Pkp1_predict = np.einsum(
                "i,ijk,ilk->jl", self.sigma_point_set_obj.Wc, diffs, diffs
            )

            # Validate predicted covariance — raises CovarianceError if invalid
            self._check_covariance(Pkp1_predict, step.k, name="Pkp1_predict")

            # Consume the next observation
            try:
                new_k, new_xkp1, new_ykp1 = next(generator)
            except StopIteration:
                return  # Data generator exhausted — normal stop, not an error

            # Update step — custom exceptions propagate naturally
            try:
                step = self._nextUpdating(
                    new_k, new_xkp1, new_ykp1, Zkp1_predict, Pkp1_predict
                )
            except (InvertibilityError, NumericalError):
                raise
            except Exception as e:
                raise FilterError(
                    f"Step {new_k}: unexpected error during update step."
                ) from e

            yield step.k, step.xkp1, step.ykp1, step.Xkp1_predict, step.Xkp1_update
