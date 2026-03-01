#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
####################################################################
Extended Pairwise Kalman filter (EPKF) implementation
####################################################################
"""

from __future__ import annotations  # Implicit use for type annotations
from typing import Generator, Optional, Union  # Used in signatures
from scipy.linalg import LinAlgError  # Used in try/except
import numpy as np  # Used throughout

from .PKF import PKF  # Parent class


class NonLinear_EPKF(PKF):
    """
    Extended Pairwise Kalman Filter (EPKF).

    Extends :class:`PKF` by introducing the EPKF.

    """

    def __init__(
        self,
        param: ParamLinear | ParamNonLinear,
        sKey: Optional[int] = None,
        verbose: int = 0,
    ) -> None:
        """
        Initialise the EPKF filter.

        Parameters
        ----------
        param : ParamLinear | ParamNonLinear
            Object holding the model parameters (transition function,
            Jacobians, noise covariances, etc.).
        sKey : int, optional
            Random seed for reproducibility (default ``None``).
        verbose : int, optional
            Verbosity level passed to the parent class (default ``0``).
        """
        super().__init__(param, sKey, verbose)

    def process_filter(
        self,
        N: Optional[int] = None,
        data_generator: Optional[
            Generator[tuple[int, np.ndarray, np.ndarray], None, None]
        ] = None,
    ) -> Generator[
        tuple[int, Optional[np.ndarray], np.ndarray, np.ndarray, np.ndarray]
    ]:
        """
        Run the EPKF filter as a generator.

        At each time step, the method yields the current filter outputs.
        Data is consumed either from ``data_generator`` if provided, or from
        the internal :meth:`_data_generation` method.

        Parameters
        ----------
        N : int, optional
            Maximum number of time steps to process. If ``None`` (default),
            the filter runs until the data generator is exhausted.
        data_generator : Generator, optional
            External data generator yielding tuples
            ``(k, x_true, y_observed)`` at each time step, where:

            - ``k``          : int         — time step index
            - ``x_true``     : np.ndarray  — ground truth state, shape ``(dim_x, 1)``;
                               may be ``None`` if no ground truth is available
            - ``y_observed`` : np.ndarray  — observation vector, shape ``(dim_y, 1)``

            If ``None``, the internal generator is used.

        Yields
        ------
        k : int
            Current time step index.
        x_true : np.ndarray or None
            Ground truth state at step ``k``, shape ``(dim_x, 1)``.
            ``None`` if ground truth is unavailable.
        y_observed : np.ndarray
            Observation vector at step ``k``, shape ``(dim_y, 1)``.
        X_predict : np.ndarray
            Predicted (prior) state estimate at step ``k``, shape ``(dim_x, 1)``.
        X_update : np.ndarray
            Updated (posterior) state estimate at step ``k``, shape ``(dim_x, 1)``.

        Raises
        ------
        ValueError
            If ``N`` is not a strictly positive integer or ``None``.
        ValueError
            If a Jacobian returns a matrix with an unexpected shape.
        LinAlgError
            If a linear algebra error occurs during the update step
            (e.g. non-invertible innovation covariance matrix).
        """

        self._validate_N(N)

        generator = (
            data_generator if data_generator is not None else self._data_generation()
        )

        # Short-cut to the Jacobian function
        jg = self.param.jacobiens_g

        # --- First estimate -----------------------------------------------------------
        step = self._firstEstimate(generator)

        if step.xkp1 is None:  # There is no ground truth
            self.ground_truth = False

        yield step.k, step.xkp1, step.ykp1, step.Xkp1_predict, step.Xkp1_update

        # --- Subsequent steps ---------------------------------------------------------
        accel_xy_xy: np.ndarray = self.zeros_dim_xy_xy.copy()
        z_iterated: np.ndarray = np.zeros((self.dim_xy, 1))

        while N is None or step.k < N:

            # here ykp1 still gives the previous : it is yk indeed!
            z_iterated[: self.dim_x] = step.Xkp1_update
            z_iterated[self.dim_x :] = step.ykp1

            # Prediction
            Zkp1_predict = self.g(z_iterated, self.zeros_dim_xy_1, self.dt)
            An, Bn = jg(z_iterated, self.zeros_dim_xy_1, self.dt)
            if An.shape != (self.dim_xy, self.dim_xy) or Bn.shape != (
                self.dim_xy,
                self.dim_xy,
            ):
                raise ValueError(
                    f"Jacobian returned matrices of wrong shape: An={An.shape}, Bn={Bn.shape}"
                )
            accel_xy_xy[: self.dim_x, : self.dim_x] = step.PXXkp1_update
            Pkp1_predict = An @ accel_xy_xy @ An.T + Bn @ self.mQ @ Bn.T
            self._check_covariance(Pkp1_predict, step.k, name="Pkp1_predict")

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
            except Exception as e:
                # self.logger.error(f"Step {new_k}: LinAlgError during update")
                raise

            yield step.k, step.xkp1, step.ykp1, step.Xkp1_predict, step.Xkp1_update
