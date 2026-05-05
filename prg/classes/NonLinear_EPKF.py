"""
####################################################################
Extended Pairwise Kalman filter (EPKF) implementation
####################################################################
"""

from __future__ import annotations

from collections.abc import Generator

import numpy as np

from prg.classes.PKF import PKF
from prg.utils.exceptions import (
    FilterError,
    InvertibilityError,
    NumericalError,
    ParamError,
)

__all__ = ["NonLinear_EPKF"]


class NonLinear_EPKF(PKF):
    """
    Extended Pairwise Kalman Filter (EPKF).

    Extends :class:`PKF` by introducing the EPKF.
    """

    def __init__(
        self,
        param,
        sKey: int | None = None,
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
        N: int | None = None,
        data_generator: Generator[tuple[int, np.ndarray, np.ndarray], None, None] | None = None,
    ) -> Generator[
        tuple[int, np.ndarray | None, np.ndarray, np.ndarray, np.ndarray]
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
        ParamError
            If ``N`` is not a strictly positive integer or ``None``
            (raised by :meth:`_validate_N` in the parent).
        ParamError
            If a Jacobian returns a matrix of unexpected shape.
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
        accel_xy_xy: np.ndarray = self.zeros_dim_xy_xy.copy()
        z_iterated: np.ndarray = np.zeros((self.dim_xy, 1))
        expected_shape = (self.dim_xy, self.dim_xy)

        while N is None or step.k < N:

            # here ykp1 still gives the previous : it is yk indeed!
            z_iterated[: self.dim_x] = step.Xkp1_update
            z_iterated[self.dim_x :] = step.ykp1

            # Prediction
            try:
                Zkp1_predict = self.param.g(z_iterated, self.zeros_dim_xy_1, self.dt)
                An, Bn = self.param.jacobiens_g(
                    z_iterated, self.zeros_dim_xy_1, self.dt
                )

            except Exception as e:
                raise FilterError(
                    f"Step {step.k}: unexpected error during prediction step."
                ) from e

            # Validate Jacobian shapes — model parametrisation error
            if An.ndim == 2:
                if An.shape != expected_shape or Bn.shape != expected_shape:
                    raise ParamError(
                        f"Jacobian returned matrices of wrong shape: "
                        f"An={An.shape}, Bn={Bn.shape}, expected {expected_shape}."
                    )
            else:
                if An.shape[1:] != expected_shape or Bn.shape[1:] != expected_shape:
                    raise ParamError(
                        f"Jacobian returned matrices of wrong shape: "
                        f"An={An.shape}, Bn={Bn.shape}, "
                        f"expected (N, {self.dim_xy}, {self.dim_xy})."
                    )

            accel_xy_xy[: self.dim_x, : self.dim_x] = step.PXXkp1_update
            Pkp1_predict = An @ accel_xy_xy @ An.T + Bn @ self.param.mQ @ Bn.T

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
