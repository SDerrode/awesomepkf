"""
####################################################################
Linear Pairwise Kalman filter (PKF) implementation
####################################################################
"""

from __future__ import annotations

from collections.abc import Generator

import numpy as np

from prg.classes.ParamLinear import ParamLinear
from prg.classes.ParamNonLinear import ParamNonLinear
from prg.classes.PKF import PKF
from prg.utils.exceptions import FilterError, InvertibilityError, NumericalError

__all__ = ["Linear_PKF"]


class Linear_PKF(PKF):
    """
    Linear Pairwise Kalman Filter (PKF).

    Implements the coupled Kalman filter for linear state-space models.
    The transition and observation models are assumed to be linear, allowing
    the Jacobians to be replaced by the constant matrices ``A`` and ``B``
    from the parameter object.

    The filter operates as a generator: it consumes observations one by one
    and yields the filter outputs at each time step.

    Attributes
    ----------
    param : ParamLinear | ParamNonLinear
        Object holding the model parameters (transition matrices,
        noise covariances, etc.).
    """

    def __init__(
        self,
        param: ParamLinear | ParamNonLinear,
        sKey: int | None = None,
        verbose: int = 0,
    ) -> None:
        """
        Initialise the Linear PKF filter.

        Parameters
        ----------
        param : ParamLinear | ParamNonLinear
            Object holding the model parameters (transition matrices ``A``,
            ``B``, noise covariance ``Q``, etc.).
        sKey : int, optional
            Random seed for reproducibility (default ``None``).
        verbose : int, optional
            Verbosity level passed to the parent class (default ``0``).
        """
        super().__init__(param, sKey, verbose)

        self._A: np.ndarray = self.param.A
        self._AT: np.ndarray = self.param.A.T
        self._BmQBT: np.ndarray = self.param.B @ self.param.mQ @ self.param.B.T

    def process_filter(
        self,
        N: int | None = None,
        data_generator: Generator[tuple[int, np.ndarray, np.ndarray], None, None] | None = None,
    ) -> Generator[
        tuple[int, np.ndarray | None, np.ndarray, np.ndarray, np.ndarray],
        None,
        None,
    ]:
        """
        Run the Linear PKF filter as a generator.

        At each time step, the method performs a prediction step using the
        constant linear matrices ``A`` and ``B``, then an update step upon
        receiving a new observation, and yields the current filter outputs.

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

        # print(f"self.param.pairwiseModel:={self.param.pairwiseModel:}")
        # print(f"self.param.augmented:={self.param.augmented:}")

        generator = (
            data_generator if data_generator is not None else self._data_generation()
        )

        # --- First estimate -----------------------------------------------------------
        step = self._firstEstimate(generator)
        if step.xkp1 is None:
            self.ground_truth = False

        yield step.k, step.xkp1, step.ykp1, step.Xkp1_predict, step.Xkp1_update

        # --- Subsequent steps ---------------------------------------------------------
        P_augmented: np.ndarray = self.zeros_dim_xy_xy.copy()
        z_augmented: np.ndarray = self.zeros_dim_xy_1.copy()

        while N is None or step.k < N:

            # Assemble augmented state vector [X_update ; y]
            z_augmented[: self.dim_x] = step.Xkp1_update
            z_augmented[self.dim_x :] = step.ykp1

            # Prediction step
            Zkp1_predict: np.ndarray = self.param.g(
                z_augmented, self.zeros_dim_xy_1, self.dt
            )
            # Embed the state covariance into the augmented covariance matrix
            P_augmented[: self.dim_x, : self.dim_x] = step.PXXkp1_update
            Pkp1_predict: np.ndarray = self._A @ P_augmented @ self._AT + self._BmQBT

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
                # Known numerical errors — let them propagate as-is
                raise
            except Exception as e:
                raise FilterError(
                    f"Step {new_k}: unexpected error during update step."
                ) from e

            yield step.k, step.xkp1, step.ykp1, step.Xkp1_predict, step.Xkp1_update
