"""
####################################################################
Linear Pairwise Kalman filter (PKF) implementation
####################################################################
"""

from __future__ import annotations

import itertools
from collections.abc import Generator

import numpy as np

from prg.classes.param_linear import ParamLinear
from prg.classes.param_nonlinear import ParamNonLinear
from prg.classes.pkf import PKF
from prg.utils.exceptions import (
    FilterError,
    InvertibilityError,
    NumericalError,
    ParamError,
)

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
            - ``y_observed`` : np.ndarray  — observation vector, shape ``(dim_y, 1)``.
                               An **all-NaN** vector marks a *missing*
                               observation: the update is skipped and the full
                               joint covariance is carried to the next
                               prediction (exact marginalisation over the
                               missing ``y``). Partially-NaN observations
                               raise ``ParamError``, and the first observation
                               must not be missing.

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
        # The first observation must be present and fully observed: the filter
        # is initialised by conditioning on y_0. Peek, validate, then chain
        # the (normalised) item back for _firstEstimate.
        try:
            first_item = next(generator)
        except StopIteration:
            raise FilterError(
                "Linear_PKF: the data generator yielded no items."
            ) from None
        first_k = first_item[0]
        y0, y0_missing = self._classify_observation(first_k, first_item[2])
        if y0_missing:
            raise ParamError(
                "The first observation must not be missing (NaN): the filter "
                "is initialised by conditioning on y_0."
            )
        generator = itertools.chain([(first_k, first_item[1], y0)], generator)

        step = self._firstEstimate(generator)
        if step.xkp1 is None:
            self.ground_truth = False

        # Joint posterior mean/covariance carried across steps. After an
        # observed update the posterior is exactly [X_update ; y] with
        # covariance blkdiag(PXX_update, 0): conditioning on the exactly
        # observed Y block zeroes its covariance (Joseph identity). After a
        # MISSING observation (all-NaN y) the full predicted covariance must
        # be carried instead — in a pairwise model y is a component of the
        # Markov chain, so a gap requires marginalising over it, which leaves
        # nonzero Y and cross blocks. Rebuilding the block-diagonal form there
        # (the classical "skip the update" recipe) silently misestimates the
        # state and can destabilise the filter.
        # Seeded BEFORE the yield: the consumer holds the yielded arrays until
        # it resumes us, so an in-place mutation there must not reach the
        # recursion (every later step also seeds before its yield).
        z_joint: np.ndarray = self.zeros_dim_xy_1.copy()
        P_joint: np.ndarray = self.zeros_dim_xy_xy.copy()
        z_joint[: self.dim_x] = step.Xkp1_update
        z_joint[self.dim_x :] = step.ykp1
        P_joint[: self.dim_x, : self.dim_x] = step.PXXkp1_update

        yield step.k, step.xkp1, step.ykp1, step.Xkp1_predict, step.Xkp1_update

        # --- Subsequent steps ---------------------------------------------------------
        while N is None or step.k < N:

            # Prediction step on the joint posterior
            Zkp1_predict: np.ndarray = self.param.g(
                z_joint, self.zeros_dim_xy_1, self.dt
            )
            Pkp1_predict: np.ndarray = self._A @ P_joint @ self._AT + self._BmQBT

            # Validate predicted covariance — raises CovarianceError if invalid
            self._check_covariance(Pkp1_predict, step.k, name="Pkp1_predict")

            # Consume the next observation
            try:
                new_k, new_xkp1, new_ykp1 = next(generator)
            except StopIteration:
                return  # Data generator exhausted — normal stop, not an error

            y_arr, missing = self._classify_observation(new_k, new_ykp1)

            if missing:
                # Missing observation: no update. Posterior = prior, and the
                # FULL joint covariance is carried to the next prediction.
                try:
                    step = self._noUpdate(
                        new_k, new_xkp1, y_arr, Zkp1_predict, Pkp1_predict
                    )
                except (InvertibilityError, NumericalError, ParamError):
                    # Known custom errors — let them propagate as-is
                    raise
                except Exception as e:
                    raise FilterError(
                        f"Step {new_k}: unexpected error during "
                        f"missing-observation step."
                    ) from e
                z_joint = Zkp1_predict.copy()
                P_joint = Pkp1_predict.copy()
            else:
                # Update step — custom exceptions propagate naturally
                try:
                    step = self._nextUpdating(
                        new_k, new_xkp1, y_arr, Zkp1_predict, Pkp1_predict
                    )
                except (InvertibilityError, NumericalError, ParamError):
                    # Known custom errors — let them propagate as-is
                    raise
                except Exception as e:
                    raise FilterError(
                        f"Step {new_k}: unexpected error during update step."
                    ) from e
                z_joint[: self.dim_x] = step.Xkp1_update
                z_joint[self.dim_x :] = step.ykp1
                P_joint[:] = 0.0
                P_joint[: self.dim_x, : self.dim_x] = step.PXXkp1_update

            yield step.k, step.xkp1, step.ykp1, step.Xkp1_predict, step.Xkp1_update

    def _classify_observation(
        self, k: int, y: np.ndarray | None
    ) -> tuple[np.ndarray, bool]:
        """
        Validate an observation and classify it as observed or missing.

        Parameters
        ----------
        k : int
            Time step index, used in error messages.
        y : np.ndarray or None
            Raw observation from the data generator.

        Returns
        -------
        y_arr : np.ndarray
            The observation as a float array of shape ``(dim_y, 1)``.
        missing : bool
            ``True`` if the observation is an all-NaN gap marker.

        Raises
        ------
        ParamError
            If ``y`` is ``None``, has the wrong size, or is partially NaN.
        """
        if y is None:
            raise ParamError(
                f"Step {k}: observation is None. Mark a missing observation "
                f"with an all-NaN vector of shape ({self.dim_y}, 1)."
            )
        y_arr = np.asarray(y, dtype=float)
        if y_arr.size != self.dim_y:
            raise ParamError(
                f"Step {k}: observation has size {y_arr.size}, expected "
                f"dim_y = {self.dim_y}."
            )
        nan_mask = np.isnan(y_arr)
        if nan_mask.any() and not nan_mask.all():
            raise ParamError(
                f"Step {k}: partially missing observation (some "
                f"components NaN). Only fully missing observations "
                f"(all-NaN) are supported."
            )
        return y_arr.reshape(self.dim_y, 1), bool(nan_mask.all())
