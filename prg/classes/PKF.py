#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Standard library
from __future__ import annotations
from typing import Generator, Optional
from dataclasses import dataclass
import logging

# Third-party
import numpy as np
from scipy.linalg import cho_factor, cho_solve, LinAlgError
from rich import print

# Local imports
from prg.classes.HistoryTracker import HistoryTracker
from prg.classes.SeedGenerator import SeedGenerator
from prg.classes.ParamLinear import ParamLinear
from prg.classes.ParamNonLinear import ParamNonLinear
from prg.classes.MatrixDiagnostics import CovarianceMatrix, InvertibleMatrix
from prg.exceptions import (
    CovarianceError,
    FilterError,
    InvertibilityError,
    ParamError,
    StepValidationError,
)
from prg.utils.utils import rich_show_fields

logger = logging.getLogger(__name__)

__all__ = ["PKFStep", "PKF"]


@dataclass(slots=True, frozen=True)
class PKFStep:
    """
    Immutable container for the outputs of one PKF filter step.

    Stores both the predicted and updated state estimates, along with
    the associated covariance matrices, innovation, and Kalman gain.

    .. note::
        This dataclass uses ``frozen=True`` — validation methods must never
        write to any attribute, or a ``FrozenInstanceError`` will be raised.

    Attributes
    ----------
    k : int
        Time step index.
    xkp1 : np.ndarray or None
        Ground truth state at step ``k``, shape ``(dim_x, 1)``.
        ``None`` if no ground truth is available.
    ykp1 : np.ndarray
        Observation vector at step ``k``, shape ``(dim_y, 1)``.
    Xkp1_predict : np.ndarray
        Predicted state estimate, shape ``(dim_x, 1)``.
    PXXkp1_predict : np.ndarray
        Predicted state covariance matrix, shape ``(dim_x, dim_x)``.
    ikp1 : np.ndarray or None
        Innovation vector ``y - H*X_predict``, shape ``(dim_y, 1)``.
    Skp1 : np.ndarray or None
        Innovation covariance matrix, shape ``(dim_y, dim_y)``.
    Kkp1 : np.ndarray or None
        Kalman gain matrix, shape ``(dim_x, dim_y)``.
    Xkp1_update : np.ndarray or None
        Updated (posterior) state estimate, shape ``(dim_x, 1)``.
    PXXkp1_update : np.ndarray or None
        Updated state covariance matrix (maybe Joseph form), shape ``(dim_x, dim_x)``.
    """

    k: int
    xkp1: Optional[np.ndarray]
    ykp1: np.ndarray
    Xkp1_predict: np.ndarray
    PXXkp1_predict: np.ndarray

    # Optional fields — None at the prediction-only step
    ikp1: Optional[np.ndarray] = None
    Skp1: Optional[np.ndarray] = None
    Kkp1: Optional[np.ndarray] = None
    Xkp1_update: Optional[np.ndarray] = None
    PXXkp1_update: Optional[np.ndarray] = None


class PKF:
    """
    Base class for Pairwise Kalman Filters (PKF).

    Provides the shared infrastructure for all PKF variants: parameter
    validation, matrix pre-allocation, random number generation, data
    simulation, covariance diagnostics, and the core prediction/update steps.

    Subclasses must implement :meth:`process_filter`.

    Attributes
    ----------
    param : ParamLinear | ParamNonLinear
        Model parameters (transition function, noise covariances, etc.).
    verbose : int
        Verbosity level: ``0`` = silent, ``1`` = warnings only, ``2`` = debug.
    dt : int
        Time step, fixed to ``1``.
    ground_truth : bool
        Whether ground truth data is available. Set to ``False`` if the
        first observation yields ``xkp1 = None``.
    history : HistoryTracker
        Records all filter steps for post-processing.
    """

    def __init__(
        self,
        param: ParamLinear | ParamNonLinear,
        sKey: Optional[int] = None,
        verbose: int = 0,
    ) -> None:
        """
        Initialise the PKF base class.

        Parameters
        ----------
        param : ParamLinear | ParamNonLinear
            Object holding the model parameters.
        sKey : int, optional
            Random seed for reproducibility (default ``None``).
        verbose : int, optional
            Verbosity level: ``0`` = silent, ``1`` = warnings, ``2`` = debug
            (default ``0``).

        Raises
        ------
        TypeError
            If ``param`` is not an instance of ``ParamLinear`` or ``ParamNonLinear``.
        ParamError
            If ``sKey`` is not a strictly positive integer or ``None``.
        ParamError
            If ``verbose`` is not in ``{0, 1, 2}``.
        """
        if not isinstance(param, (ParamLinear, ParamNonLinear)):
            raise TypeError(
                "param must be an instance of ParamLinear or ParamNonLinear"
            )
        if not ((isinstance(sKey, int) and sKey > 0) or sKey is None):
            raise ParamError("sKey must be None or a strictly positive integer")
        if verbose not in [0, 1, 2]:
            raise ParamError("verbose must be 0, 1 or 2")

        self.param = param
        self.verbose = verbose
        self.dt: int = 1  # Time step — fixed to 1 throughout

        # Random number generator
        self.__randSimulation = SeedGenerator(sKey)

        # Ground truth availability — set to False if xkp1 is None at first step
        self.ground_truth: bool = True

        # Dimension shortcuts
        self.dim_x: int = self.param.dim_x
        self.dim_y: int = self.param.dim_y
        self.dim_xy: int = self.param.dim_xy

        # Model shortcuts — avoids repeated attribute lookups in tight loops
        self.mz0: np.ndarray = self.param._mz0
        self.Pz0: np.ndarray = self.param._Pz0

        # Pre-allocated constant matrices — reused across all filter steps
        self.eye_dim_y: np.ndarray = np.eye(self.dim_y)
        self.eye_dim_x: np.ndarray = np.eye(self.dim_x)
        self.zeros_dim_x_y: np.ndarray = np.zeros((self.dim_x, self.dim_y))
        self.zeros_dim_y_1: np.ndarray = np.zeros((self.dim_y, 1))
        self.zeros_dim_xy_1: np.ndarray = np.zeros((self.dim_xy, 1))
        self.zeros_dim_xy: np.ndarray = np.zeros(self.dim_xy)
        self.zeros_dim_xy_xy: np.ndarray = np.zeros((self.dim_xy, self.dim_xy))

        # History tracker
        self.history = HistoryTracker(self.verbose)

        # Logger — level is always DEBUG on the logger itself;
        # the handler filters according to the verbose setting.
        self.logger = logging.getLogger(f"{self.__class__.__name__}.{id(self)}")
        self.logger.setLevel(logging.DEBUG)

        if not self.logger.handlers:
            ch = logging.StreamHandler()
            if verbose == 0:
                ch.setLevel(logging.CRITICAL + 1)  # Nothing is displayed
            elif verbose == 1:
                ch.setLevel(logging.WARNING)  # Warnings and errors only
            else:
                ch.setLevel(logging.DEBUG)  # Everything is displayed

            formatter = logging.Formatter(
                fmt="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
            ch.setFormatter(formatter)
            self.logger.addHandler(ch)

    # ------------------------------------------------------------------
    # Data simulation & processing
    # ------------------------------------------------------------------

    def simulate_N_data(self, N: int) -> list[tuple[int, np.ndarray, np.ndarray]]:
        """
        Simulate ``N`` steps of data and return them as a list.

        Parameters
        ----------
        N : int
            Number of steps to simulate.

        Returns
        -------
        list of tuple[int, np.ndarray, np.ndarray]
            Each tuple is ``(k, x_true, y_observed)``.

        Raises
        ------
        ParamError
            If ``N`` is not a strictly positive integer.
        """
        self._validate_N(N)
        return list(self._data_generation(N))

    def process_N_data(
        self,
        N: Optional[int],
        data_generator: Optional[Generator] = None,
    ) -> list[tuple[int, Optional[np.ndarray], np.ndarray, np.ndarray, np.ndarray]]:
        """
        Run the filter for ``N`` steps and return all outputs as a list.

        Parameters
        ----------
        N : int or None
            Number of steps to process. If ``None``, runs until the generator
            is exhausted.
        data_generator : Generator, optional
            External data generator. If ``None``, the internal generator is used.

        Returns
        -------
        list of tuple
            Each tuple is ``(k, x_true, y_observed, X_predict, X_update)``.

        Raises
        ------
        ParamError
            If ``N`` is invalid.
        FilterError
            If the filter raises an unhandled runtime error.
        """
        try:
            result = self.process_filter(N=N, data_generator=data_generator)
        except RuntimeError as e:
            raise FilterError("Unexpected runtime error in process_filter.") from e
        return list(result)

    @staticmethod
    def _validate_N(N: Optional[int]) -> None:
        """
        Validate the ``N`` parameter shared by all filter variants.

        Parameters
        ----------
        N : int or None
            Number of steps. Must be a strictly positive integer or ``None``.

        Raises
        ------
        ParamError
            If ``N`` is not a strictly positive integer or ``None``.
        """
        if not ((isinstance(N, int) and N > 0) or N is None):
            raise ParamError("N must be None or a strictly positive integer")

    # ------------------------------------------------------------------
    # Data generation
    # ------------------------------------------------------------------

    def _data_generation(
        self, N: Optional[int] = None
    ) -> Generator[tuple[int, np.ndarray, np.ndarray], None, None]:
        """
        Simulate state-space data and yield one step at a time.

        Handles both standard and augmented model structures. At each step,
        process noise is sampled and the transition function ``g`` is applied.

        Parameters
        ----------
        N : int or None
            Number of steps to generate. If ``None``, generates indefinitely.

        Yields
        ------
        k : int
            Current time step index.
        Xkp1_simul : np.ndarray
            Simulated state vector at step ``k``, shape ``(dim_x, 1)``.
        Ykp1_simul : np.ndarray
            Simulated observation vector at step ``k``, shape ``(dim_y, 1)``.
        """
        Zkp1_simul = np.zeros((self.dim_xy, 1))

        # First step — sample initial state from prior distribution
        if self.param.augmented:
            Zkp1_simul[: self.dim_x, 0] = self.__randSimulation.rng.multivariate_normal(
                mean=self.mz0[: self.dim_x, 0],
                cov=self.Pz0[: self.dim_x, : self.dim_x],
            )
            Zkp1_simul[self.dim_x :, 0] = Zkp1_simul[
                self.dim_x - self.dim_y : self.dim_x, 0
            ]
        else:
            Zkp1_simul[:, 0] = self.__randSimulation.rng.multivariate_normal(
                mean=self.mz0[:, 0], cov=self.Pz0
            )

        Xkp1_simul, Ykp1_simul = np.split(Zkp1_simul, [self.dim_x])
        k = 0
        yield k, Xkp1_simul, Ykp1_simul

        # Subsequent steps — propagate state and sample process noise
        zerosvector_xy = np.zeros(self.dim_xy)
        zerosvector_x = np.zeros(self.dim_x)
        noise_z = np.zeros((self.dim_xy, 1))

        while N is None or k < N:
            if self.param.augmented:
                noise_z[: self.dim_x, 0] = (
                    self.__randSimulation.rng.multivariate_normal(
                        mean=zerosvector_x,
                        cov=self.param.mQ[: self.dim_x, : self.dim_x],
                    )
                )
                noise_z[self.dim_x :, 0] = noise_z[
                    self.dim_x - self.dim_y : self.dim_x, 0
                ]
            else:
                noise_z[:, 0] = self.__randSimulation.rng.multivariate_normal(
                    mean=zerosvector_xy, cov=self.param.mQ
                )
            Zkp1_simul = self.param.g(Zkp1_simul, noise_z, self.dt)
            Xkp1_simul, Ykp1_simul = np.split(Zkp1_simul, [self.dim_x])
            k += 1
            yield k, Xkp1_simul, Ykp1_simul

    # ------------------------------------------------------------------
    # Covariance diagnostics
    # ------------------------------------------------------------------

    def _check_covariance(self, mat: np.ndarray, k: int, name: str = "") -> None:
        """
        Check whether a matrix is a valid covariance matrix using
        :class:`CovarianceMatrix` diagnostics.

        Skipped entirely for augmented models. Logs a warning on WARNING
        status and the full diagnostic report at DEBUG level. Attempts
        regularization on FAIL status before raising.

        Parameters
        ----------
        mat : np.ndarray
            Matrix to validate, shape ``(n, n)``.
        k : int
            Current time step index, used in log messages.
        name : str, optional
            Name of the matrix, used in log messages.

        Raises
        ------
        CovarianceError
            If the matrix is not a valid covariance matrix and regularization fails.
        """
        if self.param.augmented:
            return

        report = CovarianceMatrix(mat).check()
        if not report.is_ok:
            if self.verbose > 1:
                self.logger.warning("Step %d: %s — %s", k, name, report.overall_status)

            if not report.is_valid:
                try:
                    mat[:] = CovarianceMatrix(mat).regularized()
                    if self.verbose > 1:
                        self.logger.warning(
                            "Step %d: %s regularized successfully.", k, name
                        )
                except ValueError as e:
                    self.logger.error(
                        "Step %d: %s regularization failed — %s", k, name, e
                    )
                    raise CovarianceError(
                        f"Step {k}: {name} is not a valid covariance matrix "
                        f"and could not be regularized.",
                        matrix_name=name,
                        step=k,
                    ) from e

    def _check_invertible(self, mat: np.ndarray, k: int, name: str = "") -> bool:
        """
        Check whether a matrix is invertible using :class:`InvertibleMatrix`
        diagnostics.

        Logs a warning on WARNING status and the full diagnostic report at
        DEBUG level. Raises on FAIL status.

        Parameters
        ----------
        mat : np.ndarray
            Matrix to validate, shape ``(n, n)``.
        k : int
            Current time step index, used in log messages.
        name : str, optional
            Name of the matrix, used in log messages.

        Returns
        -------
        bool
            ``True`` if the matrix is invertible (is_valid).

        Raises
        ------
        InvertibilityError
            If the matrix is not invertible (FAIL status).
        """
        report = InvertibleMatrix(mat).check()

        if not report.is_ok:
            self.logger.warning(
                "Step %d: %s has invertibility status %s.",
                k,
                name,
                report.overall_status,
            )
            if self.verbose > 1:
                self.logger.debug("Step %d: %s full diagnostic:\n%s", k, name, report)

            if not report.is_valid:
                raise InvertibilityError(
                    f"Step {k}: matrix {name} is not invertible (FAIL).",
                    matrix_name=name,
                    step=k,
                )

        return report.is_valid

    # ------------------------------------------------------------------
    # First estimate
    # ------------------------------------------------------------------

    def _firstEstimate(
        self,
        generator: Generator[tuple[int, Optional[np.ndarray], np.ndarray], None, None],
    ) -> PKFStep:
        """
        Compute the initial filter estimate from the first data point.

        The initial state estimate is obtained via Gaussian conditioning
        of the prior ``p(x_0)`` on the first observation ``y_0``:

            X_update   = mu_x + Sigma_xy @ Sigma_yy^{-1} @ (y - mu_y)
            PXX_update = Sigma_xx - Sigma_xy @ Sigma_yy^{-1} @ Sigma_yx

        Parameters
        ----------
        generator : Generator
            Data generator yielding ``(k, x_true, y_observed)`` tuples.

        Returns
        -------
        PKFStep
            The first filter step with predicted and updated estimates.

        Raises
        ------
        InvertibilityError
            If ``Sigma22`` (prior observation covariance) is not invertible.
        CovarianceError
            If ``PXXkp1_update`` is not a valid covariance matrix.
        StepValidationError
            If ``PKFStep`` construction fails due to invalid data.
        """
        k, xkp1, ykp1 = next(generator)

        # Gaussian conditioning on the first observation
        mu_x0, mu_y0 = np.split(self.mz0, [self.dim_x])
        Sigma11 = self.Pz0[: self.dim_x, : self.dim_x]
        Sigma12 = self.Pz0[: self.dim_x, self.dim_x :]
        Sigma21 = self.Pz0[self.dim_x :, : self.dim_x]
        Sigma22 = self.Pz0[self.dim_x :, self.dim_x :]

        # Validate Sigma22 before inversion — raises InvertibilityError on failure
        self._check_invertible(Sigma22, k, name="Sigma22")

        Sigma22_inv: np.ndarray = np.linalg.inv(Sigma22)
        Xkp1_update: np.ndarray = mu_x0 + Sigma12 @ Sigma22_inv @ (ykp1 - mu_y0)
        PXXkp1_update: np.ndarray = Sigma11 - Sigma12 @ Sigma22_inv @ Sigma21

        # Validate updated covariance — raises CovarianceError on failure
        self._check_covariance(PXXkp1_update, k, name="PXXkp1_update")

        try:
            step = PKFStep(
                k=k,
                xkp1=xkp1.copy() if xkp1 is not None else None,
                ykp1=ykp1.copy(),
                Xkp1_predict=np.zeros((self.dim_x, 1)),
                PXXkp1_predict=self.eye_dim_x.copy(),
                ikp1=self.zeros_dim_y_1.copy(),
                Skp1=self.eye_dim_y.copy(),
                Kkp1=self.zeros_dim_x_y.copy(),
                Xkp1_update=Xkp1_update.copy(),
                PXXkp1_update=PXXkp1_update,
            )
        except (ValueError, LinAlgError) as e:
            self.logger.error("Step %d: PKFStep construction failed — %s", k, e)
            raise StepValidationError(
                f"Step {k}: PKFStep construction failed in _firstEstimate.",
                step=k,
            ) from e

        self.history.record(step)
        if self.verbose > 1:
            rich_show_fields(step, title="First Estimate")

        return step

    # ------------------------------------------------------------------
    # Update step
    # ------------------------------------------------------------------

    def _nextUpdating(
        self,
        k: int,
        xkp1: Optional[np.ndarray],
        ykp1: np.ndarray,
        Zkp1_predict: np.ndarray,
        Pkp1_predict: np.ndarray,
        store: bool = True,
    ) -> PKFStep:
        """
        Perform one Kalman update step given a new observation.

        Computes the innovation, Kalman gain, and updated state estimate.
        The updated covariance is computed using the numerically stable
        Joseph form to preserve positive semi-definiteness.

        Parameters
        ----------
        k : int
            Current time step index.
        xkp1 : np.ndarray or None
            Ground truth state at step ``k``, shape ``(dim_x, 1)``.
            ``None`` if unavailable.
        ykp1 : np.ndarray
            Observation vector at step ``k``, shape ``(dim_y, 1)``.
        Zkp1_predict : np.ndarray
            Augmented predicted state ``[X_predict; Y_predict]``,
            shape ``(dim_xy, 1)``.
        Pkp1_predict : np.ndarray
            Augmented predicted covariance matrix, shape ``(dim_xy, dim_xy)``.
        store : bool, optional
            Whether to record the step in the history tracker (default ``True``).

        Returns
        -------
        PKFStep
            The updated filter step.

        Raises
        ------
        InvertibilityError
            If the innovation covariance ``Skp1`` is not invertible.
        CovarianceError
            If the Cholesky factorisation fails or the updated covariance
            ``PXXkp1_update`` is not a valid covariance matrix.
        StepValidationError
            If ``PKFStep`` construction fails due to invalid data.
        """
        Xkp1_predict, Ykp1_predict = np.split(Zkp1_predict, [self.dim_x])
        PXXkp1_predict = Pkp1_predict[: self.dim_x, : self.dim_x]
        PXYkp1_predict = Pkp1_predict[: self.dim_x, self.dim_x :]
        PYXkp1_predict = Pkp1_predict[self.dim_x :, : self.dim_x]
        PYYkp1_predict = Pkp1_predict[self.dim_x :, self.dim_x :]

        ikp1: np.ndarray = ykp1 - Ykp1_predict
        Skp1: np.ndarray = PYYkp1_predict.copy()

        # Validate innovation covariance — raises InvertibilityError on failure
        self._check_invertible(Skp1, k, name="Skp1")

        # Kalman gain via Cholesky solve — more stable than direct inversion
        try:
            c, low = cho_factor(Skp1)
            Kkp1: np.ndarray = PXYkp1_predict @ cho_solve((c, low), self.eye_dim_y)
        except (LinAlgError, ValueError) as e:
            self.logger.error(
                "Step %d: Cholesky factorisation failed on Skp1 — %s", k, e
            )
            raise CovarianceError(
                f"Step {k}: Cholesky factorisation failed — Skp1 may not be "
                f"positive definite.",
                matrix_name="Skp1",
                step=k,
            ) from e

        Xkp1_update: np.ndarray = Xkp1_predict + Kkp1 @ ikp1

        # Joseph form for numerical stability
        Joseph_factor: np.ndarray = np.vstack((self.eye_dim_x, -Kkp1.T))
        PXXkp1_update_Joseph: np.ndarray = (
            Joseph_factor.T @ Pkp1_predict @ Joseph_factor
        )

        # Validate updated covariance — raises CovarianceError on failure
        self._check_covariance(PXXkp1_update_Joseph, k, name="PXXkp1_update_Joseph")

        try:
            step = PKFStep(
                k=k,
                xkp1=xkp1.copy() if xkp1 is not None else None,
                ykp1=ykp1.copy(),
                Xkp1_predict=Xkp1_predict.copy(),
                PXXkp1_predict=PXXkp1_predict.copy(),
                ikp1=ikp1.copy(),
                Skp1=Skp1.copy(),
                Kkp1=Kkp1.copy(),
                Xkp1_update=Xkp1_update.copy(),
                PXXkp1_update=PXXkp1_update_Joseph.copy(),
            )
        except (ValueError, LinAlgError) as e:
            self.logger.error("Step %d: PKFStep construction failed — %s", k, e)
            raise StepValidationError(
                f"Step {k}: PKFStep construction failed in _nextUpdating.",
                step=k,
            ) from e

        if store:
            self.history.record(step)

        if self.verbose > 1:
            rich_show_fields(step, title=f"Step {k} Update")

        return step
