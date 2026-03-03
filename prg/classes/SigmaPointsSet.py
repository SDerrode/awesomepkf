#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from abc import ABC, abstractmethod  # Used for the abstract base class
from typing import Dict, Type, Optional  # Used in registry and signatures
import numpy as np  # Used throughout
from itertools import product  # Used in SetIto2000 for tensor product

from prg.utils.numerics import EPS_ABS  # Used in _chol and weight validation

__all__ = ["SigmaPointsSet", "SetWAN2000", "SetCPKF", "SetLERNER2002", "SetIto2000"]


class SigmaPointsSet(ABC):
    """
    Abstract base class for all sigma-point sets.

    Subclasses are automatically registered in :attr:`registry` via
    :meth:`__init_subclass__` using a unique string ``key``. This allows
    dynamic lookup and instantiation by name (e.g. ``"wan2000"``).

    Each subclass must implement :meth:`_sigma_point`, which computes the
    set of sigma points from a mean vector and a covariance matrix.

    Attributes
    ----------
    registry : dict[str, type[SigmaPointsSet]]
        Class-level registry mapping string keys to subclass types.
    dim : int
        Dimension of the state space.
    """

    registry: Dict[str, Type["SigmaPointsSet"]] = {}

    def __init_subclass__(cls, *, key: str, **kwargs) -> None:
        """
        Automatically register each subclass under the given ``key``.

        Parameters
        ----------
        key : str
            Unique identifier for the sigma-point set (e.g. ``"wan2000"``).

        Raises
        ------
        RuntimeError
            If ``key`` is already present in the registry.
        """
        super().__init_subclass__(**kwargs)
        if key in SigmaPointsSet.registry:
            raise RuntimeError(f"Key already registered: {key}")
        SigmaPointsSet.registry[key] = cls

    def __init__(self, dim: int) -> None:
        """
        Initialise the sigma-point set with the given state dimension.

        Parameters
        ----------
        dim : int
            Dimension of the state (and augmented state) space.
        """
        self.dim: int = dim

    @abstractmethod
    def _sigma_point(self, x: np.ndarray, P: np.ndarray) -> np.ndarray:
        """
        Compute the sigma-point set from mean ``x`` and covariance ``P``.

        Parameters
        ----------
        x : np.ndarray
            Mean vector, shape ``(dim, 1)``.
        P : np.ndarray
            Covariance matrix, shape ``(dim, dim)``.

        Returns
        -------
        np.ndarray
            Array of sigma points, shape ``(nbSigmaPoint, dim, 1)``.
        """
        ...

    def _chol(self, P: np.ndarray) -> np.ndarray:
        """
        Compute the lower Cholesky factor of ``P``.

        If the standard decomposition fails due to numerical issues,
        a small regularisation term ``EPS_ABS * I`` is added before
        retrying.

        Parameters
        ----------
        P : np.ndarray
            Symmetric positive semi-definite matrix, shape ``(dim, dim)``.

        Returns
        -------
        np.ndarray
            Lower triangular Cholesky factor, shape ``(dim, dim)``.

        Raises
        ------
        np.linalg.LinAlgError
            If Cholesky decomposition fails even after regularisation.
        """
        try:
            return np.linalg.cholesky(P)
        except np.linalg.LinAlgError:
            try:
                return np.linalg.cholesky(P + EPS_ABS * np.eye(self.dim))
            except np.linalg.LinAlgError:
                raise np.linalg.LinAlgError(
                    "_chol: Cholesky decomposition failed even after regularisation"
                )

    def _normalize_weights(self, Wm: np.ndarray) -> np.ndarray:
        """
        Normalize weight vector ``Wm`` to sum to exactly 1.

        Parameters
        ----------
        Wm : np.ndarray
            Weight vector to normalize, shape ``(nbSigmaPoint,)``.

        Returns
        -------
        np.ndarray
            Normalized weight vector summing to 1.

        Raises
        ------
        ValueError
            If the sum of ``Wm`` deviates from 1 by more than ``EPS_ABS``
            before normalization.
        """
        if not np.isclose(Wm.sum(), 1.0, atol=EPS_ABS):
            raise ValueError(f"Wm weights do not sum to 1 (sum={Wm.sum():.6f})")
        return Wm / Wm.sum()

    @staticmethod
    def _as_column(x: np.ndarray) -> np.ndarray:
        """
        Ensure ``x`` is a column vector of shape ``(n, 1)``.

        Parameters
        ----------
        x : np.ndarray
            Input vector of any compatible shape.

        Returns
        -------
        np.ndarray
            Column vector, shape ``(n, 1)``.
        """
        return np.atleast_2d(x).reshape(-1, 1)


class SetWAN2000(SigmaPointsSet, key="wan2000"):
    """
    Sigma-point set from the Unscented Kalman Filter (UKF).

    The most widely used sigma-point set, based on:

        E. A. Wan and R. V. D. Merwe, "The unscented Kalman filter for
        nonlinear estimation," in Proc. IEEE Adaptive Syst. Signal Process.
        Commun. Control Symp. (ASSPCCS'00), 2000, pp. 153–158.

    Produces ``2 * dim + 1`` sigma points. The mean weight ``Wm`` and
    covariance weight ``Wc`` are identical except for the central point,
    where ``Wc[0]`` includes a corrective term controlled by ``alpha``
    and ``beta``.

    Attributes
    ----------
    nbSigmaPoint : int
        Total number of sigma points: ``2 * dim + 1``.
    Wm : np.ndarray
        Mean weights, shape ``(nbSigmaPoint,)``.
    Wc : np.ndarray
        Covariance weights, shape ``(nbSigmaPoint,)``.
        Differs from ``Wm`` only at index 0.
    gamma : float
        Scaling factor for the sigma point spread: ``sqrt(dim + lambda_)``.
    """

    def __init__(self, dim: int, param) -> None:
        """
        Initialise the WAN2000 sigma-point set.

        Parameters
        ----------
        dim : int
            State dimension.
        param : object
            Parameter object exposing ``lambda_``, ``alpha``, and ``beta``.
        """
        super().__init__(dim)

        self.nbSigmaPoint: int = 2 * self.dim + 1

        self.Wm: np.ndarray = np.full(
            self.nbSigmaPoint, 1.0 / (2.0 * (self.dim + param.lambda_))
        )
        self.Wm[0] = param.lambda_ / (self.dim + param.lambda_)
        self._normalize_weights(self.Wm)

        # Wc equals Wm except at index 0, which includes a corrective term
        self.Wc: np.ndarray = np.copy(self.Wm)
        self.Wc[0] += 1.0 - param.alpha**2 + param.beta

        self.gamma: float = np.sqrt(self.dim + param.lambda_)

    def _sigma_point(self, x: np.ndarray, P: np.ndarray) -> np.ndarray:
        """
        Compute the ``2 * dim + 1`` WAN2000 sigma points.

        Parameters
        ----------
        x : np.ndarray
            Mean vector, shape ``(dim, 1)``.
        P : np.ndarray
            Covariance matrix, shape ``(dim, dim)``.

        Returns
        -------
        np.ndarray
            Sigma points, shape ``(2 * dim + 1, dim, 1)``.
        """
        x = self._as_column(x)
        sqrt_P: np.ndarray = self._chol(P)

        sigma: np.ndarray = np.empty((self.nbSigmaPoint, self.dim, 1))
        sigma[0] = x
        for i in range(self.dim):
            delta = self.gamma * sqrt_P[:, i].reshape(-1, 1)
            sigma[2 * i + 1] = x + delta
            sigma[2 * i + 2] = x - delta
        return sigma


class SetCPKF(SigmaPointsSet, key="cpkf"):
    """
    Sigma-point set from the Cubature Kalman Filter (CKF).

    Based on:

        I. Arasaratnam, S. Haykin, and T. R. Hurd, "Cubature Kalman
        Filtering for Continuous-Discrete Systems: Theory and Simulations,"
        IEEE Trans. Signal Process., vol. 58, no. 10, pp. 4977–4993, 2010.

    Note: this is not strictly a UKF, but can be implemented analogously.
    Produces ``2 * dim`` sigma points with uniform weights.

    Attributes
    ----------
    nbSigmaPoint : int
        Total number of sigma points: ``2 * dim``.
    Wm : np.ndarray
        Mean weights, shape ``(nbSigmaPoint,)``. All equal to ``1 / (2*dim)``.
    Wc : np.ndarray
        Covariance weights, identical to ``Wm``.
    gamma : float
        Scaling factor: ``sqrt(dim)``.
    """

    def __init__(self, dim: int, param) -> None:
        """
        Initialise the CPKF sigma-point set.

        Parameters
        ----------
        dim : int
            State dimension.
        param : object
            Parameter object (unused here, kept for interface consistency).
        """
        super().__init__(dim)

        self.nbSigmaPoint: int = 2 * self.dim
        self.Wm: np.ndarray = np.full(self.nbSigmaPoint, 1.0 / (2.0 * self.dim))
        self._normalize_weights(self.Wm)
        self.Wc: np.ndarray = np.copy(self.Wm)
        self.gamma: float = np.sqrt(self.dim)

    def _sigma_point(self, x: np.ndarray, P: np.ndarray) -> np.ndarray:
        """
        Compute the ``2 * dim`` cubature sigma points.

        Parameters
        ----------
        x : np.ndarray
            Mean vector, shape ``(dim, 1)``.
        P : np.ndarray
            Covariance matrix, shape ``(dim, dim)``.

        Returns
        -------
        np.ndarray
            Sigma points, shape ``(2 * dim, dim, 1)``.
        """
        x = self._as_column(x)
        sqrt_P: np.ndarray = self._chol(P)

        sigma: np.ndarray = np.empty((self.nbSigmaPoint, self.dim, 1))
        for i in range(self.dim):
            delta = self.gamma * sqrt_P[:, i].reshape(-1, 1)
            sigma[2 * i] = x + delta
            sigma[2 * i + 1] = x - delta
        return sigma


class SetLERNER2002(SigmaPointsSet, key="lerner2002"):
    """
    High-order sigma-point set from Lerner (2002).

    Based on:

        U. N. Lerner, "Hybrid Bayesian networks for reasoning about complex
        systems," Ph.D. dissertation, Stanford University, 2002.

    Produces ``2 * dim^2 + 1`` sigma points. The unscented transform is
    exact up to 4th order. Includes both axial points (along each eigenvector
    of the Cholesky factor) and cross points (along pairwise sums/differences).

    Attributes
    ----------
    nbSigmaPoint : int
        Total number of sigma points: ``2 * dim**2 + 1``.
    Wm : np.ndarray
        Mean weights, shape ``(nbSigmaPoint,)``.
    Wc : np.ndarray
        Covariance weights. Differs from ``Wm`` only at index 0.
    gamma : float
        Scaling factor: ``sqrt(3)``.
    """

    def __init__(self, dim: int, param) -> None:
        """
        Initialise the LERNER2002 sigma-point set.

        Weight formulas follow Lerner (2002):

        - ``Wm[0]     = (dim^2 - 7*dim) / 18 + 1``
        - ``Wm[1:2n+1] = (4 - dim) / 18``
        - ``Wm[2n+1:]  = 1 / 36``

        Parameters
        ----------
        dim : int
            State dimension.
        param : object
            Parameter object exposing ``alpha`` and ``beta`` for the
            corrective term on ``Wc[0]``.
        """
        super().__init__(dim)

        self.nbSigmaPoint: int = 2 * self.dim**2 + 1

        self.Wm: np.ndarray = np.zeros(self.nbSigmaPoint)
        # Weight formulas from Lerner (2002)
        self.Wm[0] = (self.dim**2 - 7.0 * self.dim) / 18 + 1.0
        self.Wm[1 : 2 * self.dim + 1] = (4 - self.dim) / 18.0
        self.Wm[2 * self.dim + 1 : 2 * self.dim**2 + 1] = 1.0 / 36.0
        self._normalize_weights(self.Wm)

        # Wc equals Wm except at index 0 — same corrective term as WAN2000
        self.Wc: np.ndarray = np.copy(self.Wm)
        self.Wc[0] += 1.0 - param.alpha**2 + param.beta

        self.gamma: float = np.sqrt(3.0)

    def _sigma_point(self, x: np.ndarray, P: np.ndarray) -> np.ndarray:
        """
        Compute the ``2 * dim^2 + 1`` LERNER2002 sigma points.

        Includes one central point, ``2 * dim`` axial points, and
        ``4 * dim * (dim - 1) / 2`` cross points.

        Parameters
        ----------
        x : np.ndarray
            Mean vector, shape ``(dim, 1)``.
        P : np.ndarray
            Covariance matrix, shape ``(dim, dim)``.

        Returns
        -------
        np.ndarray
            Sigma points, shape ``(2 * dim**2 + 1, dim, 1)``.
        """
        x = self._as_column(x)
        sqrt_P: np.ndarray = self._chol(P)

        # Central point
        sigma = [x.copy()]

        # Axial points — one pair per eigenvector column
        for i in range(self.dim):
            delta = self.gamma * sqrt_P[:, i].reshape(self.dim, 1)
            sigma.append(x + delta)
            sigma.append(x - delta)

        # Cross points — pairwise sums and differences of eigenvector columns
        # Pre-allocated buffers to avoid repeated memory allocation in the loop
        delta_plus: np.ndarray = np.empty((self.dim, 1))
        delta_minus: np.ndarray = np.empty((self.dim, 1))

        for i in range(self.dim):
            col_i = sqrt_P[:, i]  # view, no copy
            for j in range(i + 1, self.dim):
                col_j = sqrt_P[:, j]  # view, no copy

                # In-place operations to avoid intermediate allocations
                np.add(col_i, col_j, out=delta_plus[:, 0])
                delta_plus *= self.gamma

                np.subtract(col_i, col_j, out=delta_minus[:, 0])
                delta_minus *= self.gamma

                sigma.append(x + delta_plus)
                sigma.append(x - delta_plus)
                sigma.append(x + delta_minus)
                sigma.append(x - delta_minus)

        return np.array(sigma)


class SetIto2000(SigmaPointsSet, key="ito2000"):
    """
    Gauss-Hermite quadrature sigma-point set from Ito & Xiong (2000).

    Based on:

        K. Ito and K. Xiong, "Gaussian filters for nonlinear filtering
        problems," IEEE Trans. Autom. Control, vol. 45, no. 5,
        pp. 910–927, May 2000.

    Uses a tensor-product Gauss-Hermite quadrature rule of order ``p``
    (fixed to 3). The number of sigma points grows as ``p^dim``, so this
    set is only practical for low-dimensional problems (``dim <= 4``
    with ``p = 3``).

    Attributes
    ----------
    p : int
        Quadrature order, fixed to 3.
    nbSigmaPoint : int
        Total number of sigma points: ``p^dim``.
    Xi : np.ndarray
        Quadrature nodes, shape ``(nbSigmaPoint, dim)``.
    Wm : np.ndarray
        Mean weights, shape ``(nbSigmaPoint,)``.
    Wc : np.ndarray
        Covariance weights, identical to ``Wm``.
    """

    def __init__(self, dim: int, param) -> None:
        """
        Initialise the ITO2000 sigma-point set.

        Quadrature nodes and weights are computed via
        :func:`numpy.polynomial.hermite.hermgauss` and assembled
        by tensor product over all ``dim`` dimensions.

        Parameters
        ----------
        dim : int
            State dimension. Keep ``dim <= 4`` with ``p = 3`` to avoid
            exponential growth in the number of sigma points.
        param : object
            Parameter object (unused here, kept for interface consistency).
        """
        super().__init__(dim)

        self.p: int = (
            3  # Quadrature order — fixed; increase with caution (p^dim points)
        )
        self.nbSigmaPoint: int = self.p**self.dim

        xi_1d, w_1d = np.polynomial.hermite.hermgauss(self.p)

        # Tensor product of 1D nodes and weights over all dimensions
        self.Xi: np.ndarray = np.array(list(product(xi_1d, repeat=self.dim)))
        self.Wm: np.ndarray = np.prod(
            np.array(list(product(w_1d, repeat=self.dim))), axis=1
        )

        # Normalise by pi^(dim/2) — standard Gauss-Hermite convention
        self.Wm /= np.pi ** (self.dim / 2)
        self._normalize_weights(self.Wm)

        self.Wc: np.ndarray = np.copy(self.Wm)

    def _sigma_point(self, x: np.ndarray, P: np.ndarray) -> np.ndarray:
        """
        Compute the ``p^dim`` Gauss-Hermite sigma points.

        Each point is obtained by mapping a quadrature node through the
        Cholesky factor of ``P``:  ``x + sqrt(P) @ xi``.

        Parameters
        ----------
        x : np.ndarray
            Mean vector, shape ``(dim, 1)``.
        P : np.ndarray
            Covariance matrix, shape ``(dim, dim)``.

        Returns
        -------
        np.ndarray
            Sigma points, shape ``(p^dim, dim, 1)``.
        """
        x = self._as_column(x)
        sqrt_P: np.ndarray = self._chol(P)

        sigma: np.ndarray = np.empty((self.nbSigmaPoint, self.dim, 1))
        for idx, xi in enumerate(self.Xi):
            sigma[idx] = x + (sqrt_P @ xi).reshape(self.dim, 1)
        return sigma
