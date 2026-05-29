"""
Method-of-moments estimator for the 1D linear Pairwise Markov Model (PMM).

Estimates the five PMM parameters (a, b, c, d, e) from a two-column time series
(X_n, Y_n) — typically a standardised real signal — by computing the empirical
4x4 covariance of the lagged vector (X_n, Y_n, X_{n+1}, Y_{n+1}).

Assumes scalar X and Y, zero-mean and unit-variance (the caller is responsible
for centring/standardising — :func:`estimate_pmm_params` does it by default).
For multi-dimensional states or non-linear models (EPKF/UPKF) a different
estimation procedure is required.

The PMM covariance is::

    Gamma = [[1, b, a, d],
             [b, 1, e, c],
             [a, e, 1, b],
             [d, c, b, 1]]

The classical HMM (Kalman) case is recovered when c = a*b^2 and d = e = a*b.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np
import pandas as pd
from scipy import linalg as LA

from prg.utils.exceptions import NumericalError, ParamError

__all__ = [
    "PMMParams",
    "estimate_pmm_params",
    "pmm_to_linear_params",
    "validate_pmm",
]


# ----------------------------------------------------------------------
# Container
# ----------------------------------------------------------------------


class PMMParams(NamedTuple):
    """Five scalar PMM parameters. Iterable, hashable, immutable."""

    a: float
    b: float
    c: float
    d: float
    e: float

    def as_hmm(self) -> PMMParams:
        """Project onto the HMM submanifold (c = a*b^2, d = e = a*b)."""
        return self._replace(c=self.a * self.b**2, d=self.a * self.b, e=self.a * self.b)

    def is_hmm(self, tol: float = 1e-9) -> bool:
        return (
            abs(self.c - self.a * self.b**2) < tol
            and abs(self.d - self.a * self.b) < tol
            and abs(self.e - self.a * self.b) < tol
        )

    def __repr__(self) -> str:
        return (
            f"PMMParams(a={self.a:.4f}, b={self.b:.4f}, "
            f"c={self.c:.4f}, d={self.d:.4f}, e={self.e:.4f})"
        )


# ----------------------------------------------------------------------
# Internal helpers
# ----------------------------------------------------------------------


def _build_gamma(params: PMMParams) -> np.ndarray:
    a, b, c, d, e = params
    return np.array(
        [
            [1.0, b, a, d],
            [b, 1.0, e, c],
            [a, e, 1.0, b],
            [d, c, b, 1.0],
        ]
    )


def _is_pd(matrix: np.ndarray, eps: float, *, assume_symmetric: bool = False) -> bool:
    """Strict positive-definiteness: eigvals > eps. Symmetry is checked unless asserted."""
    if not assume_symmetric and not np.allclose(matrix, matrix.T, atol=eps):
        return False
    return bool(np.all(LA.eigvalsh(matrix) > eps))


def _decompose(params: PMMParams, eps: float = 1e-9) -> tuple[np.ndarray, ...] | None:
    """
    Build Gamma and derive (Q1, Q2, A, BBt). Return ``None`` if the parameters
    do not yield a strictly positive-definite PMM.

    Shared by :func:`validate_pmm` and :func:`pmm_to_linear_params` so we never
    duplicate the eigenvalue/inversion work.
    """
    if abs(params.b) >= 1.0 - eps or abs(params.c) >= 1.0 - eps:
        return None
    gamma = _build_gamma(params)
    if not _is_pd(gamma, eps, assume_symmetric=True):
        return None
    try:
        Q1 = gamma[0:2, 0:2]
        Q2 = gamma[2:4, 0:2]
        A_mat = Q2 @ LA.inv(Q1)
        BBt = Q1 - A_mat @ Q2.T
    except (LA.LinAlgError, ValueError):
        return None
    if not _is_pd(BBt, eps):
        return None
    return gamma, Q1, Q2, A_mat, BBt


# ----------------------------------------------------------------------
# Validation
# ----------------------------------------------------------------------


def validate_pmm(params: PMMParams, eps: float = 1e-9) -> bool:
    """
    Check that (a, b, c, d, e) define a valid 1D PMM covariance.

    Structural constraints:

    - ``|b| < 1`` — otherwise ``Q1`` is singular and the transition matrix is undefined.
    - ``|c| < 1`` — otherwise ``alpha_3 = beta_3 = 0`` which breaks the MSE computation.
    - ``Gamma`` is symmetric positive definite.
    - ``BBt = Q1 - A @ Q2.T`` is positive definite.
    """
    return _decompose(params, eps) is not None


# ----------------------------------------------------------------------
# Estimation
# ----------------------------------------------------------------------


def estimate_pmm_params(
    data: pd.DataFrame | np.ndarray,
    x_col: int | str = 0,
    y_col: int | str = 1,
    standardise: bool = True,
    verbose: int = 0,
) -> PMMParams:
    """
    Estimate (a, b, c, d, e) from a two-column time series by the method of moments.

    Parameters
    ----------
    data : pandas.DataFrame or numpy.ndarray
        Time series with at least two columns. If a DataFrame, ``x_col`` and
        ``y_col`` may be column names or positional indices. If an ndarray of
        shape ``(T, K)``, they must be positional indices.
    x_col, y_col : int or str, optional
        Identifiers for the state and observation columns.
    standardise : bool, optional
        If ``True`` (default), each column is mean-centred and rescaled to unit
        variance before computing the lagged covariance.
    verbose : int, optional
        ``0`` silent, ``1`` print the estimated parameters, ``2`` also print
        intermediate matrices.

    Returns
    -------
    PMMParams
        Estimated parameters. Off-diagonal correlations are clipped to
        ``[-0.9999, 0.9999]`` to absorb sampling noise; ``b`` is symmetrised
        as the average of ``cov(X_n, Y_n)`` and ``cov(X_{n+1}, Y_{n+1})``.

    Raises
    ------
    ParamError
        If the data has fewer than two columns or fewer than three rows.
    NumericalError
        If the empirical covariance cannot be computed.
    """
    arr = _extract_xy(data, x_col, y_col)
    if arr.shape[0] < 3:
        raise ParamError(f"At least 3 time steps are required, got {arr.shape[0]}.")

    if standardise:
        arr = (arr - arr.mean(axis=0)) / arr.std(axis=0, ddof=1)

    T = arr.shape[0]
    lagged = np.empty((4, T - 1))
    lagged[0, :] = arr[:-1, 0]
    lagged[1, :] = arr[:-1, 1]
    lagged[2, :] = arr[1:, 0]
    lagged[3, :] = arr[1:, 1]

    try:
        gamma = np.cov(lagged)
    except (ValueError, np.linalg.LinAlgError) as exc:
        raise NumericalError(f"Empirical covariance failed: {exc}") from exc

    if verbose >= 2:
        print(f"empirical Gamma =\n{gamma}")

    gamma = np.clip(gamma, -0.9999, 0.9999)
    np.fill_diagonal(gamma, 1.0)
    params = PMMParams(
        a=gamma[0, 2],
        b=0.5 * (gamma[0, 1] + gamma[2, 3]),
        c=gamma[1, 3],
        d=gamma[0, 3],
        e=gamma[1, 2],
    )
    if verbose >= 1:
        print(f"estimated {params}")
        print(f"HMM projection: {params.as_hmm()}")
    return params


def _extract_xy(
    data: pd.DataFrame | np.ndarray, x_col: int | str, y_col: int | str
) -> np.ndarray:
    if isinstance(data, pd.DataFrame):
        x = data[x_col] if isinstance(x_col, str) else data.iloc[:, x_col]
        y = data[y_col] if isinstance(y_col, str) else data.iloc[:, y_col]
        return np.column_stack([x.to_numpy(dtype=float), y.to_numpy(dtype=float)])
    if isinstance(data, np.ndarray):
        if data.ndim != 2 or data.shape[1] < 2:
            raise ParamError(
                f"Expected a 2D array with at least 2 columns, got shape {data.shape}."
            )
        if not isinstance(x_col, int) or not isinstance(y_col, int):
            raise ParamError("Column names require a pandas DataFrame.")
        return np.column_stack([data[:, x_col].astype(float), data[:, y_col].astype(float)])
    raise ParamError(f"Unsupported data type: {type(data).__name__}")


# ----------------------------------------------------------------------
# Conversion to LinearAmQ kwargs
# ----------------------------------------------------------------------


def pmm_to_linear_params(params: PMMParams) -> dict[str, np.ndarray]:
    """
    Convert (a, b, c, d, e) into the kwargs expected by ``LinearAmQ`` for a
    pairwise scalar model (``dim_x = dim_y = 1``).

    The pairwise state is ``z = (X, Y)`` of dimension 2. The transition is
    ``A = Q2 @ Q1^{-1}`` with ``Q1 = [[1, b], [b, 1]]`` and
    ``Q2 = [[a, e], [d, c]]``; the process-noise covariance is
    ``mQ = Q1 - A @ Q2.T``; ``B`` is the principal (symmetric) square root of
    ``mQ`` so that ``B @ B.T = mQ``; the initial mean is zero and ``Pz0 = Q1``.

    Returns
    -------
    dict
        Keys ``A``, ``B``, ``mQ``, ``mz0``, ``Pz0`` suitable for
        ``LinearAmQ(dim_x=1, dim_y=1, **kwargs, pairwiseModel=True)``.

    Raises
    ------
    NumericalError
        If the parameters do not yield a valid PMM.
    """
    decomposed = _decompose(params)
    if decomposed is None:
        raise NumericalError(f"Invalid PMM parameters: {params}")
    _, Q1, _, A_mat, BBt = decomposed

    # Symmetric PSD square root via eigendecomposition: B = V @ diag(√λ) @ Vᵀ.
    # Valid because _decompose guarantees BBt is strictly PD (λ > eps).
    w, V = LA.eigh(BBt)
    B_mat = (V * np.sqrt(w)) @ V.T

    return {
        "A": A_mat,
        "B": B_mat,
        "mQ": BBt,
        "mz0": np.zeros((2, 1)),
        "Pz0": Q1.copy(),
    }
