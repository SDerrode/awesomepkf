"""Error-metric and diagnostic helpers (MSE/MAE/RMSE/NEES/NIS, equality checks)."""

from __future__ import annotations

import warnings as _warnings
from typing import Any

import numpy as np

from prg.classes.matrix_diagnostics import InvertibleMatrix
from prg.utils.numerics import EPS_ABS, EPS_REL

__all__ = ["check_equality", "compute_errors"]


# ----------------------------------------------------------------------
# Quadratic forms (NEES / NIS)
# ----------------------------------------------------------------------
def _compute_quadratic_form(
    errors: np.ndarray,
    cov_list: list[np.ndarray],
) -> np.ndarray:
    """
    Compute ``e_k^T @ Cov_k^{-1} @ e_k`` for each time step ``k``.

    Uses ``np.linalg.pinv`` (pseudo-inverse via SVD), which handles
    singular and near-singular matrices without raising exceptions.
    Steps with non-finite covariance entries yield ``NaN``.

    Parameters
    ----------
    errors : np.ndarray
        Array of error vectors, shape ``(N, dim)``.
    cov_list : list of np.ndarray
        List of ``N`` covariance matrices, each of shape ``(dim, dim)``.

    Returns
    -------
    np.ndarray
        Array of quadratic form values, shape ``(N,)``.
        Entries are ``NaN`` where computation was not possible.
    """
    N = errors.shape[0]
    vals = np.full(N, np.nan)

    for k in range(N):
        ek = errors[k].reshape(-1, 1)
        Pk = cov_list[k]

        if not np.all(np.isfinite(Pk)):
            continue

        try:
            Pk_inv = InvertibleMatrix(Pk).inverse()
        except RuntimeError:
            # Pk near-singular (particle collapse) — Moore-Penrose pseudo-inverse
            Pk_inv = np.linalg.pinv(Pk)

        vals[k] = float((ek.T @ Pk_inv @ ek).squeeze())

    return vals


def compute_errors(
    model: Any,
    x_true: list[np.ndarray],
    x_hat: list[np.ndarray],
    P_list: list[np.ndarray],
    i_list: np.ndarray | None = None,
    S_list: list[np.ndarray] | None = None,
) -> dict:
    """
    Compute error metrics between two state sequences.

    Computes global MSE, MAE, RMSE, and mean NEES. If innovation sequences
    are provided, the mean NIS is also computed. For augmented models, MSE
    and MAE are additionally reported separately for the X and Y components.

    Parameters
    ----------
    model : PKF subclass instance
        Filter model, used to access ``param.augmented``, ``dim_x``, ``dim_y``.
    x_true : list of np.ndarray
        Ground truth state vectors, each of shape ``(dim_x, 1)`` or ``(dim_x,)``.
    x_hat : list of np.ndarray
        Estimated state vectors, same shape as ``x_true``.
    P_list : list of np.ndarray
        State covariance matrices at each step, each of shape ``(dim_x, dim_x)``.
    i_list : np.ndarray, optional
        Innovation vectors stacked as ``(N, dim_y)``. Required for NIS computation.
    S_list : list of np.ndarray, optional
        Innovation covariance matrices at each step, each of shape ``(dim_y, dim_y)``.
        Required for NIS computation.

    Returns
    -------
    dict
        Dictionary with the following keys:

        - ``mse_total``          : float — global mean squared error
        - ``mae_total``          : float — global mean absolute error
        - ``nees_mean``          : float — mean normalised estimation error squared
        - ``nis_mean``           : float or ``"na"`` — mean normalised innovation squared
        - ``list_mses_X_and_Y``  : list[float] — per-component MSE (augmented models only)
        - ``list_maes_X_and_Y``  : list[float] — per-component MAE (augmented models only)
    """
    # Stack into (N, dim) arrays
    x_true = np.hstack(x_true).T
    x_hat = np.hstack(x_hat).T

    errors = x_true - x_hat

    # FIX: errors_flat = errors.flatten() (the original used np.concatenate on a 2D array
    #        after hstack().T, which flattened row by row in an inconsistent way)
    errors_flat = errors.flatten()
    mse_total = float(np.mean(errors_flat**2))
    mae_total = float(np.mean(np.abs(errors_flat)))
    # rmse = float(np.sqrt(mse_total))

    report = {
        "mse_total": mse_total,
        "mae_total": mae_total,
        "nees_mean": "na",
        "nis_mean": "na",
    }

    # FIX: errors already computed above — duplicate line removed

    # if not model.param.augmented:
    # Mean NEES
    tab_Pk = np.stack(P_list, axis=0)
    nees_all = _compute_quadratic_form(errors, tab_Pk)
    nees_mean = float(np.nanmean(nees_all))
    report["nees_mean"] = nees_mean

    # Mean NIS (optional)
    # FIX: S_list validated explicitly — TypeError crash if None when i_list is provided
    if i_list is not None:
        if S_list is None:
            raise ValueError("S_list must be provided when i_list is not None.")
        tab_Sk = np.stack(S_list, axis=0)
        nis_all = _compute_quadratic_form(i_list, tab_Sk)
        nis_mean = float(np.nanmean(nis_all))
    else:
        nis_mean = "na"
    report["nis_mean"] = nis_mean

    # Per-component MSE and MAE for augmented models
    if model.param.augmented:
        dim_x = model.dim_x
        dim_y = model.dim_y
        report["list_mses_X_and_Y"] = [
            float(np.mean(errors[:, : dim_x - dim_y] ** 2)),
            float(np.mean(errors[:, dim_x - dim_y :] ** 2)),
        ]
        report["list_maes_X_and_Y"] = [
            float(np.mean(np.abs(errors[:, : dim_x - dim_y]))),
            float(np.mean(np.abs(errors[:, dim_x - dim_y :]))),
        ]

    return report


# ----------------------------------------------------------------------
# Matrix equality check
# ----------------------------------------------------------------------
def check_equality(**kwargs: np.ndarray) -> None:
    """
    Check that all provided matrices are numerically equal.

    The first matrix serves as the reference. All others are compared
    against it using ``np.allclose`` with tolerances ``EPS_ABS`` and
    ``EPS_REL``. Differences are reported via the module logger.

    Parameters
    ----------
    **kwargs : np.ndarray
        Named matrices to compare. At least two must be provided.
    """
    if len(kwargs) < 2:
        # FIX: warnings.warn instead of print (signals the anomaly to the caller)
        _warnings.warn(
            "check_equality: at least 2 matrices required.", UserWarning, stacklevel=2
        )
        return

    names = list(kwargs.keys())
    matrices = [np.asarray(m) for m in kwargs.values()]
    shapes = [m.shape for m in matrices]

    if len(set(shapes)) != 1:
        _warnings.warn(
            f"Matrices have different shapes: {dict(zip(names, shapes, strict=False))}",
            UserWarning,
            stacklevel=2,
        )
        return

    ref, ref_name = matrices[0], names[0]

    for name, M in zip(names[1:], matrices[1:], strict=False):
        if not np.allclose(ref, M, atol=EPS_ABS, rtol=EPS_REL):
            diff_norm = float(np.linalg.norm(ref - M))
            _warnings.warn(
                f"Matrices '{ref_name}' and '{name}' differ (‖Δ‖={diff_norm:.3e})",
                UserWarning,
                stacklevel=2,
            )
