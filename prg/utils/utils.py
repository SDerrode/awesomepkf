#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
utils.py — Utility functions for Kalman filters (PKF/UKF/EKF).

Provides tools for:
- Rich-based display of dataclasses and dictionaries
- DataFrame I/O (CSV read/write)
- Error metrics (MSE, MAE, RMSE, NEES, NIS)
- Robust file reading (CSV, TSV, Parquet, JSON, Excel)
- Covariance matrix diagnostics and validation
- Random covariance generation
"""

from __future__ import annotations
from typing import (
    Any,
    Generator,
)  # FIX: Any added (was used but not imported); List/Optional removed (legacy)

import os  # Used in read_unknown_file
import math  # Used in format_value
import csv  # Used in read_unknown_file
import chardet  # Used in read_unknown_file
from pathlib import Path  # Used in save_dataframe_to_csv
import numpy as np
import pandas as pd
from dataclasses import is_dataclass, asdict  # Used in rich_show_fields
from rich.table import Table
from rich.console import Console
from rich.text import Text

from prg.utils.numerics import EPS_ABS, EPS_REL
from prg.classes.MatrixDiagnostics import CovarianceMatrix, InvertibleMatrix

__all__ = [
    "rich_show_fields",
    "save_dataframe_to_csv",
    "data_to_dataframe",
    "compute_errors",
    "read_unknown_file",
    "file_data_generator",
    "check_equality",
    "name_analysis",
]

# FIX: force_terminal=True removed — avoids spurious ANSI sequences in file/pipe logs
import sys as _sys

console = Console(color_system="truecolor" if _sys.stdout.isatty() else None)


# ----------------------------------------------------------------------
# Rich display
# ----------------------------------------------------------------------
def rich_show_fields(
    d: dict | Any,
    fields: list[str] | None = None,
    title: str = "Data selection",
    decimals: int = 4,
    max_items: int = 10,
) -> None:
    """
    Display a dictionary or dataclass in a readable Rich table.

    Floats are rounded to ``decimals`` digits. NumPy booleans are cast to
    Python bools. Arrays longer than ``max_items`` are truncated. Nested
    dicts and lists are supported.

    Parameters
    ----------
    d : dict or dataclass
        Data to display. Dataclasses are converted via ``asdict``.
    fields : list of str, optional
        Subset of keys to display. If ``None``, all keys are shown.
    title : str, optional
        Title of the Rich table (default ``"Data selection"``).
    decimals : int, optional
        Number of decimal places for float formatting (default ``4``).
    max_items : int, optional
        Maximum number of items shown for arrays and lists (default ``10``).
    """
    if is_dataclass(d):
        d = asdict(d)

    if fields is None:
        fields = list(d.keys())

    table = Table(title=title)
    table.add_column("Field", no_wrap=True)
    table.add_column("Value", justify="left")

    def format_value(obj) -> str:
        """Recursive formatter for scientific display."""
        if isinstance(obj, np.generic):
            obj = obj.item()
        if isinstance(obj, (np.bool_, bool)):
            return str(bool(obj))
        if isinstance(obj, float):
            if math.isinf(obj) or math.isnan(obj):
                return str(obj)
            return f"{obj:.{decimals}f}"
        if isinstance(obj, np.ndarray):
            return format_value(obj.tolist())
        if isinstance(obj, (list, tuple)):
            if len(obj) > max_items:
                displayed = [format_value(v) for v in obj[:max_items]]
                return "[" + ", ".join(displayed) + ", ...]"
            return "[" + ", ".join(format_value(v) for v in obj) + "]"
        if isinstance(obj, dict):
            items = [f"{k}: {format_value(v)}" for k, v in obj.items()]
            return "{ " + ", ".join(items) + " }"
        return str(obj)

    for key in fields:
        if key in d:
            table.add_row(
                Text(key, style="cyan"),
                Text(format_value(d[key]), style="magenta"),
            )

    console.print(table)


# ----------------------------------------------------------------------
# DataFrame I/O
# ----------------------------------------------------------------------
def save_dataframe_to_csv(
    df: pd.DataFrame,
    filepath: str | Path,
    index: bool = False,
) -> None:
    """
    Save a DataFrame to a UTF-8 CSV file.

    Parent directories are created automatically if they do not exist.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to save.
    filepath : str or Path
        Destination file path.
    index : bool, optional
        Whether to write the row index (default ``False``).

    Raises
    ------
    Exception
        Any I/O error encountered during writing is logged and re-raised.
    """
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)
    # FIX: bare try/except with raise removed (added nothing)
    df.to_csv(path, encoding="utf-8", index=index, float_format="%.15f")


def data_to_dataframe(
    listData: list[tuple],
    dim_x: int,
    dim_y: int,
    withoutX: bool = False,
) -> pd.DataFrame:
    """
    Convert a list of PKF/UKF output tuples to a pandas DataFrame.

    Each tuple is expected to be ``(idx, x_array, y_array)``. Columns are
    named ``X0, X1, ..., Y0, Y1, ...`` depending on ``withoutX``.

    Parameters
    ----------
    listData : list of tuple
        List of ``(idx, x_array, y_array)`` tuples.
    dim_x : int
        Expected dimension of the state vector ``x``.
    dim_y : int
        Expected dimension of the observation vector ``y``.
    withoutX : bool, optional
        If ``True``, only Y columns are included (default ``False``).

    Returns
    -------
    pd.DataFrame
        DataFrame with columns ``X0..X{dim_x-1}`` and/or ``Y0..Y{dim_y-1}``.

    Raises
    ------
    TypeError
        If ``x`` or ``y`` elements do not have a ``flatten`` method.
    ValueError
        If the flattened sizes do not match ``dim_x`` or ``dim_y``.
    """
    data = []
    for idx, x, y in listData:
        if __debug__:
            if not hasattr(x, "flatten") or not hasattr(y, "flatten"):
                raise TypeError(f"Elements at index {idx} are not numpy arrays.")
        x_values = x.flatten()
        y_values = y.flatten()
        if __debug__:
            if len(x_values) != dim_x or len(y_values) != dim_y:
                raise ValueError(
                    f"Unexpected sizes at index {idx}: "
                    f"X={len(x_values)} (expected {dim_x}), "
                    f"Y={len(y_values)} (expected {dim_y})"
                )
        if withoutX:
            data.append([*y_values])
        else:
            data.append([*x_values, *y_values])

    columns = []
    if not withoutX:
        columns += [f"X{c}" for c in range(dim_x)]
    columns += [f"Y{c}" for c in range(dim_y)]

    return pd.DataFrame(data, columns=columns)


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
    model: "PKF",
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
    # rmse = float(np.sqrt(mse_total))  # noqa: F841 — available for callers

    report = {
        "mse_total": mse_total,
        "mae_total": mae_total,
        "nees_mean": "na",
        "nis_mean": "na",
    }

    # FIX: errors already computed above — duplicate line removed

    if not model.param.augmented:
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
# Robust file reader
# ----------------------------------------------------------------------
def read_unknown_file(
    filepath: str,
    nrows_detect: int = 500,
    verbose: int = 0,
) -> pd.DataFrame:
    """
    Read a data file (CSV, TSV, Parquet, JSON, Excel) robustly.

    Automatically detects encoding, delimiter, and header presence for
    text-based formats. Parquet, JSON, and Excel files are read directly.

    Parameters
    ----------
    filepath : str
        Path to the file to read.
    nrows_detect : int, optional
        Number of rows used for delimiter/header sniffing (default ``500``).
    verbose : int, optional
        Verbosity level: ``0`` = silent, ``2`` = detailed (default ``0``).

    Returns
    -------
    pd.DataFrame
        Loaded data.

    Raises
    ------
    ValueError
        If the file extension is not recognised.
    Exception
        Any I/O or parsing error is logged and re-raised.
    """
    # FIX: Path.suffix used instead of os.path.splitext (pathlib already imported)
    ext = Path(filepath).suffix.lower()
    # FIX: bare try/except with raise removed (added nothing)
    with open(filepath, "rb") as f:
        raw_data = f.read(50_000)
        enc_info = chardet.detect(raw_data)
        encoding = enc_info["encoding"] or "utf-8"
        confidence = enc_info.get("confidence", 0)
    if ext == ".parquet":
        return pd.read_parquet(filepath)
    if ext == ".json":
        return pd.read_json(filepath, encoding=encoding)
    if ext in (".xlsx", ".xls"):
        return pd.read_excel(filepath)
    if ext in (".csv", ".txt", ".dat", ".tsv", ""):

        with open(filepath, "r", encoding=encoding) as f:
            sample_lines = [next(f, "") for _ in range(min(nrows_detect, 10))]
        sample = "".join(sample_lines)

        try:
            dialect = csv.Sniffer().sniff(sample, delimiters=",;\t| ")
            sep = dialect.delimiter
            has_header = csv.Sniffer().has_header(sample)
        except csv.Error:
            sep = None
            has_header = True

        header = 0 if has_header else None
        if sep is None:
            return pd.read_csv(filepath, header=header, encoding=encoding)
        return pd.read_csv(filepath, sep=sep, header=header, encoding=encoding)

    raise ValueError(f"Unrecognised file format: {ext}")


def name_analysis(listStr: list[str]) -> dict:
    """
    Analyse a list of column names and return dimension metadata.

    Columns are classified by their prefix: ``"True"`` for ground truth,
    ``"X"`` for state components, ``"Y"`` for observation components.

    Parameters
    ----------
    listStr : list of str
        Column names to analyse.

    Returns
    -------
    dict
        Dictionary with keys:

        - ``dim_x_true`` : int  — number of columns starting with ``"True"``
        - ``dim_x``      : int  — number of columns starting with ``"X"``
        - ``dim_y``      : int  — number of columns starting with ``"Y"``
        - ``correct``    : bool — ``True`` if all X columns precede all Y columns
        - ``others``     : list — columns not starting with ``"X"``, ``"Y"``, or ``"True"``

    Raises
    ------
    TypeError
        If ``listStr`` is not a list or tuple.
    """
    if not isinstance(listStr, (list, tuple)):
        raise TypeError("Input must be a list or tuple of strings.")

    dim_x_true = sum(s.startswith("True") for s in listStr)
    dim_x = sum(s.startswith("X") for s in listStr)
    dim_y = sum(s.startswith("Y") for s in listStr)
    others = [
        s
        for s in listStr
        if not (s.startswith("X") or s.startswith("Y") or s.startswith("True"))
    ]

    ok = True
    x_ended = False
    for s in listStr:
        if s.startswith("X"):
            if x_ended:
                ok = False
                break
        elif s.startswith("Y"):
            x_ended = True

    return {
        "dim_x_true": dim_x_true,
        "dim_x": dim_x,
        "dim_y": dim_y,
        "correct": ok,
        "others": others,
    }


# ----------------------------------------------------------------------
# File data generator
# ----------------------------------------------------------------------
def file_data_generator(
    filename: str,
    dim_x: int,
    dim_y: int,
    verbose: int = 0,
) -> Generator[tuple[int, np.ndarray | None, np.ndarray], None, None]:
    """
    Read a data file and yield ``(k, x, y)`` tuples one step at a time.

    If the file contains no X columns, ``x`` is ``None`` at every step.
    Row index ``k`` is a contiguous integer starting at ``0``, regardless
    of the original DataFrame index.

    Parameters
    ----------
    filename : str
        Path to the data file.
    dim_x : int
        Expected dimension of the state vector.
    dim_y : int
        Expected dimension of the observation vector.
    verbose : int, optional
        Verbosity level passed to :func:`read_unknown_file` (default ``0``).

    Yields
    ------
    k : int
        Contiguous time step index starting at ``0``.
    x : np.ndarray or None
        State vector at step ``k``, shape ``(dim_x, 1)``.
        ``None`` if the file contains no X columns.
    y : np.ndarray
        Observation vector at step ``k``, shape ``(dim_y, 1)``.

    Raises
    ------
    ValueError
        If column order, ``dim_x``, or ``dim_y`` do not match expectations.
    """
    df = read_unknown_file(filename, verbose=verbose)
    dico = name_analysis(list(df.columns))
    has_x_columns = dico["dim_x"] != 0

    if has_x_columns:
        if not dico["correct"]:
            raise ValueError(
                f"X and Y columns are not in the expected order.\n"
                f"Columns found: {list(df.columns)}"
            )
        if dico["dim_x"] != dim_x:
            raise ValueError(
                f"Incorrect X dimension: expected {dim_x}, found {dico['dim_x']}.\n"
                f"Columns: {list(df.columns)}"
            )
        if dico["dim_y"] != dim_y:
            raise ValueError(
                f"Incorrect Y dimension: expected {dim_y}, found {dico['dim_y']}.\n"
                f"Columns: {list(df.columns)}"
            )

    for k, (_, row) in enumerate(df.iterrows()):
        values = row.values.reshape(-1, 1)
        if has_x_columns:
            xkp1, ykp1 = np.split(values, [dico["dim_x"]])
            yield k, xkp1, ykp1
        else:
            yield k, None, values


# ----------------------------------------------------------------------
# Matrix equality check
# ----------------------------------------------------------------------
import warnings as _warnings


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
            f"Matrices have different shapes: {dict(zip(names, shapes))}",
            UserWarning,
            stacklevel=2,
        )
        return

    ref, ref_name = matrices[0], names[0]

    for name, M in zip(names[1:], matrices[1:]):
        if not np.allclose(ref, M, atol=EPS_ABS, rtol=EPS_REL):
            diff_norm = float(np.linalg.norm(ref - M))
            _warnings.warn(
                f"Matrices '{ref_name}' and '{name}' differ (‖Δ‖={diff_norm:.3e})",
                UserWarning,
                stacklevel=2,
            )
