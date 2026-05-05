"""File and DataFrame I/O helpers (CSV/TSV/Parquet/JSON/Excel readers, writers, generators)."""

from __future__ import annotations

import csv
from collections.abc import Generator
from pathlib import Path

import chardet
import numpy as np
import pandas as pd

__all__ = [
    "data_to_dataframe",
    "file_data_generator",
    "name_analysis",
    "read_unknown_file",
    "save_dataframe_to_csv",
]


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
        if __debug__ and (not hasattr(x, "flatten") or not hasattr(y, "flatten")):
            raise TypeError(f"Elements at index {idx} are not numpy arrays.")
        x_values = x.flatten()
        y_values = y.flatten()
        if __debug__ and (len(x_values) != dim_x or len(y_values) != dim_y):
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
    with Path(filepath).open("rb") as f:
        raw_data = f.read(50_000)
        enc_info = chardet.detect(raw_data)
        encoding = enc_info["encoding"] or "utf-8"
        _confidence = enc_info.get("confidence", 0)
    if ext == ".parquet":
        return pd.read_parquet(filepath)
    if ext == ".json":
        return pd.read_json(filepath, encoding=encoding)
    if ext in (".xlsx", ".xls"):
        return pd.read_excel(filepath)
    if ext in (".csv", ".txt", ".dat", ".tsv", ""):

        with Path(filepath).open(encoding=encoding) as f:
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
