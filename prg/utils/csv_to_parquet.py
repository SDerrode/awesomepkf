"""
csv_to_parquet.py
-----------------
Robustly converts a CSV file to Parquet format.

Usage
-----
    python3 csv_to_parquet.py <file.csv> <file.parquet> [--engine pyarrow|fastparquet]
"""

import argparse
import sys
import warnings
from pathlib import Path

import chardet
import pandas as pd

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Number of bytes read for encoding detection.
# 50 000 bytes covers the majority of cases; increase if files
# with rare encodings are frequent in your corpus.
_ENCODING_SAMPLE_BYTES = 50_000

_SUPPORTED_ENGINES = ("pyarrow", "fastparquet")


# ---------------------------------------------------------------------------
# Functions
# ---------------------------------------------------------------------------


def detect_encoding(file_path: Path, n_bytes: int = _ENCODING_SAMPLE_BYTES) -> str:
    """
    Detects the encoding of a text file via ``chardet``.

    Parameters
    ----------
    file_path : Path
        Path to the file to analyse.
    n_bytes : int
        Number of bytes to read for analysis (default: 50 000).

    Returns
    -------
    str
        Detected encoding, or ``"utf-8"`` if detection fails.

    Warns
    -----
    UserWarning
        If detection fails, a warning is emitted before the fallback.
        (FIX: the original silently swallowed the exception without any log)
    """
    try:
        with Path(file_path).open("rb") as f:
            raw_data = f.read(n_bytes)
    except OSError as e:
        # FIX: raise file access error explicitly
        raise OSError(
            f"Cannot read '{file_path}' for encoding detection: {e}"
        ) from e

    result = chardet.detect(raw_data)
    encoding = result.get("encoding")

    if not encoding:
        # FIX: visible fallback (warning) instead of silent failure
        warnings.warn(
            f"Encoding not detected for '{file_path}' — fallback to utf-8.",
            UserWarning,
            stacklevel=2,
        )
        return "utf-8"

    return encoding


def csv_to_parquet(
    csv_path: Path | str,
    parquet_path: Path | str,
    engine: str = "pyarrow",
) -> None:
    """
    Converts a CSV file to Parquet.

    Parameters
    ----------
    csv_path : Path | str
        Path to the source CSV file.
    parquet_path : Path | str
        Path to the output Parquet file.
    engine : {"pyarrow", "fastparquet"}
        Parquet engine to use (default: ``"pyarrow"``).

    Raises
    ------
    ValueError
        If ``engine`` is not supported.
    FileNotFoundError
        If ``csv_path`` does not exist.
    OSError
        For any other file access problem.
    """
    csv_path = Path(csv_path)
    parquet_path = Path(parquet_path)

    # FIX: engine validation with a clear message (pandas raises a cryptic error otherwise)
    if engine not in _SUPPORTED_ENGINES:
        raise ValueError(
            f"Engine {engine!r} not supported. " f"Choose from: {_SUPPORTED_ENGINES}"
        )

    # FIX: explicit check for source file existence
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: '{csv_path}'")

    encoding = detect_encoding(csv_path)
    print(f"  Detected encoding: {encoding}")

    df = pd.read_csv(csv_path, encoding=encoding)
    print(f"  Rows / columns: {df.shape[0]:,} × {df.shape[1]}")

    # Create output directory if needed
    parquet_path.parent.mkdir(parents=True, exist_ok=True)

    df.to_parquet(parquet_path, engine=engine, index=False)
    print(f"  Parquet written: '{parquet_path}'")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Convert a CSV file to Parquet.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("csv_file", help="Source CSV file")
    p.add_argument("parquet_file", help="Output Parquet file")
    # FIX: --engine exposed as CLI argument (the original did not allow choosing it)
    p.add_argument(
        "--engine",
        choices=_SUPPORTED_ENGINES,
        default="pyarrow",
        help="Parquet engine to use",
    )
    return p


if __name__ == "__main__":
    args = _build_parser().parse_args()

    # FIX: explicit user feedback on success or failure
    try:
        print(f"Conversion: '{args.csv_file}' -> '{args.parquet_file}'")
        csv_to_parquet(args.csv_file, args.parquet_file, engine=args.engine)
        print("Conversion completed successfully.")
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
