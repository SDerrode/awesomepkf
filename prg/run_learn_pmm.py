"""
CLI entry point — estimate 1D linear PMM parameters (a, b, c, d, e) from a CSV.

Usage::

    awesomepkf-fit-pkf --data-filename path/to/series.csv --x-col 0 --y-col 1 \
        --output params.npz --verbose 1

The data file may be any format supported by :func:`prg.utils.io.read_unknown_file`
(CSV, TSV, Parquet, JSON, Excel). Columns are selected by positional index
(``--x-col 0``) or by name (``--x-col ActivePower_KWh``).

If ``--output`` is given, the parameters and the original data are saved to a
NumPy ``.npz`` archive with keys ``a, b, c, d, e, data, columns``.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np

from prg.learning.pmm_moments import (
    PMMParams,
    estimate_pmm_params,
    validate_pmm,
)
from prg.utils.exceptions import NumericalError, ParamError, PKFError
from prg.utils.io import read_unknown_file

__all__ = ["main", "run"]


def _parse_col(value: str) -> int | str:
    """Parse a column identifier — int if numeric, else string."""
    try:
        return int(value)
    except ValueError:
        return value


def _parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Estimate 1D linear PMM parameters (a, b, c, d, e) from a CSV by the method of moments."
    )
    parser.add_argument(
        "--data-filename",
        type=str,
        required=True,
        help="Path to the input time series (CSV/TSV/Parquet/JSON/Excel).",
    )
    parser.add_argument(
        "--x-col",
        type=_parse_col,
        default=0,
        help="State column (positional index or column name). Default: 0.",
    )
    parser.add_argument(
        "--y-col",
        type=_parse_col,
        default=1,
        help="Observation column (positional index or column name). Default: 1.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional path to save the estimated parameters and data as .npz.",
    )
    parser.add_argument(
        "--no-standardise",
        dest="standardise",
        action="store_false",
        help="Disable mean-centring/variance-scaling of the input (assume already standardised).",
    )
    parser.add_argument(
        "--verbose",
        type=int,
        choices=[0, 1, 2],
        default=1,
        help="Verbosity level. Default: 1.",
    )
    return parser.parse_args()


def run(args: argparse.Namespace) -> PMMParams:
    """Read the data, estimate the PMM parameters, optionally save them."""
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
        datefmt="%H:%M:%S",
    )
    log = logging.getLogger("learn_pmm")

    data_path = Path(args.data_filename)
    if not data_path.exists():
        raise ParamError(f"Data file not found: {data_path}")

    df = read_unknown_file(str(data_path), verbose=args.verbose)
    log.info("loaded %d rows x %d cols from %s", len(df), df.shape[1], data_path.name)

    params = estimate_pmm_params(
        df,
        x_col=args.x_col,
        y_col=args.y_col,
        standardise=args.standardise,
        verbose=args.verbose,
    )

    if not validate_pmm(params):
        raise NumericalError(
            f"Estimated parameters are not a valid PMM: {params}. "
            "Consider checking column selection, standardisation, or stationarity of the series."
        )

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        x_col = df.columns[args.x_col] if isinstance(args.x_col, int) else args.x_col
        y_col = df.columns[args.y_col] if isinstance(args.y_col, int) else args.y_col
        # Save only the two selected series as a float matrix — keeps the file
        # loadable with ``np.load(..., allow_pickle=False)``. The original
        # timestamp index is left on the source CSV.
        xy = df[[x_col, y_col]].to_numpy(dtype=float)
        np.savez(
            args.output,
            a=params.a,
            b=params.b,
            c=params.c,
            d=params.d,
            e=params.e,
            columns=np.array([str(x_col), str(y_col)]),
            data=xy,
        )
        log.info("saved parameters to %s", args.output)

    return params


def main() -> None:
    try:
        run(_parse_arguments())
    except ParamError as e:
        print(f"[PARAMETER ERROR] {e}", file=sys.stderr)
        sys.exit(2)
    except NumericalError as e:
        print(f"[NUMERICAL ERROR] {e}", file=sys.stderr)
        sys.exit(1)
    except PKFError as e:
        print(f"[PKF ERROR] {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
