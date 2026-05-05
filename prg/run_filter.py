"""
Generic CLI dispatcher for the 6 filter families.

Replaces the per-filter CLI scripts that all duplicated the same
parse_arguments / try-except shell. Each ``run_<filter>.py`` now is a
4-line wrapper that calls :func:`run` with its filter name.

Direct usage::

    python -m prg.run_filter epkf --nonlinear-model-name model_x1_y1_pairwise --N 100

But normally, you would invoke through the per-filter wrappers
(``python -m prg.run_nonlinear_epkf``) or the entry-point scripts
(``awesomepkf-epkf``).
"""

import argparse
import logging
import sys

from prg.base_classes.filter_runner import FilterRunner, RunOptions
from prg.base_classes.filter_specs import FILTER_SPECS
from prg.utils.exceptions import FilterError, NumericalError, ParamError, PKFError
from prg.utils.parser import add_arguments

__all__ = ["main", "run"]


# ----------------------------------------------------------------------
# Logging
# ----------------------------------------------------------------------

_VERBOSE_TO_LEVEL = {0: logging.WARNING, 1: logging.INFO, 2: logging.DEBUG}


def _setup_logging(verbose: int) -> None:
    logging.basicConfig(
        level=_VERBOSE_TO_LEVEL.get(verbose, logging.DEBUG),
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
        datefmt="%H:%M:%S",
    )


# ----------------------------------------------------------------------
# Argument parsing
# ----------------------------------------------------------------------


def _parse_arguments(filter_name: str) -> tuple[argparse.Namespace, str]:
    try:
        spec = FILTER_SPECS[filter_name]
    except KeyError as e:
        raise ParamError(
            f"Unknown filter {filter_name!r}. "
            f"Available: {sorted(FILTER_SPECS)}."
        ) from e
    kind = "Linear" if spec.is_linear else "NonLinear"
    parser = argparse.ArgumentParser(description=f"Run {kind} {spec.acronym}")

    arg_keys: list[str] = []
    if spec.is_linear:
        arg_keys.append("linear-model-name")
    else:
        arg_keys.extend(["nonlinear-model-name", "linear-model-name"])
    arg_keys.extend(["N", "s-key"])
    if "sigmaSet" in spec.requires:
        arg_keys.append("sigma-set")
    if "n_particles" in spec.requires:
        arg_keys.append("n-particles")
    arg_keys.append("data-filename")
    add_arguments(parser, arg_keys)

    args = parser.parse_args()

    if args.data_filename is not None and args.N is not None:
        parser.error("--N should not be used with --data-filename")
    if args.data_filename is None and args.N is None:
        parser.error("--N must be used when --data-filename is not specified")

    if spec.is_linear:
        model_name = args.linear_model_name
    else:
        if args.linear_model_name is not None and args.nonlinear_model_name is not None:
            parser.error(
                "--nonlinear-model-name should not be used with --linear-model-name. One or the other!"
            )
        if args.linear_model_name is None and args.nonlinear_model_name is None:
            parser.error("--nonlinear-model-name OR --linear-model-name must be used.")
        model_name = (
            args.nonlinear_model_name
            if args.linear_model_name is None
            else args.linear_model_name
        )

    return args, model_name


# ----------------------------------------------------------------------
# Error-handling wrapper
# ----------------------------------------------------------------------


# Order matters: most specific first. ParamError, NumericalError and
# FilterError all inherit from PKFError, so PKFError must come last among
# the PKF-family entries — otherwise the others are unreachable.
_ERROR_TABLE: tuple[tuple[type[Exception], str, int], ...] = (
    (ParamError,     "PARAMETER ERROR", 2),
    (NumericalError, "NUMERICAL ERROR", 1),
    (FilterError,    "FILTER ERROR",    1),
    (PKFError,       "PKF ERROR",       1),
    (ValueError,     "PARAMETER ERROR", 2),
    (RuntimeError,   "RUNTIME ERROR",   3),
)


def _handle_errors(func, *args, **kwargs):
    """Run ``func(*args, **kwargs)``; map known exceptions to stderr + exit code."""
    try:
        return func(*args, **kwargs)
    except Exception as e:
        for exc_type, label, code in _ERROR_TABLE:
            if isinstance(e, exc_type):
                print(f"[{label}] {e}", file=sys.stderr)
                sys.exit(code)
        raise


# ----------------------------------------------------------------------
# Public dispatcher
# ----------------------------------------------------------------------


def run(filter_name: str) -> None:
    """Parse CLI args and run the named filter (no exception handling)."""
    args, model_name = _parse_arguments(filter_name)
    _setup_logging(args.verbose)
    mode = "from_file" if args.data_filename is not None else "simulation"

    options = RunOptions(
        verbose=args.verbose,
        plot=args.plot,
        save_history=args.save_history,
    )
    runner = FilterRunner(
        filter_name,
        model_name,
        mode,
        N=args.N,
        sKey=args.s_key,
        data_filename=args.data_filename,
        sigmaSet=getattr(args, "sigma_set", None),
        n_particles=getattr(args, "n_particles", None),
        options=options,
    )
    runner.run()


def main(filter_name: str) -> None:
    """Entry-point wrapper used by ``run_<filter>.py`` modules."""
    _handle_errors(run, filter_name)
