#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import sys

from prg.base_classes.nonlinear_ppf_runner_simulation import NonLinearPPFRunnerSim
from prg.base_classes.nonlinear_ppf_runner_from_file import (
    NonLinearPPFRunnerFromFile,
)
from prg.utils.parser import add_arguments
from prg.utils.exceptions import NumericalError, FilterError, PKFError, ParamError

import logging


def setup_logging(verbose: int):
    level = {0: logging.WARNING, 1: logging.INFO, 2: logging.DEBUG}.get(
        verbose, logging.DEBUG
    )
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
        datefmt="%H:%M:%S",
    )


def parse_arguments():
    parser = argparse.ArgumentParser(description="Run NonLinear PPF")

    add_arguments(
        parser,
        [
            "nonLinearModelName",
            "linearModelName",
            "N",
            "sKey",
            "n_particles",
            "dataFileName",
        ],
    )

    args = parser.parse_args()

    if args.dataFileName is not None and args.N is not None:
        parser.error("--N should not be used with --dataFileName")

    if args.dataFileName is None and args.N is None:
        parser.error("--N must be used when --dataFileName is not specified")

    if args.linearModelName is not None and args.nonLinearModelName is not None:
        parser.error(
            "--nonLinearModelName should not be used with --linearModelName. One or the other!"
        )
    if args.linearModelName is None and args.nonLinearModelName is None:
        parser.error("--nonLinearModelName OR --linearModelName must be used.")

    model_name = (
        args.nonLinearModelName
        if args.linearModelName is None
        else args.linearModelName
    )

    return args, model_name


def main() -> None:
    args, model_name = parse_arguments()

    setup_logging(args.verbose)

    try:
        if args.dataFileName is not None:
            runner = NonLinearPPFRunnerFromFile(
                model_name=model_name,
                n_particles=args.n_particles,
                data_filename=args.dataFileName,
                verbose=args.verbose,
                plot=args.plot,
                save_history=args.saveHistory,
            )
        else:
            runner = NonLinearPPFRunnerSim(
                model_name=model_name,
                N=args.N,
                sKey=args.sKey,
                n_particles=args.n_particles,
                verbose=args.verbose,
                plot=args.plot,
                save_history=args.saveHistory,
            )

        runner.run()

    except NumericalError as e:
        print(f"[ERREUR NUMÉRIQUE] {e}", file=sys.stderr)
        sys.exit(1)
    except FilterError as e:
        print(f"[ERREUR FILTRE] {e}", file=sys.stderr)
    except PKFError as e:
        print(f"[ERREUR PKF] {e}", file=sys.stderr)
        sys.exit(1)
    except ParamError as e:
        print(f"[ERREUR PARAMÈTRE] {e}", file=sys.stderr)
        sys.exit(2)
    except ValueError as e:
        print(f"[ERREUR PARAMÈTRE] {e}", file=sys.stderr)
        sys.exit(2)
    except RuntimeError as e:
        print(f"[ERREUR EXÉCUTION] {e}", file=sys.stderr)
        sys.exit(3)


if __name__ == "__main__":
    main()
