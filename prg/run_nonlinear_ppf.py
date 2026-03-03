#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import sys

from prg.base_classes.nonlinear_ppf_runner_simulation import BaseNonLinearPPFRunnerSim
from prg.base_classes.nonlinear_ppf_runner_from_file import (
    BaseNonLinearPPFRunnerFromFile,
)
from prg.utils.parser import addParseToParser
from prg.exceptions import NumericalError, FilterError, PKFError, ParamError


def parse_arguments():
    parser = argparse.ArgumentParser(description="Run NonLinear PPF")

    addParseToParser(
        parser,
        [
            "nonLinearModelName",
            "linearModelName",
            "N",
            "sKey",
            "nbParticles",
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

    try:
        if args.dataFileName is not None:
            runner = BaseNonLinearPPFRunnerFromFile(
                model_name=model_name,
                nbParticles=args.nbParticles,
                data_filename=args.dataFileName,
                verbose=args.verbose,
                plot=args.plot,
                save_history=args.saveHistory,
            )
        else:
            runner = BaseNonLinearPPFRunnerSim(
                model_name=model_name,
                N=args.N,
                sKey=args.sKey,
                nbParticles=args.nbParticles,
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
        sys.exit(1)
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
