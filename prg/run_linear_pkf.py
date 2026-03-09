#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import sys

from prg.base_classes.linear_pkf_runner_simulation import LinearPKFRunnerSim
from prg.base_classes.linear_pkf_runner_from_file import LinearPKFRunnerFromFile
from prg.utils.parser import add_arguments
from prg.utils.exceptions import NumericalError, FilterError, PKFError, ParamError


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Linear PKF")

    add_arguments(parser, ["linearModelName", "N", "sKey", "dataFileName"])

    args = parser.parse_args()

    if args.dataFileName is not None and args.N is not None:
        parser.error("--N should not be used with --dataFileName")

    if args.dataFileName is None and args.N is None:
        parser.error("--N must be used when --dataFileName is not specified")

    return args


def main() -> None:
    args = parse_arguments()

    try:
        if args.dataFileName is not None:
            runner = LinearPKFRunnerFromFile(
                model_name=args.linearModelName,
                data_filename=args.dataFileName,
                verbose=args.verbose,
                plot=args.plot,
                save_history=args.saveHistory,
            )
        else:
            runner = LinearPKFRunnerSim(
                model_name=args.linearModelName,
                N=args.N,
                sKey=args.sKey,
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
