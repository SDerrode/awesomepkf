#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse

from base_classes.nonlinear_upkf_runner_simulation import NonLinearUPKFRunner
from base_classes.nonlinear_upkf_runner_from_file import NonLinearUPKFRunnerFromFile
from others.parser import addParseToParser


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run NonLinear UPKF"
    )

    parser.add_argument(
        "--mode",
        choices=["sim", "file"],
        required=True,
        help="Execution mode: sim (simulation) or file (from CSV)"
    )

    addParseToParser(
        parser,
        ['nonLinearModelName', 'N', 'sKey', 'sigmaSet', 'dataFileName']
    )

    return parser.parse_args()


def main() -> None:
    args = parse_arguments()

    if args.mode == "sim":
        runner = NonLinearUPKFRunner(
            model_name=args.nonLinearModelName,
            N=args.N,
            sKey=args.sKey,
            sigmaSet=args.sigmaSet,
            verbose=args.verbose,
            plot=args.plot,
            save_history=args.saveHistory,
        )

    elif args.mode == "file":
        runner = NonLinearUPKFRunnerFromFile(
            model_name=args.nonLinearModelName,
            sigmaSet=args.sigmaSet,
            data_filename=args.dataFileName,
            verbose=args.verbose,
            plot=args.plot,
            save_history=args.saveHistory,
        )

    runner.run()


if __name__ == "__main__":
    main()
