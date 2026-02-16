#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse

from base_classes.nonlinear_epkf_runner_simulation import NonLinearEPKFRunner
from base_classes.nonlinear_epkf_runner_from_file import NonLinearEPKFRunnerFromFile
from others.parser import addParseToParser


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run NonLinear EPKF"
    )

    parser.add_argument(
        "--mode",
        choices=["sim", "file"],
        required=True,
        help="Execution mode: sim (simulation) or file (from CSV)"
    )

    addParseToParser(
        parser,
        ['nonLinearModelName', 'N', 'sKey', 'ell', 'dataFileName']
    )

    return parser.parse_args()


def main() -> None:
    args = parse_arguments()

    if args.mode == "sim":
        runner = NonLinearEPKFRunner(
            model_name=args.nonLinearModelName,
            N=args.N,
            sKey=args.sKey,
            ell=args.ell,
            verbose=args.verbose,
            plot=args.plot,
            save_history=args.saveHistory,
        )

    elif args.mode == "file":
        runner = NonLinearEPKFRunnerFromFile(
            model_name=args.nonLinearModelName,
            ell=args.ell,
            data_filename=args.dataFileName,
            verbose=args.verbose,
            plot=args.plot,
            save_history=args.saveHistory,
        )

    runner.run()


if __name__ == "__main__":
    main()
