#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse

from base_classes.nonlinear_pf_runner_simulation import NonLinearPFRunner
from base_classes.nonlinear_pf_runner_from_file import NonLinearPFRunnerFromFile
from others.parser import addParseToParser


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run NonLinear PF"
    )

    parser.add_argument(
        "--mode",
        choices=["sim", "file"],
        required=True,
        help="Execution mode: sim (simulation) or file (from CSV)"
    )

    addParseToParser(
        parser,
        ['nonLinearModelName', 'N', 'sKey', 'nbParticles', 'dataFileName']
    )

    return parser.parse_args()


def main() -> None:
    args = parse_arguments()

    if args.mode == "sim":
        runner = NonLinearPFRunner(
            model_name=args.nonLinearModelName,
            N=args.N,
            sKey=args.sKey,
            nbParticles=args.nbParticles,
            verbose=args.verbose,
            plot=args.plot,
            save_history=args.saveHistory,
        )

    elif args.mode == "file":
        runner = NonLinearPFRunnerFromFile(
            model_name=args.nonLinearModelName,
            nbParticles=args.nbParticles,
            data_filename=args.dataFileName,
            verbose=args.verbose,
            plot=args.plot,
            save_history=args.saveHistory,
        )

    runner.run()


if __name__ == "__main__":
    main()
