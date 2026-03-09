#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse

from prg.base_classes.simulator_linear import LinearDataSimulator
from prg.base_classes.simulator_nonlinear import NonLinearDataSimulator
from prg.utils.parser import add_arguments


# =============================================================
# Parser
# =============================================================


def parse_arguments():
    parser = argparse.ArgumentParser(description="Simulate linear or non-linear data")

    add_arguments(
        parser,
        [
            "linearModelName",
            "nonLinearModelName",
            "N",
            "sKey",
            "dataFileName",
            "withoutX",
        ],
    )

    args = parser.parse_args()

    # Validation logique
    if args.linearModelName is not None and args.nonLinearModelName is not None:
        parser.error(
            "--nonLinearModelName should not be used with --linearModelName. One or the other!"
        )
    if args.linearModelName is None and args.nonLinearModelName is None:
        parser.error("--nonLinearModelName OR --linearModelName must be used.")

    return parser.parse_args()


# =============================================================
# Main
# =============================================================


def main() -> None:
    args = parse_arguments()

    if args.linearModelName and args.nonLinearModelName:
        raise ValueError(
            "Please provide only one of --linearModelName or --nonLinearModelName"
        )

    if args.linearModelName:
        simulator = LinearDataSimulator(
            model_name=args.linearModelName,
            N=args.N,
            sKey=args.sKey,
            data_file_name=args.dataFileName,
            verbose=args.verbose,
            withoutX=args.withoutX,
        )

    elif args.nonLinearModelName:
        simulator = NonLinearDataSimulator(
            model_name=args.nonLinearModelName,
            N=args.N,
            sKey=args.sKey,
            data_file_name=args.dataFileName,
            verbose=args.verbose,
            withoutX=args.withoutX,
        )

    else:
        raise ValueError(
            "Please provide either --linearModelName or --nonLinearModelName"
        )

    try:
        simulator.run()
    except RuntimeError as rte:
        raise


if __name__ == "__main__":
    main()
