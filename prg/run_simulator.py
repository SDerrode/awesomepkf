#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import logging

from base_classes.simulator_linear import LinearDataSimulator
from base_classes.simulator_nonlinear import NonLinearDataSimulator


# =============================================================
# Logger
# =============================================================

def setup_logging(verbose: int) -> None:
    level = {0: logging.WARNING, 1: logging.INFO}.get(verbose, logging.DEBUG)
    logging.basicConfig(
        level=level,
        format="[%(levelname)s] %(name)s: %(message)s"
    )


# =============================================================
# Parser
# =============================================================

def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Simulate linear or non-linear data"
    )

    # Arguments communs
    parser.add_argument("--N", type=int, default=1000, help="Number of samples")
    parser.add_argument("--sKey", type=int, default=None, help="Seed key")
    parser.add_argument("--dataFileName", default=None, help="Output filename")
    parser.add_argument("--verbose", type=int, default=1, help="Verbosity level")
    parser.add_argument("--withoutX", action="store_true", help="Do not store X in output")

    # Arguments spécifiques
    parser.add_argument("--linearModelName", help="Name of linear model")
    parser.add_argument("--nonLinearModelName", help="Name of non-linear model")

    return parser.parse_args()


# =============================================================
# Main
# =============================================================

def main() -> None:
    args = parse_arguments()
    setup_logging(args.verbose)

    if args.linearModelName and args.nonLinearModelName:
        raise ValueError("Please provide only one of --linearModelName or --nonLinearModelName")

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
        raise ValueError("Please provide either --linearModelName or --nonLinearModelName")

    simulator.run()


if __name__ == "__main__":
    main()
