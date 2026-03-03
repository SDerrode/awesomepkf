#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse

from prg.base_classes.linear_pkf_runner_simulation import LinearPKFRunnerSim
from prg.base_classes.linear_pkf_runner_from_file import LinearPKFRunnerFromFile
from prg.utils.parser import addParseToParser


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Linear PKF")

    addParseToParser(parser, ["linearModelName", "N", "sKey", "dataFileName"])

    args = parser.parse_args()

    # Validation logique
    if args.dataFileName is not None and args.N is not None:
        parser.error("--N should not be used with --dataFileName")

    if args.dataFileName is None and args.N is None:
        parser.error("--N must be used when --dataFileName is not specified")

    return args


def main() -> None:
    args = parse_arguments()

    # 🔎 Distinction automatique selon dataFileName
    if args.dataFileName is not None:

        # Mode FILE
        runner = LinearPKFRunnerFromFile(
            model_name=args.linearModelName,
            data_filename=args.dataFileName,
            verbose=args.verbose,
            plot=args.plot,
            save_history=args.saveHistory,
        )
    else:
        # Mode SIM
        runner = LinearPKFRunnerSim(
            model_name=args.linearModelName,
            N=args.N,
            sKey=args.sKey,
            verbose=args.verbose,
            plot=args.plot,
            save_history=args.saveHistory,
        )

    try:
        runner.run()
    except RuntimeError as rte:
        raise


if __name__ == "__main__":
    main()
