#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse

from base_classes.linear_pkf_runner_simulation import LinearPKFRunner
from base_classes.linear_pkf_runner_from_file import LinearPKFRunnerFromFile
from others.parser import addParseToParser


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Linear PKF"
    )

    parser.add_argument(
        "--mode",
        choices=["sim", "file"],
        required=True,
        help="Execution mode: sim (simulation) or file (from CSV)"
    )

    addParseToParser(
        parser,
        ['linearModelName', 'N', 'sKey', 'dataFileName']
    )

    return parser.parse_args()


def main() -> None:
    args = parse_arguments()

    if args.mode == "sim":
        runner = LinearPKFRunner(
            model_name=args.linearModelName,
            N=args.N,
            sKey=args.sKey,
            verbose=args.verbose,
            plot=args.plot,
            save_history=args.saveHistory,
        )

    elif args.mode == "file":
        runner = LinearPKFRunnerFromFile(
            model_name=args.linearModelName,
            data_filename=args.dataFileName,
            verbose=args.verbose,
            plot=args.plot,
            save_history=args.saveHistory,
        )

    runner.run()


if __name__ == "__main__":
    main()
