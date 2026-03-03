#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse

from prg.base_classes.nonlinear_upkf_runner_simulation import BaseNonLinearUPKFRunnerSim
from prg.base_classes.nonlinear_upkf_runner_from_file import (
    BaseNonLinearUPKFRunnerFromFile,
)
from prg.utils.parser import addParseToParser


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run NonLinear UPKF")

    addParseToParser(
        parser,
        [
            "nonLinearModelName",
            "linearModelName",
            "N",
            "sKey",
            "sigmaSet",
            "dataFileName",
        ],
    )

    args = parser.parse_args()

    # Validation logique
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

    if args.linearModelName is None:
        model_name = args.nonLinearModelName
    else:
        model_name = args.linearModelName

    return args, model_name


def main() -> None:
    args, model_name = parse_arguments()

    # 🔎 Distinction automatique selon dataFileName
    if args.dataFileName is not None:

        # Mode FILE
        runner = BaseNonLinearUPKFRunnerFromFile(
            model_name=model_name,
            sigmaSet=args.sigmaSet,
            data_filename=args.dataFileName,
            verbose=args.verbose,
            plot=args.plot,
            save_history=args.saveHistory,
        )
    else:

        # Mode SIM
        runner = BaseNonLinearUPKFRunnerSim(
            model_name=model_name,
            N=args.N,
            sKey=args.sKey,
            sigmaSet=args.sigmaSet,
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
