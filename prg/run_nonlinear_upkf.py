#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse

from base_classes.nonlinear_upkf_runner_simulation import BaseNonLinearUPKFRunnerSim
from base_classes.nonlinear_upkf_runner_from_file import BaseNonLinearUPKFRunnerFromFile
from others.parser import addParseToParser


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run NonLinear UPKF"
    )

    addParseToParser(
        parser,
        ['nonLinearModelName', 'N', 'sKey', 'sigmaSet', 'dataFileName']
    )

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
        runner = BaseNonLinearUPKFRunnerFromFile(
            model_name=args.nonLinearModelName,
            sigmaSet=args.sigmaSet,
            data_filename=args.dataFileName,
            verbose=args.verbose,
            plot=args.plot,
            save_history=args.saveHistory,
        )
    else:
        
        # Mode SIM
        runner = BaseNonLinearUPKFRunnerSim(
            model_name=args.nonLinearModelName,
            N=args.N,
            sKey=args.sKey,
            sigmaSet=args.sigmaSet,
            verbose=args.verbose,
            plot=args.plot,
            save_history=args.saveHistory,
        )

    runner.run()


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        printf(f'{e}')
        
