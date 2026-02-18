#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse

from base_classes.nonlinear_pf_runner_simulation import BaseNonLinearPFRunnerSim
from base_classes.nonlinear_pf_runner_from_file import BaseNonLinearPFRunnerFromFile
from others.parser import addParseToParser


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run NonLinear PF"
    )

    addParseToParser(
        parser,
        ['nonLinearModelName', 'N', 'sKey', 'nbParticles', 'dataFileName']
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
        runner = BaseNonLinearPFRunnerFromFile(
            model_name=args.nonLinearModelName,
            nbParticles=args.nbParticles,
            data_filename=args.dataFileName,
            verbose=args.verbose,
            plot=args.plot,
            save_history=args.saveHistory,
        )
    else:
        
        # Mode SIM
        runner = BaseNonLinearPFRunnerSim(
            model_name=args.nonLinearModelName,
            N=args.N,
            sKey=args.sKey,
            nbParticles=args.nbParticles,
            verbose=args.verbose,
            plot=args.plot,
            save_history=args.saveHistory,
        )

    runner.run()


if __name__ == "__main__":
    main()
