#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse

from base_classes.nonlinear_pf_runner_simulation import BaseNonLinearPFRunnerSim
from base_classes.nonlinear_pf_runner_from_file  import BaseNonLinearPFRunnerFromFile
from others.parser import addParseToParser


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run NonLinear PF"
    )

    addParseToParser(
        parser,
        ['nonLinearModelName', 'linearModelName', 'N', 'sKey', 'nbParticles', 'dataFileName']
    )

    args = parser.parse_args()

    # Validation logique
    if args.dataFileName is not None and args.N is not None:
        parser.error("--N should not be used with --dataFileName")

    if args.dataFileName is None and args.N is None:
        parser.error("--N must be used when --dataFileName is not specified")
        
    if args.linearModelName is not None and args.nonLinearModelName is not None:
        parser.error("--nonLinearModelName should not be used with --linearModelName. One or the other!")
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
        runner = BaseNonLinearPFRunnerFromFile(
            model_name=model_name,
            nbParticles=args.nbParticles,
            data_filename=args.dataFileName,
            verbose=args.verbose,
            plot=args.plot,
            save_history=args.saveHistory,
        )
    else:
        
        # Mode SIM
        runner = BaseNonLinearPFRunnerSim(
            model_name=model_name,
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
