#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import sys

from prg.base_classes.simulator_linear import LinearDataSimulator
from prg.base_classes.simulator_nonlinear import NonLinearDataSimulator
from prg.utils.parser import add_arguments


# =============================================================
# Parser
# =============================================================


def _print_model_list() -> None:
    """Affiche les modèles linéaires et non-linéaires disponibles puis quitte."""
    from prg.models.nonLinear import ModelFactoryNonLinear
    from prg.models.linear import ModelFactoryLinear

    nl_models = sorted(ModelFactoryNonLinear.list_models())
    lin_models = sorted(ModelFactoryLinear.list_models())

    print("\nModèles non-linéaires disponibles :")
    for name in nl_models:
        print(f"  {name}")

    print("\nModèles linéaires disponibles :")
    for name in lin_models:
        print(f"  {name}")

    print()


def parse_arguments():
    parser = argparse.ArgumentParser(description="Simulate linear or non-linear data")

    parser.add_argument(
        "--list-models",
        action="store_true",
        help="Affiche les modèles disponibles et quitte",
    )

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

    # Traitement immédiat de --list-models (avant toute validation)
    if args.list_models:
        _print_model_list()
        sys.exit(0)

    # Validation logique
    if args.linearModelName is not None and args.nonLinearModelName is not None:
        parser.error(
            "--nonLinearModelName should not be used with --linearModelName. One or the other!"
        )
    if args.linearModelName is None and args.nonLinearModelName is None:
        parser.error("--nonLinearModelName OR --linearModelName must be used.")

    return args


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
