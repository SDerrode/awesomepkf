#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse


def int_ge_1(value: str) -> int:
    try:
        ivalue = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"{value} n'est pas un entier valide")

    if ivalue < 1:
        raise argparse.ArgumentTypeError(f"{value} doit être un entier ≥ 1")

    return ivalue


def addParseToParser(parser, listOptions):
    """
    Ajoute des arguments à un ArgumentParser.

    :param parser: argparse.ArgumentParser
    :param listOptions: liste de chaînes, options optionnelles à ajouter
    """

    # =========================
    # Options toujours disponibles
    # =========================
    parser.add_argument(
        "--verbose",
        type=int,
        choices=range(0, 3),
        default=0,
        dest="verbose",
        help="Set the verbose level (0=quiet, 2=maximum, default=0)",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        dest="plot",
        help="Plot and signals on disk (default: True if not specified)",
    )

    parser.add_argument(
        "--saveHistory",
        action="store_true",
        dest="saveHistory",
        help="Save parameters trace on disk (default: False if not specified)",
    )

    # =========================
    # Options optionnelles configurables
    # =========================
    option_config = {
        "N": {
            "type": int,
            "default": None,
            "help": "Set the number of samples to process (default: None)",
        },
        "nbParticles": {
            "type": int,
            "default": 300,
            "help": "Set the number of particles to deal with (default: 300)",
        },
        "sKey": {
            "type": int,
            "default": None,
            "help": "Set the random generator seed (default: None)",
        },
        "linearModelName": {
            "choices": [
                "A_mQ_x1_y1",
                "A_mQ_x1_y1_VPgreaterThan1",
                "A_mQ_x1_y1_augmented",
                "A_mQ_x2_y2",
                "A_mQ_x3_y1",
                "Sigma_x1_y1",
                "Sigma_x2_y2",
                "Sigma_x3_y1",
            ],
            "default": None,
            "help": "Linear model to process data (default: None)",
        },
        "sigmaSet": {
            "choices": ["wan2000", "cpkf", "lerner2002", "ito2000"],
            "default": "wan2000",
            "help": "Sigma set points to use with UPKF (default: wan2000)",
        },
        "nonLinearModelName": {
            "choices": [
                "x1_y1_cubique",
                "x1_y1_ext_saturant",
                "x1_y1_gordon",
                "x1_y1_sinus",
                "x1_y1_withRetroactions",
                "x1_y1_withRetroactions_augmented",
                "x2_y1",
                "x2_y1_rapport",
                "x2_y1_withRetroactionsOfObservations",
                "x2_y1_withRetroactionsOfObservations_augmented",
                "x2_y2_withRetroactions",
            ],
            "default": None,
            "help": "Non linear model to process data (default: None)",
        },
        "dataFileName": {
            "type": str,
            "default": None,
            "help": "Full path where trajectories are stored (default: None)",
        },
        "withoutX": {
            "action": "store_true",
            "default": False,
            "help": "Save true X or not (default: False)",
        },
    }

    # Ajouter dynamiquement les options demandées
    for opt in listOptions:
        if opt in option_config:
            kwargs = option_config[opt].copy()
            # Définir le nom de destination
            kwargs["dest"] = opt
            parser.add_argument(f"--{opt}", **kwargs)
