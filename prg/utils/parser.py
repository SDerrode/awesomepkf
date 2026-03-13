#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import warnings


def int_ge_1(value: str) -> int:
    """Convertit *value* en ``int`` et vérifie qu'il est ≥ 1."""
    try:
        ivalue = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"{value!r} n'est pas un entier valide")

    if ivalue < 1:
        raise argparse.ArgumentTypeError(f"{value} doit être un entier ≥ 1")

    return ivalue


# ---------------------------------------------------------------------------
# Configuration des options optionnelles
# ---------------------------------------------------------------------------

_OPTION_CONFIG: dict = {
    "N": {
        "type": int_ge_1,
        "default": None,
        "help": "Nombre d'échantillons à traiter (default: None)",
    },
    "n_particles": {
        "type": int_ge_1,
        "default": 300,
        "help": "Nombre de particules à utiliser (default: 300)",
    },
    "sKey": {
        "type": int,
        "default": None,
        "help": "Graine du générateur aléatoire (default: None)",
    },
    "linearModelName": {
        "choices": [
            "model_x1_y1_AQ_augmented",
            "model_x1_y1_AQ_classic",
            "model_x1_y1_AQ_pairwise",
            "model_x1_y1_Sigma_pairwise",
            "model_x2_y2_AQ_augmented",
            "model_x2_y2_AQ_classic",
            "model_x2_y2_AQ_pairwise",
            "model_x2_y2_Sigma_pairwise",
            "model_x3_y1_AQ_augmented",
            "model_x3_y1_AQ_classic",
            "model_x3_y1_AQ_pairwise",
            "model_x3_y1_Sigma_pairwise",
        ],
        "default": None,
        "help": "Modèle linéaire à utiliser (default: None)",
    },
    "sigmaSet": {
        "choices": ["wan2000", "cpkf", "lerner2002", "ito2000"],
        "default": "wan2000",
        "help": "Ensemble de sigma points pour UPKF (default: wan2000)",
    },
    "nonLinearModelName": {
        "choices": [
            "model_x1_y1_augmented",
            "model_x1_y1_Cubique_classic",
            "model_x1_y1_ExpSaturant_classic",
            "model_x1_y1_Gordon_classic",
            "model_x1_y1_pairwise",
            "model_x1_y1_Sinus_classic",
            "model_x2_y1_augmented",
            "model_x2_y1_classic",
            "model_x2_y1_pairwise",
            "model_x2_y1_Rapport_classic",
            "model_x2_y2_pairwise",
        ],
        "default": None,
        "help": "Modèle non-linéaire à utiliser (default: None)",
    },
    "dataFileName": {
        "type": str,
        "default": None,
        "help": "Chemin du fichier de trajectoires (default: None)",
    },
    "withoutX": {
        "action": "store_true",
        "help": "Ne pas sauvegarder l'état vrai X (default: False)",
    },
    "filter": {
        "choices": ["EPKF", "UPKF", "PPF", "UKF", "PKF"],
        "default": None,
        "help": "Type de filtre à utiliser (default: None)",
    },
}


# ---------------------------------------------------------------------------
# API publique
# ---------------------------------------------------------------------------


def add_arguments(parser: argparse.ArgumentParser, list_options: list[str]) -> None:
    """
    Ajoute des arguments à un ``ArgumentParser``.

    Parameters
    ----------
    parser : argparse.ArgumentParser
        Le parseur à enrichir.
    list_options : list[str]
        Noms des options optionnelles à ajouter (clés de ``_OPTION_CONFIG``).

    Raises
    ------
    None — les clés inconnues émettent un ``UserWarning`` au lieu d'être
    ignorées silencieusement.
    """

    # --- Options toujours disponibles ---
    parser.add_argument(
        "--verbose",
        type=int,
        choices=range(0, 3),
        default=0,
        help="Niveau de verbosité (0=silencieux, 2=maximum, default=0)",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Affiche et sauvegarde les signaux sur disque (default: False)",  # FIX: was "True if not specified" → inverted
    )
    parser.add_argument(
        "--saveHistory",
        action="store_true",
        help="Sauvegarde la trace des paramètres sur disque (default: False)",
    )

    # --- Options optionnelles configurables ---
    for opt in list_options:
        if opt not in _OPTION_CONFIG:
            # FIX: unknown option → explicit warning instead of silently ignoring
            warnings.warn(
                f"add_arguments : option inconnue {opt!r} ignorée "
                f"(options disponibles : {list(_OPTION_CONFIG)})",
                UserWarning,
                stacklevel=2,
            )
            continue

        kwargs = _OPTION_CONFIG[opt].copy()
        # FIX: dest=opt removed — argparse infers it automatically from --opt
        parser.add_argument(f"--{opt}", **kwargs)
