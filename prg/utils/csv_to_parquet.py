#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
csv_to_parquet.py
-----------------
Convertit un fichier CSV en Parquet de manière robuste.

Usage
-----
    python3 csv_to_parquet.py <fichier.csv> <fichier.parquet> [--engine pyarrow|fastparquet]
"""

import argparse
import sys
import warnings
from pathlib import Path

import chardet
import pandas as pd


# ---------------------------------------------------------------------------
# Constantes
# ---------------------------------------------------------------------------

# Nombre d'octets lus pour la détection d'encodage.
# 50 000 octets couvre la majorité des cas ; augmenter si des fichiers
# à encodage rare sont fréquents dans votre corpus.
_ENCODING_SAMPLE_BYTES = 50_000

_SUPPORTED_ENGINES = ("pyarrow", "fastparquet")


# ---------------------------------------------------------------------------
# Fonctions
# ---------------------------------------------------------------------------


def detect_encoding(file_path: Path, n_bytes: int = _ENCODING_SAMPLE_BYTES) -> str:
    """
    Détecte l'encodage d'un fichier texte via ``chardet``.

    Parameters
    ----------
    file_path : Path
        Chemin vers le fichier à analyser.
    n_bytes : int
        Nombre d'octets à lire pour l'analyse (default: 50 000).

    Returns
    -------
    str
        Encodage détecté, ou ``"utf-8"`` si la détection échoue.

    Warns
    -----
    UserWarning
        Si la détection échoue, un avertissement est émis avant le fallback.
        (FIX : l'original avalait silencieusement l'exception sans aucun log)
    """
    try:
        with open(file_path, "rb") as f:
            raw_data = f.read(n_bytes)
    except OSError as e:
        # FIX : on remonte l'erreur d'accès fichier explicitement
        raise OSError(
            f"Impossible de lire '{file_path}' pour la détection d'encodage : {e}"
        ) from e

    result = chardet.detect(raw_data)
    encoding = result.get("encoding")

    if not encoding:
        # FIX : fallback visible (avertissement) au lieu d'un silence total
        warnings.warn(
            f"Encodage non détecté pour '{file_path}' — fallback sur utf-8.",
            UserWarning,
            stacklevel=2,
        )
        return "utf-8"

    return encoding


def csv_to_parquet(
    csv_path: Path | str,
    parquet_path: Path | str,
    engine: str = "pyarrow",
) -> None:
    """
    Convertit un fichier CSV en Parquet.

    Parameters
    ----------
    csv_path : Path | str
        Chemin vers le fichier CSV source.
    parquet_path : Path | str
        Chemin vers le fichier Parquet de sortie.
    engine : {"pyarrow", "fastparquet"}
        Moteur Parquet à utiliser (default: ``"pyarrow"``).

    Raises
    ------
    ValueError
        Si ``engine`` n'est pas supporté.
    FileNotFoundError
        Si ``csv_path`` n'existe pas.
    OSError
        Pour tout autre problème d'accès fichier.
    """
    csv_path = Path(csv_path)
    parquet_path = Path(parquet_path)

    # FIX : validation de l'engine avec message clair (pandas lève une erreur cryptique sinon)
    if engine not in _SUPPORTED_ENGINES:
        raise ValueError(
            f"Engine {engine!r} non supporté. " f"Choisir parmi : {_SUPPORTED_ENGINES}"
        )

    # FIX : vérification explicite de l'existence du fichier source
    if not csv_path.exists():
        raise FileNotFoundError(f"Fichier CSV introuvable : '{csv_path}'")

    encoding = detect_encoding(csv_path)
    print(f"  Encodage détecté : {encoding}")

    df = pd.read_csv(csv_path, encoding=encoding)
    print(f"  Lignes / colonnes : {df.shape[0]:,} × {df.shape[1]}")

    # Créer le répertoire de sortie si nécessaire
    parquet_path.parent.mkdir(parents=True, exist_ok=True)

    df.to_parquet(parquet_path, engine=engine, index=False)
    print(f"  ✔ Parquet écrit : '{parquet_path}'")


# ---------------------------------------------------------------------------
# Point d'entrée CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Convertit un fichier CSV en Parquet.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("csv_file", help="Fichier CSV source")
    p.add_argument("parquet_file", help="Fichier Parquet de sortie")
    # FIX : --engine exposé en argument CLI (l'original ne permettait pas de le choisir)
    p.add_argument(
        "--engine",
        choices=_SUPPORTED_ENGINES,
        default="pyarrow",
        help="Moteur Parquet à utiliser",
    )
    return p


if __name__ == "__main__":
    args = _build_parser().parse_args()

    # FIX : feedback utilisateur explicite en cas de succès ou d'échec
    try:
        print(f"Conversion : '{args.csv_file}' → '{args.parquet_file}'")
        csv_to_parquet(args.csv_file, args.parquet_file, engine=args.engine)
        print("Conversion terminée avec succès.")
    except Exception as e:
        print(f"❌ Erreur : {e}", file=sys.stderr)
        sys.exit(1)
