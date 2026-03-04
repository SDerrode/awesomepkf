#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import chardet
import sys


def detect_encoding(file_path, n_bytes=50000):
    """Détecte l'encodage d'un fichier texte."""
    try:
        with open(file_path, "rb") as f:
            raw_data = f.read(n_bytes)
        result = chardet.detect(raw_data)
        encoding = result["encoding"] if result["encoding"] else "utf-8"
        return encoding
    except Exception as e:
        return "utf-8"


def csv_to_parquet(csv_file_path, parquet_file_path, engine="pyarrow"):
    """Convertit un CSV en Parquet de manière robuste."""
    try:
        # Détecter l'encodage
        encoding = detect_encoding(csv_file_path)

        # Lire le CSV
        df = pd.read_csv(csv_file_path, encoding=encoding)

        # Sauvegarder en Parquet
        df.to_parquet(parquet_file_path, engine=engine, index=False)

    except (FileNotFoundError, Exception) as e:
        raise


# --- Exemple d'utilisation ---
if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage : python3 csv_to_parquet.py <fichier.csv> <fichier.parquet>")
    else:
        csv_file = sys.argv[1]
        parquet_file = sys.argv[2]
        csv_to_parquet(csv_file, parquet_file)
