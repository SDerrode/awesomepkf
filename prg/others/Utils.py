#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import logging
import numpy as np
import pandas as pd
import scipy as sc

import csv
import chardet # detection du format d'encodage des fichiers


# ----------------------------------------------------------------------
# Configuration globale du logging
# ----------------------------------------------------------------------
logging.basicConfig(format="[%(levelname)s] %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)


def rmse(X1, X2):
    X1 = np.asarray(X1).ravel()  # Aplatir en 1D
    X2 = np.asarray(X2).ravel()
    if X1.shape != X2.shape:
        raise ValueError(f" Les arrays doivent avoir la même forme : {X1.shape} vs {X2.shape}")
    mse = np.mean((X1 - X2)**2)
    return np.sqrt(mse)

def read_unknown_file(filepath: str, nrows_detect: int = 500, verbose : int = 0):
    """
    Lecture robuste d'un fichier de données en devinant :
    - le format (csv, parquet, json, excel…)
    - le séparateur (pour les fichiers texte)
    - la présence d'un en-tête
    - l'encodage
    """

    ext = os.path.splitext(filepath)[1].lower()

    try:
        # --- Détection de l'encodage ---
        with open(filepath, 'rb') as f:
            raw_data = f.read(50000)
            encoding_info = chardet.detect(raw_data)
            encoding = encoding_info['encoding'] or 'utf-8'
            confidence = encoding_info.get('confidence', 0)
        if verbose>0:
            logger.info(f"🧬 Encodage détecté : {encoding} (confiance={confidence:.2f})")

        # --- Lecture selon le format ---
        if ext == ".parquet":
            if verbose>0:
                logger.info(f"📦 Lecture Parquet : {filepath}")
            return pd.read_parquet(filepath)

        elif ext == ".json":
            logger.info(f"🧾 Lecture JSON : {filepath}")
            return pd.read_json(filepath, encoding=encoding)

        elif ext in [".xlsx", ".xls"]:
            if verbose>0:
                logger.info(f"📊 Lecture Excel : {filepath}")
            return pd.read_excel(filepath)

        elif ext in [".csv", ".txt", ".dat", ".tsv", ""]:
            if verbose>0:
                logger.info(f"📑 Lecture texte délimité : {filepath}")

            # Échantillon pour la détection du séparateur + header
            with open(filepath, "r", encoding=encoding) as f:
                sample_lines = [next(f) for _ in range(min(nrows_detect, 10))]
                sample = "".join(sample_lines)

                # Détection du séparateur
                try:
                    dialect = csv.Sniffer().sniff(sample, delimiters=",;\t| ")
                    sep = dialect.delimiter
                except Exception:
                    sep = ','  # valeur par défaut
                    logger.warning("⚠️ Impossible de détecter le séparateur, utilisation de ',' par défaut.")

                # Détection de la présence d'un en-tête
                has_header = csv.Sniffer().has_header(sample)
                if verbose>0:
                    logger.info(f"➡️ Séparateur : '{sep}' | Header : {has_header}")

            df = pd.read_csv(filepath, sep=sep, header=0 if has_header else None, encoding=encoding)
            return df

        else:
            raise ValueError(f"❌ Format de fichier non reconnu : {ext}")

    except Exception as e:
        logger.error(f"❌ Erreur lors de la lecture du fichier {filepath} : {e}")
        raise

def file_data_generator(filename: str, dim_x: int, verbose :int = 0):
    """
    Générateur qui lit les données d'un DataFrame pandas.
    Suppose que df contient au moins dim_x + dim_y colonnes (x puis y).
    """

    df = read_unknown_file(filename, verbose=verbose)
    
    for k, row in df.iterrows():
        values     = row.values.reshape(-1, 1)
        xkp1, ykp1 = np.split(values, [dim_x])
        yield k, (xkp1, ykp1)
        
        