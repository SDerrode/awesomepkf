from __future__ import annotations

import os
import logging
import numpy as np
import pandas as pd
import csv
import chardet
from typing import Generator, Optional, Any, Union

# ----------------------------------------------------------------------
# Logging global
# ----------------------------------------------------------------------
logging.basicConfig(format="[%(levelname)s] %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)

def save_dataframe_to_csv(df, filepath, index=False):
    """Enregistre un DataFrame en CSV (UTF-8) sans sauvegarder l'index."""
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)  # Crée le dossier si besoin
    
    try:
        df.to_csv(path, encoding="utf-8", index=index, float_format="%.6f")
        logger.info(f"✅ Fichier enregistré avec succès : {path.resolve()}")
    except Exception as e:
        logger.error(f"❌ Erreur lors de l'enregistrement du CSV : {e}")
        raise


# ----------------------------------------------------------------------
# RMSE
# ----------------------------------------------------------------------
def rmse(X1: Union[np.ndarray, list], X2: Union[np.ndarray, list]) -> float:
    X1_arr = np.asarray(X1).ravel()
    X2_arr = np.asarray(X2).ravel()
    if __debug__:
        if X1_arr.shape != X2_arr.shape:
            raise ValueError(f"❌ Arrays must have the same shape : {X1_arr.shape} vs {X2_arr.shape}")
    mse = np.mean((X1_arr - X2_arr) ** 2)
    return float(np.sqrt(mse))

# ----------------------------------------------------------------------
# Lecture de fichiers inconnus
# ----------------------------------------------------------------------
def read_unknown_file(filepath: str, nrows_detect: int = 500, verbose: int = 0) -> pd.DataFrame:
    ext = os.path.splitext(filepath)[1].lower()
    try:
        # --- Détection de l'encodage ---
        with open(filepath, 'rb') as f:
            raw_data = f.read(50000)
            enc_info = chardet.detect(raw_data)
            encoding = enc_info['encoding'] or 'utf-8'
            confidence = enc_info.get('confidence', 0)
        if verbose > 0:
            logger.info(f"🧬 Encodage détecté : {encoding} (confiance={confidence:.2f})")

        # --- Lecture selon l'extension ---
        if ext == ".parquet":
            if verbose > 0: logger.info(f"📦 Lecture Parquet : {filepath}")
            return pd.read_parquet(filepath)
        elif ext == ".json":
            if verbose > 0: logger.info(f"🧾 Lecture JSON : {filepath}")
            return pd.read_json(filepath, encoding=encoding)
        elif ext in [".xlsx", ".xls"]:
            if verbose > 0: logger.info(f"📊 Lecture Excel : {filepath}")
            return pd.read_excel(filepath)
        elif ext in [".csv", ".txt", ".dat", ".tsv", ""]:
            if verbose > 0: logger.info(f"📑 Lecture texte délimité : {filepath}")
            with open(filepath, "r", encoding=encoding) as f:
                sample_lines = [next(f) for _ in range(min(nrows_detect, 10))]
                sample = "".join(sample_lines)
                try:
                    dialect = csv.Sniffer().sniff(sample, delimiters=",;\t| ")
                    sep = dialect.delimiter
                except Exception:
                    sep = ","
                    if verbose > 0:
                        logger.warning("⚠️ Séparateur non détecté, utilisation de ',' par défaut.")
                has_header = csv.Sniffer().has_header(sample)
                if verbose > 0:
                    logger.info(f"➡️ Séparateur : '{sep}' | Header : {has_header}")
            return pd.read_csv(filepath, sep=sep, header=0 if has_header else None, encoding=encoding)
        else:
            raise ValueError(f"❌ Format de fichier non reconnu : {ext}")

    except Exception as e:
        logger.error(f"❌ Erreur lors de la lecture du fichier {filepath} : {e}")
        raise

# ----------------------------------------------------------------------
# Générateur de données à partir d'un fichier
# ----------------------------------------------------------------------
def file_data_generator(filename: str, dim_x: int, verbose: int = 0) -> Generator[tuple[int, tuple[np.ndarray, np.ndarray]], None, None]:
    df: pd.DataFrame = read_unknown_file(filename, verbose=verbose)
    for k, row in df.iterrows():
        values: np.ndarray = row.values.reshape(-1, 1)
        xkp1, ykp1 = np.split(values, [dim_x])
        yield k, (xkp1, ykp1)

# ----------------------------------------------------------------------
# Vérification cohérence des matrices
# ----------------------------------------------------------------------
def check_consistency(**kwargs: np.ndarray) -> None:
    tol = 1e-12
    if __debug__:
        for name, M in kwargs.items():
            if not np.allclose(M, M.T, atol=tol):
                logger.warning(f"⚠️ {name} matrix is not symmetrical")
            eigvals = np.linalg.eigvals(M)
            if np.any(eigvals < -tol):
                logger.warning(f"⚠️ {name} matrix is not PSD (min eig = {eigvals.min():.3e})")
            logger.debug(f"Eig of {name} matrix: {eigvals}")

# ----------------------------------------------------------------------
# Vérification égalité de matrices
# ----------------------------------------------------------------------
def check_equality(**kwargs: np.ndarray) -> None:
    if len(kwargs) < 2:
        logger.warning("⚠️ _check_equality : need at least 2 matrices.")
        return

    names = list(kwargs.keys())
    matrices = list(kwargs.values())
    shapes = [m.shape for m in matrices]
    if __debug__ and len(set(shapes)) != 1:
        logger.warning(f"⚠️ Matrices are not identically shaped ! {shapes}")
        return

    ref       = matrices[0]
    ref_name  = names[0]
    tol       = 1e-10
    all_equal = True

    for name, M in zip(names[1:], matrices[1:]):
        if __debug__ and not np.allclose(ref, M, atol=tol, rtol=tol):
            diff_norm = np.linalg.norm(ref - M)
            logger.warning(f"⚠️ Matrices '{ref_name}' and '{name}' are different (‖Δ‖={diff_norm:.3e})")
            all_equal = False

    if not all_equal:
        logger.warning("⚠️ Some matrices are different — see messages above.")
        if __debug__:
            input("Waiting for you!")
