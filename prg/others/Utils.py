from __future__ import annotations

import os
import logging
import numpy as np
import pandas as pd
import csv
import chardet
from typing import Generator, Optional, Any, Union
from pathlib import Path

# ----------------------------------------------------------------------
# Logging global
# ----------------------------------------------------------------------
logging.basicConfig(format="[%(levelname)s] %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)

def save_dataframe_to_csv(df, filepath, index=False):
    """Save a DataFrame in CSV (UTF-8) without index."""
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        df.to_csv(path, encoding="utf-8", index=index, float_format="%.15f")
        # if __debug__:
        #     logger.info(f"✅ File save with success at : {path.resolve()}")
    except Exception as e:
        logger.error(f"❌ Something went wrong during saving of CSV : {e}")
        raise

def data_to_dataframe(listData, dim_x, dim_y, withoutX=False):
    """Convert a list of tuples PKF/UKF into pandas DataFrame."""
    data = []
    for idx, (x, y) in [(i, vals) for i, vals in listData]:

        # Validation des types
        if __debug__:
            if not hasattr(x, "flatten") or not hasattr(y, "flatten"):
                raise TypeError(f"The elts for {idx} are not valid numpy.array.")
        x_values = x.flatten()
        y_values = y.flatten()
        if __debug__:
            if len(x_values) != dim_x or len(y_values) != dim_y:
                raise ValueError(f"Unexpected size for vectors at index {idx}: X={len(x_values)}, Y={len(y_values)}")
        if withoutX == True:
            data.append([*y_values])
        else:
            data.append([*x_values, *y_values])

    # dataframe
    columns = []
    if withoutX == False:
        for c in range(dim_x):
            columns.append(f"X{c}")
    for c in range(dim_y):
        columns.append(f"Y{c}")
    df = pd.DataFrame(data, columns=columns)
    # df.set_index("index", inplace=True)

    return df

# ----------------------------------------------------------------------
# MSE
# ----------------------------------------------------------------------
# def mse(x_true, x_hat) -> float:
#     x_true_arr = np.asarray(x_true).ravel()
#     x_hat_arr = np.asarray(x_hat).ravel()
    
#     if __debug__:
#         if x_true_arr.shape != x_hat_arr.shape:
#             raise ValueError(f"❌ Arrays must have the same shape : {x_true_arr.shape} vs {x_hat_arr.shape}")
#     mse = np.mean((x_true_arr - x_hat_arr) ** 2)
#     return float(np.sqrt(mse))


# ------------------------------------------------------------------

def compute_errors(x_true, x_hat, P_list=None):
    """
    Calcul MSE, MAE, RMSE et NEES moyen entre deux séquences d'états.
    
    x_true, x_hat : listes ou colonnes pandas où chaque élément est un array (n,1) ou (n,)
    P_list : liste ou ndarray de matrices covariance (n,n) pour chaque instant (facultatif)
    
    Retour : tuple (mse_total, list_mses_per_dim, mae, rmse, nees_mean) 
    """

    # Conversion en tableaux 
    x_true  = np.hstack(x_true).T      # on empile horizontalement puis on transpose
    x_hat   = np.hstack(x_hat).T       # on empile horizontalement puis on transpose
    tab_cov = np.stack(P_list, axis=0) # empile le long du premier axe

    # Calcul de la MSE par colonne
    errors  = x_true - x_hat  # forme (1000, 2)
    list_mses_per_dim = np.mean(errors**2, axis=0)
    # print(errors)
    # print('max=', np.max(errors[:, 0]))
    # print('arg max=', np.argmax(errors[:, 0]))
    # print("MSE par colonne :", mse_col)
    # exit(1)

    # concaténer pour calcul global
    x_true_flat       = np.concatenate(x_true)
    x_hat_flat        = np.concatenate(x_hat)
    errors_flat       = x_true_flat - x_hat_flat
    mse_total         = float(np.mean(errors_flat**2))
    mae               = float(np.mean(np.abs(errors_flat)))
    rmse              = float(np.sqrt(mse_total))

    # NEES moyen (si P_list fourni)
    
    nees_all = np.zeros(errors.shape[0])
    for k in range(errors.shape[0]):
        ek = errors[k].reshape(-1, 1)   # colonne (2,1)
        Pk = tab_cov[k]                 # (2,2)
        try:
            Pk_inv = np.linalg.inv(Pk)
            nees_all[k] = float(ek.T @ Pk_inv @ ek)# scalaire
        except np.linalg.LinAlgError:
            # P singulière : on ignore
            continue
    nees_mean = np.mean(nees_all)

    return mse_total, list_mses_per_dim, mae, rmse, nees_mean


    
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
            logger.info(f"🧬 Detected encoding : {encoding} (confidence={confidence:.2f})")

        # --- Lecture selon le type ---
        if ext == ".parquet":
            return pd.read_parquet(filepath)
        elif ext == ".json":
            return pd.read_json(filepath, encoding=encoding)
        elif ext in [".xlsx", ".xls"]:
            return pd.read_excel(filepath)
        elif ext in [".csv", ".txt", ".dat", ".tsv", ""]:
            if verbose > 0:
                logger.info(f"📑 Lecture texte délimité : {filepath}")
            
            # Lecture d’un échantillon
            with open(filepath, "r", encoding=encoding) as f:
                sample_lines = [next(f, '') for _ in range(min(nrows_detect, 10))]
                sample = "".join(sample_lines)

            # Tentative de détection du séparateur
            try:
                dialect = csv.Sniffer().sniff(sample, delimiters=",;\t| ")
                sep = dialect.delimiter
                has_header = csv.Sniffer().has_header(sample)
            except csv.Error:
                # Cas typique : une seule colonne sans délimiteur
                sep = None
                has_header = True
                if verbose > 0:
                    logger.warning("⚠️ Impossible de détecter le séparateur — lecture comme fichier à colonne unique.")
            
            if verbose > 0:
                logger.info(f"➡️ Separator : {repr(sep)} | Header : {has_header}")
            
            # Lecture du fichier
            if sep is None:
                # Pas de séparateur détecté → une seule colonne
                df = pd.read_csv(filepath, header=0 if has_header else None, encoding=encoding)
            else:
                df = pd.read_csv(filepath, sep=sep, header=0 if has_header else None, encoding=encoding)
            
            return df
        else:
            raise ValueError(f"❌ Unrecognized file format : {ext}")

    except Exception as e:
        logger.error(f"❌ Something went wrong when reading file {filepath} : {e}")
        raise

def name_analysis(listStr):
    """
    Analyse une liste de noms de variables et renvoie :
      - dim_x : nombre de noms commençant par 'X'
      - dim_y : nombre de noms commençant par 'Y'
      - OK : True si tous les 'X...' sont au début, sinon False
      - autres : liste des chaînes ne commençant ni par 'X' ni par 'Y'
    """
    if not isinstance(listStr, (list, tuple)):
        raise TypeError("L'entrée doit être une liste ou un tuple de chaînes.")
    
    # Comptages de base
    dim_x_true = sum(s.startswith('True') for s in listStr)
    dim_x = sum(s.startswith('X') for s in listStr)
    dim_y = sum(s.startswith('Y') for s in listStr)
    autres = [s for s in listStr if not (s.startswith('X') or s.startswith('Y') or s.startswith('True'))]

    # Vérification de l'ordre (tous les X doivent être avant les Y)
    ok      = True
    x_ended = False
    for s in listStr:
        if s.startswith('X'):
            if x_ended:
                ok = False
                break
        elif s.startswith('Y'):
            x_ended = True

    return {
        "dim_x_true": dim_x_true,
        "dim_x"     : dim_x,
        "dim_y"     : dim_y,
        "Correct"   : ok,
        "autres"    : autres
    }


# ----------------------------------------------------------------------
# Générateur de données à partir d'un fichier
# ----------------------------------------------------------------------

def file_data_generator(filename: str, dim_x: int, dim_y: int, verbose: int = 0) -> Generator[tuple[int, tuple[np.ndarray, np.ndarray]], None, None]:
    df = read_unknown_file(filename, verbose=verbose)
    # print(df.head())
    # print(df.dtypes)
    # print("{:.15f}".format(df["X0"].iloc[0]))
    # exit(1)
    dico = name_analysis(list(df.columns))

    if dico['dim_x'] != 0 and (dico['Correct'] == False  or dico['dim_x'] != dim_x or dico['dim_y']!= dim_y) : 
        print(f'Expected dimensions : dim_x x dim_y = {dim_x} x {dim_y}')
        print(f'Columns found in the file : {list(df.columns)}')
        raise ValueError(f"❌ Pb with dico: {dico}")

    for k, row in df.iterrows():
        values = row.values.reshape(-1, 1)
        if dico['dim_x'] != 0:
            xkp1, ykp1 = np.split(values, [dico['dim_x']])
            yield k, (xkp1, ykp1)
        else:
            ykp1 = values
            yield k, (None, ykp1)

# ----------------------------------------------------------------------
# Vérification cohérence des matrices
# ----------------------------------------------------------------------

def is_covariance(M: np.ndarray, name: str) -> None:
    tol = 1e-12
    if not np.allclose(M, M.T, atol=tol):
        logger.warning(f"⚠️ {name} matrix is not symmetrical")
    eigvals = np.linalg.eigvals(M)
    if np.any(eigvals < -tol):
        logger.warning(f"⚠️ {name} matrix is not positive semi-definite (min eig = {eigvals.min():.3e})")
    logger.debug(f"Eig of {name} matrix: {eigvals}")

def check_consistency(**kwargs: np.ndarray) -> None:
    if __debug__:
        for name, M in kwargs.items():
            is_covariance(M, name)

# ----------------------------------------------------------------------
# Vérification égalité de matrices
# ----------------------------------------------------------------------
def check_equality(**kwargs: np.ndarray) -> None:
    if len(kwargs) < 2:
        logger.warning("⚠️ check_equality : need at least 2 matrices.")
        return

    names = list(kwargs.keys())
    matrices = list(kwargs.values())
    shapes = [m.shape for m in matrices]
    if __debug__ and len(set(shapes)) != 1:
        logger.warning(f"⚠️ Matrices are not identically shaped ! {shapes}")
        return

    ref       = matrices[0]
    ref       = np.asarray(ref, dtype=float)
    ref_name  = names[0]
    tol       = 1e-10
    all_equal = True

    for name, M in zip(names[1:], matrices[1:]):
        M   = np.asarray(M, dtype=float)
        if __debug__ and not np.allclose(ref, M, atol=tol, rtol=tol):
            diff_norm = np.linalg.norm(ref - M)
            logger.warning(f"⚠️ Matrices '{ref_name}' and '{name}' are different (‖Δ‖={diff_norm:.3e})")
            all_equal = False

    if not all_equal:
        logger.warning("⚠️ Some matrices are different — see messages above.")
        if __debug__:
            input("Waiting for you!")
