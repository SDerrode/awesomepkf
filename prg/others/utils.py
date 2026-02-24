#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
utils.py — Utilitaires pour filtres de Kalman (PKF/UKF/EKF)
"""

from __future__ import annotations

import os
import math
import logging
import csv
import chardet
from pathlib import Path
from typing import Generator, Optional

import numpy as np
import pandas as pd
from dataclasses import is_dataclass, asdict
from rich.table import Table
from rich.console import Console
from rich.text import Text

from others.numerics import EPS_ABS, EPS_REL, COND_FAIL, COND_WARN

# ----------------------------------------------------------------------
# Logging global
# ----------------------------------------------------------------------
logging.basicConfig(format="[%(levelname)s] %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)

console = Console(force_terminal=True, color_system="truecolor")


# ----------------------------------------------------------------------
# Affichage Rich
# ----------------------------------------------------------------------


def rich_show_fields(
    d,
    fields: list | None = None,
    title: str = "Sélection de données",
    decimals: int = 4,
    max_items: int = 10,
) -> None:
    """
    Affiche un dictionnaire ou une dataclass dans un tableau Rich lisible.

    - Floats arrondis à `decimals` chiffres.
    - Booléens NumPy convertis en bool Python.
    - Vecteurs/arrays tronqués si > max_items éléments.
    - Support des dictionnaires et listes imbriqués.
    """
    if is_dataclass(d):
        d = asdict(d)

    if fields is None:
        fields = list(d.keys())

    table = Table(title=title)
    table.add_column("Champ", no_wrap=True)
    table.add_column("Valeur", justify="left")

    def format_value(obj) -> str:
        """Format récursif pour l'affichage scientifique."""
        if isinstance(obj, np.generic):
            obj = obj.item()
        if isinstance(obj, (np.bool_, bool)):
            return str(bool(obj))
        if isinstance(obj, float):
            if math.isinf(obj) or math.isnan(obj):
                return str(obj)
            return f"{obj:.{decimals}f}"
        if isinstance(obj, np.ndarray):
            return format_value(obj.tolist())
        if isinstance(obj, (list, tuple)):
            if len(obj) > max_items:
                displayed = [format_value(v) for v in obj[:max_items]]
                return "[" + ", ".join(displayed) + ", ...]"
            return "[" + ", ".join(format_value(v) for v in obj) + "]"
        if isinstance(obj, dict):
            items = [f"{k}: {format_value(v)}" for k, v in obj.items()]
            return "{ " + ", ".join(items) + " }"
        return str(obj)

    for key in fields:
        if key in d:
            table.add_row(
                Text(key, style="cyan"),
                Text(format_value(d[key]), style="magenta"),
            )

    console.print(table)


# ----------------------------------------------------------------------
# I/O DataFrames
# ----------------------------------------------------------------------


def save_dataframe_to_csv(df: pd.DataFrame, filepath, index: bool = False) -> None:
    """Sauvegarde un DataFrame en CSV UTF-8 sans index."""
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        df.to_csv(path, encoding="utf-8", index=index, float_format="%.15f")
    except Exception as e:
        logger.error(f"❌ Erreur lors de la sauvegarde CSV : {e}")
        raise


def data_to_dataframe(
    listData: list[tuple],
    dim_x: int,
    dim_y: int,
    withoutX: bool = False,
) -> pd.DataFrame:
    """
    Convertit une liste de tuples (idx, x, y) PKF/UKF en DataFrame pandas.

    Parameters
    ----------
    listData : list of (idx, x_array, y_array)
    dim_x    : dimension attendue du vecteur d'état x
    dim_y    : dimension attendue du vecteur d'observation y
    withoutX : si True, n'inclut que les colonnes Y dans le DataFrame
    """
    data = []
    for idx, x, y in listData:
        if __debug__:
            if not hasattr(x, "flatten") or not hasattr(y, "flatten"):
                raise TypeError(f"Éléments non numpy.ndarray à l'index {idx}.")
        x_values = x.flatten()
        y_values = y.flatten()
        if __debug__:
            if len(x_values) != dim_x or len(y_values) != dim_y:
                raise ValueError(
                    f"Tailles inattendues à l'index {idx} : "
                    f"X={len(x_values)} (attendu {dim_x}), "
                    f"Y={len(y_values)} (attendu {dim_y})"
                )
        if withoutX:  # Correction : comparaison pythonique
            data.append([*y_values])
        else:
            data.append([*x_values, *y_values])

    columns = []
    if not withoutX:
        columns += [f"X{c}" for c in range(dim_x)]
    columns += [f"Y{c}" for c in range(dim_y)]

    return pd.DataFrame(data, columns=columns)


# ----------------------------------------------------------------------
# Calcul des formes quadratiques (NEES / NIS)
# ----------------------------------------------------------------------


def _compute_quadratic_form(
    errors: np.ndarray,
    cov_list: list[np.ndarray],
) -> np.ndarray:
    """
    Calcule e_k^T @ Cov_k^{-1} @ e_k pour chaque instant k.

    Utilise np.linalg.solve (plus stable que pinv) avec régularisation
    minimale si la matrice est singulière.

    Returns
    -------
    vals : (N,) — valeurs de la forme quadratique, NaN si calcul impossible
    """
    N = errors.shape[0]
    vals = np.full(N, np.nan)

    for k in range(N):
        ek = errors[k].reshape(-1, 1)  # Assure un vecteur colonne
        Pk = cov_list[k]

        try:
            # Inverse robuste : pseudo-inverse si nécessaire
            Pk_inv = np.linalg.pinv(Pk)  # gère aussi les matrices singulières

            # NEES : ek.T @ Pk_inv @ ek
            vals[k] = float(
                (ek.T @ Pk_inv @ ek).squeeze()
            )  # squeeze + float pour toute dimension

        except np.linalg.LinAlgError:
            # Régularisation minimale sur singularité
            logger.warning(
                f"Matrice singulière à k={k} — régularisation EPS_ABS appliquée."
            )
            try:
                Pk_inv_reg = Pk_inv + EPS_ABS * np.eye(n)
                val = float((ek.T @ Pk_inv_reg @ ek).squeeze())
            except np.linalg.LinAlgError:
                logger.error(f"Impossible d'inverser la matrice à k={k}, NaN inséré.")
                continue

    return vals


def compute_errors(model, x_true, x_hat, P_list, i_list=None, S_list=None):
    """
    Calcul MSE, MAE, RMSE et NEES moyen entre deux séquences d'états.

    x_true, x_hat : listes ou colonnes pandas où chaque élément est un array (n,1) ou (n,)
    P_list : liste ou ndarray de matrices covariance (n,n) pour chaque instant

    Retour : dictionary with several error measures
    """

    # Conversion en tableaux
    x_true = np.hstack(x_true).T  # on empile horizontalement puis on transpose
    x_hat = np.hstack(x_hat).T  # on empile horizontalement puis on transpose
    # concaténer pour calcul global
    x_true_flat = np.concatenate(x_true)
    x_hat_flat = np.concatenate(x_hat)
    errors_flat = x_true_flat - x_hat_flat

    # --- Métriques globales ---
    mse_total = float(np.mean(errors_flat**2))
    mae_total = float(np.mean(np.abs(errors_flat)))
    rmse = float(np.sqrt(mse_total))

    errors = x_true - x_hat

    # NEES moyen
    tab_Pk = np.stack(P_list, axis=0)  # empile le long du premier axe
    nees_all = _compute_quadratic_form(errors, tab_Pk)
    nees_mean = float(np.nanmean(nees_all))

    # NIS moyen
    if i_list is not None:
        nis_all = np.zeros(i_list.shape[0])
        tab_Sk = np.stack(S_list, axis=0)  # empile le long du premier axe
        nis_all = _compute_quadratic_form(i_list, tab_Sk)
        nis_mean = float(np.nanmean(nis_all))
    else:
        nis_mean = "na"

    report = {
        "mse_total": mse_total,
        "mae_total": mae_total,
        "nees_mean": nees_mean,
        "nis_mean": nis_mean,
    }

    # Calcul de la MSE et MAE pour X et pour Y séparemment si c'est une modele a état augmenté
    if model.param.augmented:
        dim_x = model.dim_x
        dim_y = model.dim_y
        list_mses_X_and_Y = [
            np.mean(errors[:, 0 : dim_x - dim_y] ** 2),
            np.mean(errors[:, dim_x - dim_y :] ** 2),
        ]
        list_maes_X_and_Y = [
            np.mean(np.abs(errors[:, 0 : dim_x - dim_y])),
            np.mean(np.abs(errors[:, dim_x - dim_y :])),
        ]
        report["list_mses_X_and_Y"] = list_mses_X_and_Y
        report["list_maes_X_and_Y"] = list_maes_X_and_Y

    return report


# ----------------------------------------------------------------------
# Lecture robuste de fichiers de données
# ----------------------------------------------------------------------


def read_unknown_file(
    filepath: str,
    nrows_detect: int = 500,
    verbose: int = 0,
) -> pd.DataFrame:
    """
    Lit un fichier de données (CSV, TSV, Parquet, JSON, Excel) de manière robuste.

    Détecte automatiquement l'encodage, le séparateur et la présence d'en-tête.
    """
    ext = os.path.splitext(filepath)[1].lower()
    try:
        with open(filepath, "rb") as f:
            raw_data = f.read(50_000)
            enc_info = chardet.detect(raw_data)
            encoding = enc_info["encoding"] or "utf-8"
            confidence = enc_info.get("confidence", 0)
        if verbose > 1:
            logger.info(
                f"🧬 Encodage détecté : {encoding} (confiance={confidence:.2f})"
            )

        if ext == ".parquet":
            return pd.read_parquet(filepath)
        if ext == ".json":
            return pd.read_json(filepath, encoding=encoding)
        if ext in (".xlsx", ".xls"):
            return pd.read_excel(filepath)
        if ext in (".csv", ".txt", ".dat", ".tsv", ""):
            if verbose > 1:
                logger.info(f"📑 Lecture texte délimité : {filepath}")

            with open(filepath, "r", encoding=encoding) as f:
                sample_lines = [next(f, "") for _ in range(min(nrows_detect, 10))]
            sample = "".join(sample_lines)

            try:
                dialect = csv.Sniffer().sniff(sample, delimiters=",;\t| ")
                sep = dialect.delimiter
                has_header = csv.Sniffer().has_header(sample)
            except csv.Error:
                sep = None
                has_header = True
                if verbose > 1:
                    logger.warning(
                        "⚠️ Séparateur non détecté — lecture comme fichier à colonne unique."
                    )

            if verbose > 1:
                logger.info(f"➡️  Séparateur : {repr(sep)} | En-tête : {has_header}")

            header = 0 if has_header else None
            if sep is None:
                return pd.read_csv(filepath, header=header, encoding=encoding)
            return pd.read_csv(filepath, sep=sep, header=header, encoding=encoding)

        raise ValueError(f"❌ Format de fichier non reconnu : {ext}")

    except Exception as e:
        logger.error(f"❌ Erreur lors de la lecture de {filepath} : {e}")
        raise


def name_analysis(listStr: list[str]) -> dict:
    """
    Analyse une liste de noms de colonnes et retourne :
      - dim_x_true : colonnes commençant par 'True'
      - dim_x      : colonnes commençant par 'X'
      - dim_y      : colonnes commençant par 'Y'
      - Correct    : True si tous les 'X' précèdent tous les 'Y'
      - autres     : colonnes ne commençant ni par X, Y ni True
    """
    if not isinstance(listStr, (list, tuple)):
        raise TypeError("L'entrée doit être une liste ou un tuple de chaînes.")

    dim_x_true = sum(s.startswith("True") for s in listStr)
    dim_x = sum(s.startswith("X") for s in listStr)
    dim_y = sum(s.startswith("Y") for s in listStr)
    autres = [
        s
        for s in listStr
        if not (s.startswith("X") or s.startswith("Y") or s.startswith("True"))
    ]

    ok = True
    x_ended = False
    for s in listStr:
        if s.startswith("X"):
            if x_ended:
                ok = False
                break
        elif s.startswith("Y"):
            x_ended = True

    return {
        "dim_x_true": dim_x_true,
        "dim_x": dim_x,
        "dim_y": dim_y,
        "Correct": ok,
        "autres": autres,
    }


# ----------------------------------------------------------------------
# Générateur de données à partir d'un fichier
# ----------------------------------------------------------------------


def file_data_generator(
    filename: str,
    dim_x: int,
    dim_y: int,
    verbose: int = 0,
) -> Generator[tuple[int, np.ndarray | None, np.ndarray], None, None]:
    """
    Générateur qui lit un fichier et produit des tuples (k, x, y).

    Si le fichier ne contient que des colonnes Y (pas de colonnes X),
    x est None à chaque itération.

    Parameters
    ----------
    filename : chemin du fichier de données
    dim_x    : dimension attendue du vecteur d'état
    dim_y    : dimension attendue du vecteur d'observation
    verbose  : niveau de verbosité (0=silencieux, 2=détaillé)
    """
    df = read_unknown_file(filename, verbose=verbose)
    dico = name_analysis(list(df.columns))
    has_x_columns = dico["dim_x"] != 0

    if has_x_columns:
        if not dico["Correct"]:
            raise ValueError(
                f"❌ Les colonnes X et Y ne sont pas dans le bon ordre.\n"
                f"   Colonnes trouvées : {list(df.columns)}"
            )
        if dico["dim_x"] != dim_x:
            raise ValueError(
                f"❌ Dimension X incorrecte : attendu {dim_x}, trouvé {dico['dim_x']}.\n"
                f"   Colonnes : {list(df.columns)}"
            )
        if dico["dim_y"] != dim_y:
            raise ValueError(
                f"❌ Dimension Y incorrecte : attendu {dim_y}, trouvé {dico['dim_y']}.\n"
                f"   Colonnes : {list(df.columns)}"
            )

    for k, row in df.iterrows():
        values = row.values.reshape(-1, 1)
        if has_x_columns:
            xkp1, ykp1 = np.split(values, [dico["dim_x"]])
            yield k, xkp1, ykp1
        else:
            yield k, None, values


# ----------------------------------------------------------------------
# Vérification de cohérence des matrices de covariance
# ----------------------------------------------------------------------


def diagnose_covariance(
    P: np.ndarray,
    cond_warn: float = COND_WARN,
    cond_fail: float = COND_FAIL,
    eig_tol: float = EPS_ABS,
    symmetry_tol: float = EPS_ABS,
) -> tuple[bool, dict]:
    """
    Diagnostic numérique complet d'une matrice de covariance.

    Vérifie : symétrie, valeurs propres (PSD), nombre de condition,
    et décomposition de Cholesky.

    Returns
    -------
    verdict : bool — True si la matrice est numériquement saine
    report  : dict — indicateurs détaillés
    """
    P = np.asarray(P, dtype=float)
    n = P.shape[0]

    report: dict = {}
    verdict = True

    # 1) Symétrie
    sym_err = float(np.linalg.norm(P - P.T, ord="fro"))
    report["symmetry_error"] = sym_err
    report["is_symmetric"] = sym_err < symmetry_tol
    verdict &= report["is_symmetric"]

    # Force la symétrie pour la suite (évite les artefacts numériques)
    P_sym = 0.5 * (P + P.T)

    # 2) Valeurs propres (eigvalsh exploite la symétrie → plus stable)
    eigvals = np.linalg.eigvalsh(P_sym)
    lam_min = float(eigvals.min())
    lam_max = float(eigvals.max())

    report["eigenvalues"] = eigvals
    report["lambda_min"] = lam_min
    report["lambda_max"] = lam_max
    report["is_psd"] = lam_min >= -eig_tol
    verdict &= report["is_psd"]

    # 3) Nombre de condition (rapport max/min valeur propre)
    if lam_min > eig_tol:
        cond = lam_max / lam_min
    else:
        cond = np.inf
    report["condition_number"] = cond
    report["ill_conditioned"] = cond > cond_warn
    report["numerically_singular"] = cond > cond_fail
    verdict &= not report["numerically_singular"]

    # 4) Test de Cholesky — critère pratique le plus fiable
    try:
        np.linalg.cholesky(P_sym)
        report["cholesky_ok"] = True
    except np.linalg.LinAlgError:
        report["cholesky_ok"] = False
        verdict = False  # non &= car peut être déjà False

    # 5) Résidu d'inversion (uniquement si inversible)
    if not report["numerically_singular"]:
        try:
            I_approx = P_sym @ np.linalg.inv(P_sym)
            inv_residual = float(np.linalg.norm(I_approx - np.eye(n), ord="fro"))
        except np.linalg.LinAlgError:
            inv_residual = np.inf
            verdict = False
        report["inverse_residual"] = inv_residual
    else:
        report["inverse_residual"] = np.inf

    return verdict, report


def is_covariance(M: np.ndarray, name: str) -> None:
    """
    Vérifie qu'une matrice est une covariance valide (symétrique, PSD).
    Délègue à diagnose_covariance pour la cohérence des diagnostics.
    """
    ok, report = diagnose_covariance(M)
    if not report["is_symmetric"]:
        logger.warning(
            f"⚠️ Matrice '{name}' non symétrique "
            f"(erreur Frobenius={report['symmetry_error']:.3e})"
        )
    if not report["is_psd"]:
        logger.warning(
            f"⚠️ Matrice '{name}' non semi-définie positive "
            f"(λ_min={report['lambda_min']:.3e})"
        )
    if report.get("ill_conditioned"):
        logger.warning(
            f"⚠️ Matrice '{name}' mal conditionnée "
            f"(cond={report['condition_number']:.3e})"
        )
    logger.debug(f"Valeurs propres de '{name}' : {report['eigenvalues']}")


def check_consistency(**kwargs: np.ndarray) -> None:
    """Vérifie que toutes les matrices passées sont des covariances valides."""
    for name, M in kwargs.items():
        is_covariance(M, name)


def random_covariance(rng, dim_x, dim_y):
    """
    Génère aléatoirement une matrice de covariance (dim_x+dim_y) × (dim_x+dim_y)
    structurée par blocs, avec la propriété :
        Σ11 - Σ12 Σ22⁻¹ Σ12ᵀ est SPD.

    Σ = [[Σ11, Σ12],
        [Σ12ᵀ, Σ22]]
    """
    # --- Bloc bas-droite (Sigma22) SPD et inversible ---
    A2 = rng.standard_normal((dim_y, dim_y))
    Sigma22 = A2 @ A2.T + 1e-3 * np.eye(dim_y)

    # --- Bloc croisé ---
    Sigma12 = rng.standard_normal((dim_x, dim_y))

    # --- Choisir S librement SPD ---
    A1 = rng.standard_normal((dim_x, dim_x))
    S = A1 @ A1.T + 1e-3 * np.eye(dim_x)

    # --- Calcul de Sigma11 pour assurer la condition de Schur ---
    Sigma11 = S + Sigma12 @ np.linalg.inv(Sigma22) @ Sigma12.T

    # --- Assemblage ---
    top = np.hstack((Sigma11, Sigma12))
    bottom = np.hstack((Sigma12.T, Sigma22))
    Sigma = np.vstack((top, bottom))

    return Sigma


# ----------------------------------------------------------------------
# Vérification d'égalité de matrices
# ----------------------------------------------------------------------


def check_equality(**kwargs: np.ndarray) -> None:
    """
    Vérifie que toutes les matrices passées sont numériquement égales.
    La première matrice sert de référence.
    """
    if len(kwargs) < 2:
        logger.warning("⚠️ check_equality : au moins 2 matrices requises.")
        return

    names = list(kwargs.keys())
    matrices = [np.asarray(m, dtype=float) for m in kwargs.values()]
    shapes = [m.shape for m in matrices]

    if len(set(shapes)) != 1:
        logger.warning(f"⚠️ Matrices de formes différentes : {dict(zip(names, shapes))}")
        return

    ref, ref_name = matrices[0], names[0]
    all_equal = True

    for name, M in zip(names[1:], matrices[1:]):
        if not np.allclose(ref, M, atol=EPS_ABS, rtol=EPS_REL):
            diff_norm = float(np.linalg.norm(ref - M))
            logger.warning(
                f"⚠️ Matrices '{ref_name}' et '{name}' différentes "
                f"(‖Δ‖={diff_norm:.3e})"
            )
            all_equal = False

    if not all_equal and __debug__:
        input("Appuyez sur Entrée pour continuer…")
