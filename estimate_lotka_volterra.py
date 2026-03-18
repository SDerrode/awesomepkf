"""
Estimation des paramètres du modèle de Lotka-Volterra (proie-prédateur)
par régression linéaire dans l'espace logarithmique.

═══════════════════════════════════════════════════════════════════════════════
Modèle continu :
    dx/dt = alpha * x - beta  * x * y   (algues : proie)
    dy/dt = delta * x * y - gamma * y   (rotifères : prédateur)

4 paramètres positifs : alpha, beta, gamma, delta
2 variances de bruit  : sigma2_u (log-proie), sigma2_v (log-prédateur)

Point d'équilibre du modèle :  x* = gamma/delta,  y* = alpha/beta
═══════════════════════════════════════════════════════════════════════════════

Méthode — régression linéaire dans l'espace logarithmique
──────────────────────────────────────────────────────────
En posant u = log(x) et v = log(y), les équations deviennent :

    du/dt = alpha - beta * y     (linéaire en alpha, beta pour y connu)
    dv/dt = delta * x - gamma    (linéaire en delta, gamma pour x connu)

Pipeline pour chaque fichier CSV :
  1. Lecture des 3 premières colonnes (temps, proie, prédateur), suppression NaN.
  2. Transformation logarithmique u = log(x), v = log(y).
  3. Ajustement de splines de lissage sur u(t) et v(t) → dérivées analytiques
     du/dt et dv/dt (bien meilleures que les différences finies bruyantes).
  4. Moindres carrés sous contraintes de positivité :
         [1, -y_lissé] · [alpha, beta ]ᵀ ≈ du/dt
         [x_lissé, -1] · [delta, gamma]ᵀ ≈ dv/dt
  5. Variances de bruit = variance des résidus (espace log).
  6. Estimation indépendante par fichier + agrégation statistique.

Avantages de l'espace log :
  • La matrice de régression est linéaire en x ou y seulement (pas x·y),
    ce qui réduit la corrélation entre colonnes.
  • Le bruit multiplicatif (biologique) est additivé en log.
  • Les paramètres α et β (ou δ et γ) sont estimés en équations séparées :
    pas de couplage entre les deux paires.

Usage :
    python estimate_lotka_volterra.py [--data_dir PATH] [--smooth FLOAT] [--output FILE]

Options :
    --data_dir  Répertoire contenant les fichiers CSV  (défaut : dossier du script)
    --smooth    Facteur de lissage de la spline [0.1 – 5.0]  (défaut : 0.3)
                Plus grand → spline plus lisse → dérivées moins bruitées.
                Essayer 0.1 (faible bruit) à 1.0 (fort bruit).
    --output    Fichier CSV de sortie (optionnel)
"""

import argparse
import re
import warnings
from pathlib import Path

# Seuls les fichiers dont le stem est exactement [Cc] suivi de chiffres (ex. C1, C12)
_DATA_FILE_RE = re.compile(r'^[Cc]\d+$')

import numpy as np
import pandas as pd
from scipy.interpolate import UnivariateSpline
from scipy.optimize import lsq_linear


# ─────────────────────────────────────────────────────────────────────────────
# 1. Chargement des données
# ─────────────────────────────────────────────────────────────────────────────

def load_csv(filepath: Path) -> pd.DataFrame:
    """
    Charge un fichier CSV, conserve les 3 premières colonnes
    (temps, proie, prédateur), supprime les lignes avec NaN ou valeurs ≤ 0,
    et trie par ordre chronologique.
    """
    df = pd.read_csv(filepath, header=0)
    df = df.iloc[:, :3].copy()
    df.columns = ["time", "prey", "predator"]
    df = df.dropna().reset_index(drop=True)
    df = df[(df["prey"] > 0) & (df["predator"] > 0)].reset_index(drop=True)
    df = df.sort_values("time").reset_index(drop=True)
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 2. Sauvegarde du CSV nettoyé
# ─────────────────────────────────────────────────────────────────────────────

def save_cleaned_csv(df: pd.DataFrame, filepath: Path) -> tuple[Path, Path]:
    """
    Sauvegarde deux versions nettoyées du fichier CSV dans le même répertoire :

      *_clean.csv     — 3 colonnes : t (temps), X0 (proie), Y0 (prédateur)
      *_clean_xy.csv  — 2 colonnes : X0 (proie), Y0 (prédateur)
    """
    prey_pred = df[["prey", "predator"]].rename(columns={"prey": "X0", "predator": "Y0"})

    # Format 3 colonnes : t, X0, Y0
    txyz = df[["time"]].rename(columns={"time": "t"}).join(prey_pred)
    path_3col = filepath.with_name(filepath.stem + "_clean.csv")
    txyz.to_csv(path_3col, index=False)

    # Format 2 colonnes : X0, Y0
    path_2col = filepath.with_name(filepath.stem + "_clean_xy.csv")
    prey_pred.to_csv(path_2col, index=False)

    return path_3col, path_2col


# ─────────────────────────────────────────────────────────────────────────────
# 4. Lissage par spline et dérivation analytique
# ─────────────────────────────────────────────────────────────────────────────

def spline_smooth_and_derive(
    t: np.ndarray,
    z: np.ndarray,
    smooth_factor: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Ajuste une spline de lissage quintique (k=5) sur (t, z) et retourne :
      - z_smooth : valeurs lissées aux points t  (= exp pour revenir à x ou y)
      - dzdt     : dérivée dz/dt aux points t

    Le paramètre de lissage s = smooth_factor * n * Var(z).
    """
    n = len(t)
    s = smooth_factor * n * float(np.var(z))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        spl = UnivariateSpline(t, z, s=s, k=5)
    return spl(t), spl.derivative()(t)


# ─────────────────────────────────────────────────────────────────────────────
# 5. Estimation des paramètres sur un fichier
# ─────────────────────────────────────────────────────────────────────────────

def estimate_one_file(
    df: pd.DataFrame,
    smooth_factor: float = 0.3,
) -> dict:
    """
    Estime les 4 paramètres de Lotka-Volterra et les 2 variances de bruit
    (en espace log) sur un unique jeu de données.

    Formulation log-espace :
        du/dt = alpha - beta * y   →  [1, -y] · [alpha, beta]ᵀ
        dv/dt = delta * x - gamma  →  [x, -1] · [delta, gamma]ᵀ

    Régression sous contrainte de positivité (borne inférieure = 0).

    Returns
    -------
    dict : alpha, beta, gamma, delta, sigma2_u, sigma2_v,
           x_eq, y_eq (point d'équilibre), n_points.
    """
    t = df["time"].values.astype(float)
    x = df["prey"].values.astype(float)
    y = df["predator"].values.astype(float)
    n = len(t)

    # ── Log des variables (bruit multiplicatif → additif)
    u = np.log(x)
    v = np.log(y)

    # ── Lissage par spline et dérivées analytiques
    u_smooth, dudt = spline_smooth_and_derive(t, u, smooth_factor)
    v_smooth, dvdt = spline_smooth_and_derive(t, v, smooth_factor)

    # Retour en espace original pour les régresseurs
    x_smooth = np.exp(u_smooth)
    y_smooth = np.exp(v_smooth)

    # ── Matrices de régression (une par équation)
    # du/dt = alpha - beta * y   →  [1, -y_smooth]
    A_prey = np.column_stack([np.ones(n), -y_smooth])
    # dv/dt = delta * x - gamma  →  [x_smooth, -1]
    A_pred = np.column_stack([x_smooth, -np.ones(n)])

    # ── Moindres carrés avec contrainte de positivité (borne 0 ≤ p < ∞)
    res_prey = lsq_linear(A_prey, dudt, bounds=(0.0, np.inf))
    res_pred = lsq_linear(A_pred, dvdt, bounds=(0.0, np.inf))

    alpha, beta  = res_prey.x
    delta, gamma = res_pred.x

    # ── Variances de bruit (résidus en espace log)
    resid_u = dudt - A_prey @ res_prey.x
    resid_v = dvdt - A_pred @ res_pred.x
    sigma2_u = float(np.var(resid_u))
    sigma2_v = float(np.var(resid_v))

    # ── Point d'équilibre théorique
    x_eq = float(gamma / delta) if delta > 1e-12 else float("nan")
    y_eq = float(alpha / beta)  if beta  > 1e-12 else float("nan")

    return {
        "alpha":    float(alpha),
        "beta":     float(beta),
        "gamma":    float(gamma),
        "delta":    float(delta),
        "sigma2_u": sigma2_u,
        "sigma2_v": sigma2_v,
        "x_eq":     x_eq,
        "y_eq":     y_eq,
        "n_points": n,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 6. Pipeline complet sur tous les fichiers CSV
# ─────────────────────────────────────────────────────────────────────────────

PARAM_NAMES = ["alpha", "beta", "gamma", "delta", "sigma2_u", "sigma2_v"]


def run_pipeline(
    data_dir: Path,
    smooth_factor: float = 0.3,
) -> pd.DataFrame:
    """
    Applique l'estimation sur tous les fichiers CSV du répertoire `data_dir`.

    Returns
    -------
    DataFrame indexé par nom de fichier avec les paramètres estimés.
    """
    csv_files = sorted(f for f in data_dir.glob("*.csv") if _DATA_FILE_RE.match(f.stem))
    if not csv_files:
        raise FileNotFoundError(f"Aucun fichier CSV trouvé dans {data_dir}")

    records = []
    for fpath in csv_files:
        try:
            df = load_csv(fpath)
            if len(df) < 10:
                print(f"  [SKIP] {fpath.name} : seulement {len(df)} points valides")
                continue

            path_3col, path_2col = save_cleaned_csv(df, fpath)
            print(f"  → {path_3col.name}, {path_2col.name} sauvegardés ({len(df)} lignes)")

            result = estimate_one_file(df, smooth_factor=smooth_factor)
            result["file"] = fpath.name
            records.append(result)

            x_eq_str = f"{result['x_eq']:.3f}" if not np.isnan(result["x_eq"]) else "  NaN "
            y_eq_str = f"{result['y_eq']:.2f}" if not np.isnan(result["y_eq"]) else "  NaN "
            print(
                f"  {fpath.name:12s}  n={result['n_points']:3d}"
                f"  α={result['alpha']:7.4f}  β={result['beta']:8.5f}"
                f"  γ={result['gamma']:7.4f}  δ={result['delta']:7.4f}"
                f"  σ²u={result['sigma2_u']:.4f}  σ²v={result['sigma2_v']:.4f}"
                f"  x*={x_eq_str}  y*={y_eq_str}"
            )
        except Exception as exc:
            print(f"  [ERROR] {fpath.name} : {exc}")

    if not records:
        raise RuntimeError("Aucun fichier traité avec succès.")

    return pd.DataFrame(records).set_index("file")


def print_summary(results: pd.DataFrame, data_dir: Path) -> None:
    """Affiche les statistiques agrégées et une validation sur les équilibres."""
    print("\n" + "═" * 72)
    print("RÉSUMÉ AGRÉGÉ  (moyenne ± écart-type sur tous les fichiers)")
    print("═" * 72)

    for p in PARAM_NAMES:
        if p not in results.columns:
            continue
        vals = results[p].dropna()
        print(
            f"  {p:10s} : {vals.mean():10.5f}  ±  {vals.std():8.5f}"
            f"  (médiane={vals.median():.5f})"
        )

    print("\n" + "─" * 72)
    print("VALIDATION — Point d'équilibre théorique vs. moyenne empirique des données")
    print("─" * 72)
    print(f"  {'fichier':12s}  {'x* théo':>8}  {'x̄ réel':>8}  {'y* théo':>8}  {'ȳ réel':>8}")
    for fpath in sorted(f for f in data_dir.glob("*.csv") if _DATA_FILE_RE.match(f.stem)):
        if fpath.name not in results.index:
            continue
        row = results.loc[fpath.name]
        df = load_csv(fpath)
        x_mean = df["prey"].mean()
        y_mean = df["predator"].mean()
        x_eq = row["x_eq"] if not np.isnan(row["x_eq"]) else float("nan")
        y_eq = row["y_eq"] if not np.isnan(row["y_eq"]) else float("nan")
        print(
            f"  {fpath.name:12s}  {x_eq:8.3f}  {x_mean:8.3f}  {y_eq:8.2f}  {y_mean:8.2f}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# 7. Point d'entrée
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Estimation des paramètres de Lotka-Volterra — régression en espace log"
    )
    parser.add_argument(
        "--data_dir",
        type=Path,
        default=Path(__file__).parent,
        help="Répertoire contenant les fichiers CSV (défaut : même dossier que le script)",
    )
    parser.add_argument(
        "--smooth",
        type=float,
        default=0.3,
        help=(
            "Facteur de lissage de la spline (défaut 0.3). "
            "Plus élevé → spline plus lisse → dérivées moins bruitées. "
            "Plage recommandée : 0.1 à 1.0."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Fichier CSV de sortie pour les paramètres estimés (optionnel)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print(f"\nRépertoire de données : {args.data_dir.resolve()}")
    print(f"Facteur de lissage   : {args.smooth}")
    print(
        "\nColonnes d'entrée : temps (jours) | proie / algues (10⁶ cell/ml)"
        " | prédateur / rotifères (anim/ml)"
    )
    print("─" * 72)

    results = run_pipeline(args.data_dir, smooth_factor=args.smooth)

    print_summary(results, args.data_dir)

    print("\n" + "─" * 72)
    print("Tableau complet des estimations :")
    print(results[PARAM_NAMES + ["x_eq", "y_eq"]].to_string(float_format="{:.5f}".format))

    if args.output is not None:
        results.to_csv(args.output)
        print(f"\nRésultats sauvegardés dans : {args.output}")

    print()


if __name__ == "__main__":
    main()
