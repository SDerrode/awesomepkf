#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import pickle
import logging
from typing import Any, Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

from dataclasses import is_dataclass, asdict

from rich.table import Table
from rich.console import Console

from others.utils import compute_errors
from others.plot_settings import *
# A few utils functions that are used several times
from others.utils import rich_show_fields

from others.numerics import EPS_ABS, EPS_REL

# Arrondir l'affichage à 4 chiffres après la virgule
# np.set_printoptions(precision=4, suppress=True)

# ----------------------------------------------------------------------
# Configuration globale du logging
# ----------------------------------------------------------------------
logging.basicConfig(format="[%(levelname)s] %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)


class HistoryTracker:
    """
    Enregistre et visualise l'évolution de grandeurs au fil des itérations.

    Cette classe est utile pour suivre des variables dans des simulations, filtres
    (Kalman, particulaire, etc.) ou tout algorithme itératif. Elle permet de :

    - Enregistrer des grandeurs à chaque itération via `record()`.
    - Transformer l'historique en pandas DataFrame pour analyse.
    - Calculer et afficher des erreurs via `compute_errors()`.
    - Tracer les variables avec covariances et enveloppes ±2σ via `plot()`.
    - Sauvegarder/recharger l'historique via pickle.

    Attributs
    ----------
    _history : list[dict[str, Any]]
        Liste des enregistrements effectués via `record()`.
    verbose : int
        Niveau de verbosité :
        0 = avertissements uniquement
        1 = informations principales
        2 = debug détaillé
    """

    def __init__(self, verbose: int = 0):
        """
        Initialise un HistoryTracker vide.

        Parameters
        ----------
        verbose : int, optional
            Niveau de verbosité (0, 1, 2). Par défaut 0.
        """
        if verbose not in (0, 1, 2):
            raise ValueError("verbose doit être 0, 1 ou 2")
        self._history: list[dict[str, Any]] = []
        self.verbose = verbose
        self._set_log_level()

    # ------------------------------------------------------------------
    def _set_log_level(self):
        """Ajuste le niveau du logger selon la verbosité."""
        if not __debug__:
            logger.setLevel(logging.ERROR)
            return
        if self.verbose==0 or self.verbose==1:
            logger.setLevel(logging.CRITICAL + 1)
        elif self.verbose == 2:
            logger.setLevel(logging.INFO)
        else:
            logger.setLevel(logging.DEBUG)

    def record(self, *args, **kwargs) -> None:
        """
        Enregistre l'état courant.

        - Si on passe une dataclass (PKFStep), elle est convertie en dict.
        - Sinon, on accepte **kwargs comme avant.
        """
        if len(args) == 1 and is_dataclass(args[0]):
            self._history.append(asdict(args[0]))
        else:
            if not all(isinstance(k, str) for k in kwargs):
                raise TypeError("Toutes les clés doivent être des chaînes")
            self._history.append(kwargs.copy())

    def as_dataframe(self) -> pd.DataFrame:
        """
        Retourne l'historique complet sous forme de DataFrame pandas.

        Returns
        -------
        pd.DataFrame
            DataFrame avec les enregistrements.
        """
        return pd.DataFrame(self._history)

    def last(self) -> Optional[dict[str, Any]]:
        """
        Retourne le dernier enregistrement.

        Returns
        -------
        dict[str, Any] or None
            Le dernier dictionnaire enregistré, ou None si l'historique est vide.
        """
        return self._history[-1] if self._history else None

    def clear(self) -> None:
        """Efface tout l'historique."""
        self._history.clear()

    # ------------------------------------------------------------------
    def save_pickle(self, path: str) -> None:
        """
        Sauvegarde l'historique complet dans un fichier pickle (.pkl).

        Parameters
        ----------
        path : str
            Chemin du fichier de sauvegarde.
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self._history, f)
        if self.verbose > 0:
            logger.info(f"[HistoryTracker] Sauvegardé dans '{path}' ({len(self)} enregistrements)")


    @classmethod
    def load_pickle(cls, path: str) -> "HistoryTracker":
        """
        Recharge un HistoryTracker à partir d'un fichier pickle.

        Parameters
        ----------
        path : str
            Chemin du fichier pickle.

        Returns
        -------
        HistoryTracker
            Un objet HistoryTracker contenant l'historique rechargé.
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Fichier introuvable : {path}")
        with open(path, "rb") as f:
            data = pickle.load(f)
        if not isinstance(data, list):
            raise TypeError("Le fichier ne contient pas une liste d'enregistrements")
        tracker = cls()
        tracker._history = data
        if tracker.verbose > 0:
            logger.info(f"[HistoryTracker] Rechargé depuis '{path}' ({len(tracker)} enregistrements)")
        return tracker

    # ------------------------------------------------------------------
    def compute_errors(self, model, ListeA, ListeB, ListeC, ListeD=None, ListeE=None):
        """
        Calcule et affiche des rapports d'erreurs entre différentes séries de données.

        Utilise la fonction `compute_errors` (externe) pour calculer les erreurs
        et rich.Console pour un affichage lisible.

        Parameters
        ----------
        ListeA, ListeB, ListeC : list[str]
            Noms des colonnes à comparer.
        ListeD, ListeE : list[str] or None
            Colonnes supplémentaires pour certains filtres (ex : particulaire).
        """
        
        df = self.as_dataframe()
        # print(f'df.head()={df.head()}')
        from rich.pretty import Pretty
        from rich.console import Console

        console = Console(force_terminal=True, color_system="truecolor")  # console globale partagée
        
        if ListeD is None or ListeE is None:
            for a, b, c in zip(ListeA, ListeB, ListeC):
                report = compute_errors(model, \
                                        df[a].to_numpy(), df[b].to_numpy(), df[c].to_numpy(), \
                                        None, None)
                if self.verbose>0:
                    rich_show_fields(report, ["mse_total", "mae_total", "nees_mean", "nis_mean", 'list_mses_X_and_Y', 'list_maes_X_and_Y'], title=f"{a} vs {b}")
        else:
            for a, b, c, d, e in zip(ListeA, ListeB, ListeC, ListeD, ListeE):
                report = compute_errors(model, \
                                        df[a].to_numpy(), df[b].to_numpy(), df[c].to_numpy(), \
                                        df[d].to_numpy(), df[e].to_numpy())
                if self.verbose>0:
                    rich_show_fields(report, ["mse_total", "mae_total", "nees_mean", "nis_mean", 'list_mses_X_and_Y', 'list_maes_X_and_Y'], title=f"{a} vs {b}")


    def _compute_sigma_envelope(self, var_series: pd.Series, col_name: str) -> np.ndarray:
        """
        Calcule un sigma stable à partir d'une série de variances et détecte les anomalies.

        - Clamp les variances légèrement négatives à 0
        - Signale les variances fortement négatives

        Parameters
        ----------
        var_series : pd.Series
            Série pandas contenant les covariances diagonales.
        col_name : str
            Nom de la variable (pour les messages d'erreur/log).

        Returns
        -------
        np.ndarray
            Tableau numpy contenant σ corrigé.
        """
        v = var_series.values
        scale = np.nanmax(np.abs(v))
        tol = max(EPS_ABS, EPS_REL * scale)

        slightly_negative = (v < 0) & (v >= -tol)
        strongly_negative = v < -tol

        if slightly_negative.any() and self.verbose > 0:
            logger.info(f"[{col_name}] Variances légèrement négatives corrigées : {np.sum(slightly_negative)} points")

        if strongly_negative.any():
            idx = np.where(strongly_negative)[0][:5]
            raise ValueError(
                f"Variance fortement négative détectée dans {col_name} "
                f"(indices exemples {idx}, valeurs {v[idx]})"
            )

        v_corrected = v.copy()
        v_corrected[slightly_negative] = 0.0
        return np.sqrt(v_corrected)


    # ------------------------------------------------------------------
    def plot(self, title, list_param, list_label, list_covar, window, basename="plot", show=True, base_dir=None, **kwargs):
        """
        Trace l'évolution des états avec leurs covariances ±2σ.

        Si `show=False`, la figure est sauvegardée dans `base_dir`.

        Parameters
        ----------
        title : str
            Titre global de la figure.
        list_param : list[str]
            Noms des colonnes à tracer.
        list_label : list[str]
            Labels pour la légende.
        list_covar : list[str or None]
            Colonnes contenant la covariance associée (ou None).
        window : dict
            Fenêtre temporelle à tracer, avec clés 'xmin' et 'xmax'.
        basename : str, optional
            Nom du fichier si sauvegarde (par défaut "plot").
        show : bool, optional
            Affiche la figure si True (défaut True).
        base_dir : str, optional
            Dossier de sauvegarde si show=False.
        **kwargs :
            Arguments supplémentaires pour personnalisation future.

        Returns
        -------
        fig, axes : tuple
            Figure et axes matplotlib.
        """
        
        if not (len(list_param) == len(list_label) == len(list_covar)):
            raise ValueError("list_param, list_label et list_covar doivent avoir la même longueur")
        
        for key in ("xmin", "xmax"):
            if key not in window:
                raise KeyError(f"window doit contenir '{key}'")

        df = self.as_dataframe().iloc[window['xmin']:window['xmax']]
        if df.empty:
            raise ValueError("Aucune donnée enregistrée.")
        for p in list_param:
            if p not in df.columns:
                raise KeyError(f"'{p}' n'est pas une colonne connue : {list(df.columns)}")


        # Vérifie que le premier élément est bien un vecteur
        first = df[list_param[0]].iloc[0]
        if not hasattr(first, "shape"):
            raise TypeError(
                f"Le premier élément de '{list_param[0]}' n'est pas un vecteur numpy"
            )
        # On récupère le nombre de composantes de X
        nb_components = first.shape[0]
        # print('nb_components=', nb_components)
        
        df_subset     = pd.DataFrame()
        df_subset_var = pd.DataFrame()
        list_labels_p = []
        list_labels_e = []
        list_has_var  = []
        for p, e in zip(list_param, list_covar):
            has_var = e is not None
            list_has_var += [has_var] * nb_components
            
            list_labels_p_local = []
            list_labels_e_local = []
            for component in range(nb_components):
                list_labels_p_local.append(f'{p}_{component}')
                list_labels_e_local.append(f'{e}_{component}')
            list_labels_p += list_labels_p_local
            list_labels_e += list_labels_e_local

            df_subset[list_labels_p_local] = df[p].apply(lambda x: pd.Series(x.flatten()))
            if has_var:
                df_subset_var[list_labels_e_local] = df[e].apply(lambda x: pd.Series(x.diagonal()))

        fig, axes = plt.subplots(nb_components, 1, figsize=(7, 2*nb_components), sharex=True, facecolor=FACECOLOR)
        if nb_components == 1: axes = [axes]
        fig.suptitle(title, y=0.85, fontsize=BIG_SIZE)

        for i, (col_p, col_e, has_var) in enumerate(zip(list_labels_p, list_labels_e, list_has_var)):
 
            j = i%nb_components
            k = i // nb_components
            df_subset[col_p].plot(ax=axes[j], label=list_label[k], alpha=0.5)
            
            if has_var and col_e not in df_subset_var:
                raise KeyError(f"Variance '{col_e}' absente de df_subset_var")

            if has_var:
                
                # Detection de variances très légèrement négatives (et celles qui le serainet plus que légèrement)
                sigma = self._compute_sigma_envelope(df_subset_var[col_e], col_e)

                # On dessine l'enveloppe
                y_upper   = df_subset[col_p] + 2.*sigma
                y_lower   = df_subset[col_p] - 2.*sigma
                last_line = axes[j].lines[-1]       # dernière courbe tracée
                color     = last_line.get_color()
                axes[j].fill_between(df_subset.index, y_lower, y_upper, color=color, alpha=0.2, label=f'{list_label[k]} ± 2*' + r'$\sigma$')
                

        for ax in axes:
            ax.grid(True, linestyle="--", alpha=0.6)
        handles, labels = axes[-1].get_legend_handles_labels()
        unique = dict(zip(labels, handles))
        axes[-1].legend(
            unique.values(),
            unique.keys(),
            loc="upper center",
            bbox_to_anchor=(0.5, 1.15),
            ncol=len(unique)
        )
        axes[-1].set_xlim(window['xmin'], window['xmax']-1)
        axes[-1].set_xlabel('n')

        # Nettoyage explicite de l’axe partagé
        axes[-1].minorticks_off()
        axes[-1].xaxis.set_major_locator(mticker.MaxNLocator(nbins=5, integer=True))
        # Optionnel mais très efficace
        fig.canvas.draw_idle()

        if show:
            plt.show()
        else:
            os.makedirs(base_dir or ".", exist_ok=True)
            save_path = os.path.join(base_dir or ".", f"{basename}.png")
            fig.savefig(save_path, dpi=DPI, bbox_inches="tight", facecolor=FACECOLOR)
            if self.verbose > 0:
                logger.info(f"[HistoryTracker] Graphique sauvegardé : {save_path}")
            plt.close(fig)
        
        return fig, axes

    # ------------------------------------------------------------------
    def __len__(self) -> int:
        """Nombre d'enregistrements dans l'historique."""
        return len(self._history)

    def __repr__(self) -> str:
        """Représentation courte de l'objet."""
        return f"<HistoryTracker n_records={len(self)} - address: {hex(id(self))}>"


# ======================================================================

from dataclasses import dataclass

@dataclass
class SimpleStep:
    iter:  int
    x:     float
    new_x: float
    diff:  float
    
class A:
    """Classe jouet pour illustrer l'usage de HistoryTracker."""

    def __init__(self, x0: float = 1.0, verbose: int = 1):
        assert isinstance(x0, (int, float)), "x0 doit être un nombre"
        assert verbose in [0, 1, 2], "verbose doit être 0, 1 ou 2"

        self.x = float(x0)
        self.verbose = verbose
        self.history = HistoryTracker(verbose=verbose)

    def iterate_gen(self, n: Optional[int] = None):
        k = 0
        while n is None or k < n:
            new_x = np.cos(self.x)
            diff = abs(new_x - self.x)
            step = SimpleStep(iter=k, x=self.x, new_x=new_x, diff=diff)
            self.history.record(step)
            if self.verbose > 1:
                logger.debug(f"[A] it={k} x={self.x:.4f} diff={diff:.4e}")
            yield step
            self.x = new_x
            k += 1

    def iterate_list(self, n: int):
        """Retourne la liste complète des itérations."""
        assert isinstance(n, int) and n > 0, "n doit être un entier positif"
        return list(self.iterate_gen(n))


# ======================================================================
if __name__ == "__main__":
    verbose = 1
    graph_dir = os.path.join('.', 'data', 'plot')
    tracker_dir = os.path.join('.', 'data', 'historyTracker')
    os.makedirs(graph_dir, exist_ok=True)
    os.makedirs(tracker_dir, exist_ok=True)

    a = A(x0=1.0, verbose=verbose)
    for step in a.iterate_gen(5):
        print(step)

    a.history.plot(["x"], ["x"], color="blue", show=False, base_dir=graph_dir)
    a.history.save_pickle(os.path.join(tracker_dir, "history_run_a.pkl"))
