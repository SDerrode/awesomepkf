#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import pickle
import logging
from typing import Any, Optional

from rich import print

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# pour éviter l'info sur le symbol sigma dans le label de la figure
import logging
logging.getLogger("matplotlib").setLevel(logging.WARNING)

from others.utils import compute_errors
from others.plot_settings import *


# ----------------------------------------------------------------------
# Configuration globale du logging
# ----------------------------------------------------------------------
logging.basicConfig(format="[%(levelname)s] %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)


class HistoryTracker:
    """
    Enregistre et visualise l'évolution de grandeurs au fil des itérations.
    Permet la sauvegarde / rechargement via pickle et la visualisation via Matplotlib.
    """

    def __init__(self, verbose: int = 0):
        assert verbose in [0, 1, 2], "verbose doit être 0, 1 ou 2"
        self._history: list[dict[str, Any]] = []
        self.verbose = verbose
        self._set_log_level()

    # ------------------------------------------------------------------
    def _set_log_level(self):
        """Ajuste le niveau du logger selon la verbosité."""
        if not __debug__:
            logger.setLevel(logging.ERROR)  # Mode rapide = silence
            return
        if self.verbose == 0:
            logger.setLevel(logging.WARNING)
        elif self.verbose == 1:
            logger.setLevel(logging.INFO)
        else:
            logger.setLevel(logging.DEBUG)

    # ------------------------------------------------------------------
    def record(self, **quantities: Any) -> None:
        """Sauvegarde l'état courant sous forme de dictionnaire."""
        assert all(isinstance(k, str) for k in quantities.keys()), "Toutes les clés doivent être des chaînes"
        self._history.append(quantities.copy())

    def as_dataframe(self) -> pd.DataFrame:
        """Retourne l'historique sous forme de DataFrame pandas."""
        return pd.DataFrame(self._history)

    def last(self) -> Optional[dict[str, Any]]:
        """Retourne le dernier enregistrement."""
        return self._history[-1] if self._history else None

    def clear(self) -> None:
        """Efface tout l'historique."""
        self._history.clear()

    # ------------------------------------------------------------------
    def save_pickle(self, path: str) -> None:
        """Sauvegarde l'historique complet dans un fichier .pkl"""
        assert isinstance(path, str), "Le chemin doit être une chaîne"
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self._history, f)
        if self.verbose > 0:
            logger.info(f"[HistoryTracker] Sauvegardé dans '{path}' ({len(self)} enregistrements)")

    @classmethod
    def load_pickle(cls, path: str) -> "HistoryTracker":
        """Recharge un HistoryTracker à partir d'un fichier pickle."""
        assert os.path.exists(path), f"Fichier introuvable : {path}"
        with open(path, "rb") as f:
            data = pickle.load(f)
        assert isinstance(data, list), "Le fichier ne contient pas une liste d'enregistrements"
        tracker = cls()
        tracker._history = data
        if tracker.verbose > 0:
            logger.info(f"[HistoryTracker] Rechargé depuis '{path}' ({len(tracker)} enregistrements)")
        return tracker

    # ------------------------------------------------------------------
    def compute_errors(self, ListeA, ListeB, ListeC, ListeD, ListeE):
        df = self.as_dataframe()
        
        from rich.pretty import Pretty
        from rich.console import Console

        for a, b, c, d, e in zip(ListeA, ListeB, ListeC, ListeD, ListeE):
            report = compute_errors(df[a].to_numpy(), df[b].to_numpy(), df[c].to_numpy(), df[d].to_numpy(), df[e].to_numpy())
            print(f"ERROR ({a}, {b})")
            console = Console()
            console.print(
                Pretty(
                    report,
                    expand_all=True,
                    indent_guides=True
                )
            )

    # ------------------------------------------------------------------
    def plot(self, title, list_param, list_label, list_covar, window, basename="plot", iter_key="iter", show=True, base_dir=None, **kwargs):
        """
        Trace l'évolution des états 
        Si show=False, chaque figure est sauvegardée dans base_dir.
        """

        df = self.as_dataframe().iloc[window['xmin']:window['xmax']]
        assert not df.empty, "Aucune donnée enregistrée."
        for p in list_param:
            assert p in df.columns, f"'{p}' n'est pas une colonne connue : {list(df.columns)}"

        # Vérifie que le premier élément est bien un vecteur
        first = df[list_param[0]].iloc[0]
        assert hasattr(first, "shape"), f"Le premier élément de '{list_param[0]}' n'est pas un vecteur numpy"
        # On récupère le nombre de composantes de X
        nb_components = first.shape[0]
        # print('nb_components=', nb_components)
        
        df_subset     = pd.DataFrame()
        df_subset_var = pd.DataFrame()
        list_labels_p = []
        list_labels_e = []
        for p, e in zip(list_param, list_covar):
            
            list_labels_p_local = []
            list_labels_e_local = []
            for component in range(nb_components):
                list_labels_p_local.append(f'{p}_{component}')
                list_labels_e_local.append(f'{e}_{component}')
            list_labels_p += list_labels_p_local
            list_labels_e += list_labels_e_local

            df_subset[list_labels_p_local] = df[p].apply(lambda x: pd.Series(x.flatten()))
            if e!= None:
                df_subset_var[list_labels_e_local] = df[e].apply(lambda x: pd.Series(x.diagonal()))

        fig, axes = plt.subplots(nb_components, 1, figsize=(7, 2*nb_components), sharex=True, facecolor=facecolor)
        if nb_components == 1: axes = [axes]
        fig.suptitle(title, y=0.85, fontsize=BIGGER_SIZE)

        for i, (col_p, col_e) in enumerate(zip(list_labels_p, list_labels_e)):
            j = i%nb_components
            k = i // nb_components
            # print(f'   toto - i={i}, j={j}, col_p={col_p}, col_e={col_e}, label_p={list_label[k]}')
            df_subset[col_p].plot(ax=axes[j], label=list_label[k], alpha=0.5)
            if not 'None' in col_e:
                # On dessine l'enveloppe
                # print(f'df_subset_var[col_e]={df_subset_var[col_e]}')
                y_upper   = df_subset[col_p] + 2.*np.sqrt(df_subset_var[col_e])
                y_lower   = df_subset[col_p] - 2.*np.sqrt(df_subset_var[col_e])
                last_line = axes[j].lines[-1]       # dernière courbe tracée
                color     = last_line.get_color()
                axes[j].fill_between(df_subset.index, y_lower, y_upper, color=color, alpha=0.2, label=f'{list_label[k]} ± 2*'+r'$𝛔$')
                axes[j].grid(True, linestyle="--", alpha=0.6)
        axes[-1].legend()
        axes[-1].set_xlim(window['xmin'], window['xmax']-1)
        axes[-1].set_xlabel('n')
        # axes[-1].set_xlabel(iter_key)

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
            fig.savefig(save_path, dpi=dpi, bbox_inches="tight", facecolor=facecolor)
            if self.verbose > 0:
                logger.info(f"[HistoryTracker] Graphique sauvegardé : {save_path}")
            plt.close(fig)

    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self._history)

    def __repr__(self) -> str:
        return f"<HistoryTracker n_records={len(self)}>"


# ======================================================================
class A:
    """Classe jouet pour illustrer l'usage de HistoryTracker."""

    def __init__(self, x0: float = 1.0, verbose: int = 1):
        assert isinstance(x0, (int, float)), "x0 doit être un nombre"
        assert verbose in [0, 1, 2], "verbose doit être 0, 1 ou 2"

        self.x = float(x0)
        self.verbose = verbose
        self.history = HistoryTracker(verbose=verbose)

    def iterate_gen(self, n: Optional[int] = None):
        """Générateur qui calcule x_{k+1} = cos(x_k)."""
        assert n is None or (isinstance(n, int) and n >= 0), "n doit être un entier positif ou None"

        k = 0
        while n is None or k < n:
            new_x = np.cos(self.x)
            diff = abs(new_x - self.x)
            record = {"iter": k, "x": self.x, "new_x": new_x, "diff": diff}
            self.history.record(**record)
            if self.verbose > 1:
                logger.debug(f"[A] it={k} x={self.x:.4f} diff={diff:.4e}")
            yield record
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
