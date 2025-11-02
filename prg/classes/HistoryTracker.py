#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import pickle
import numbers
import logging
from typing import Any, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker


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
        
        if verbose not in [0, 1, 2]:
            raise ValueError("verbose must be 0, 1 or 2")
        self._history: list[dict[str, Any]] = []
        self.verbose = verbose
        
        # Configuration du logger selon verbose
        self._set_log_level()

    # ------------------------------------------------------------------
    # Gestion du logging selon le niveau de verbosité
    # ------------------------------------------------------------------
    def _set_log_level(self):
        if self.verbose == 0:
            logger.setLevel(logging.WARNING)
        elif self.verbose == 1:
            logger.setLevel(logging.INFO)
        else:
            logger.setLevel(logging.DEBUG)

    # ------------------------------------------------------------------
    #  Gestion des enregistrements
    # ------------------------------------------------------------------
    def record(self, **quantities: Any) -> None:
        """Sauvegarde l'état courant sous forme de dictionnaire."""
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
    #  Sauvegarde et chargement
    # ------------------------------------------------------------------
    def save_pickle(self, path: str) -> None:
        """Sauvegarde l'historique complet dans un fichier .pkl"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self._history, f)
        if self.verbose>0:
            logger.info(f"[HistoryTracker] Historique sauvegardé dans '{path}' ({len(self)} enregistrements)")

    @classmethod
    def load_pickle(cls, path: str) -> "HistoryTracker":
        """Recharge un HistoryTracker à partir d'un fichier pickle."""
        with open(path, "rb") as f:
            data = pickle.load(f)
        tracker = cls()
        tracker._history = data
        if self.verbose>0:
            logger.info(f"[HistoryTracker] Rechargé depuis '{path}' ({len(tracker)} enregistrements)")
        return tracker

    # ------------------------------------------------------------------
    #  Visualisation
    # ------------------------------------------------------------------
    def plot(self, list_param, list_label, iter_key="iter", show=True, base_dir=None, **kwargs):
        """
        Trace l'évolution d'un paramètre au fil des itérations.
        Si show=False, chaque figure est sauvegardée dans base_dir avec un indice dans le nom.

        Arguments :
        -----------
        list_param : str
            List of parameters to draw on the same plot (scalar or vector)
        list_param : str
            List of labels to appear in the legend
        iter_key : str
            Key for X axis (default: 'iter')
        show : bool
            If True → plot the graphic else save in base_dir
        base_dir : str | None
            Store repository if show=False
        kwargs :
            Parameters to be passed to matplotlib.plot() (color, style, etc.)
        """

        df = pd.DataFrame(self._history.copy())

        if df.empty:
            raise ValueError("Aucune donnée enregistrée.")
        if param not in df.columns:
            raise KeyError(f"'{param}' n'est pas une colonne enregistrée. Colonnes disponibles : {list(df.columns)}")

        # Récupération des données
        y_values = df[param]
        print(y_values)
        input('attente')
        x = df[iter_key] if iter_key in df.columns else df.index

        # Vérifie si les entrées sont des vecteurs
        first_val = y_values.iloc[0]
        is_vector = isinstance(first_val, (list, np.ndarray))

        if not is_vector:
            # --- Cas scalaire -------------------------------------------------
            if not all(isinstance(v, numbers.Number) for v in y_values):
                raise TypeError(f"La colonne '{param}' contient des valeurs non scalaires et non vectorielles.")

            created_fig = False
            if ax is None:
                fig, ax = plt.subplots(figsize=(6, 4))
                created_fig = True

            ax.plot(x, y_values, **kwargs)
            ax.set_xlabel(iter_key)
            ax.set_ylabel(param)
            ax.set_title(f"Évolution de '{param}' ({len(df)} points)")
            ax.grid(True, linestyle="--", alpha=0.6)
            ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))

            if show:
                plt.show()
            else:
                os.makedirs(base_dir or ".", exist_ok=True)
                save_path = os.path.join(base_dir or ".", f"plot_{param}.png")
                ax.figure.savefig(save_path, dpi=150, bbox_inches="tight")
                if self.verbose>0:
                    print(f"[HistoryTracker] Graphique sauvegardé : {save_path}")
                if created_fig:
                    plt.close(fig)
            return ax, fig
        else:
            # --- Cas vectoriel ------------------------------------------------
            n_components = y_values[0].shape[0]

            for i in range(n_components):

                created_fig = False
                if ax is None:
                    fig, ax = plt.subplots(figsize=(6, 4))
                    created_fig = True
                
                yi = y_values.apply(lambda x: x[i])
                ax.plot(x, yi, **kwargs)
                ax.set_xlabel(iter_key)
                ax.set_ylabel(f"{param}[{i}]")
                ax.set_title(f"Évolution de '{param}[{i}]' ({len(df)} points)")
                ax.grid(True, linestyle="--", alpha=0.6)
                ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))

                if show:
                    plt.show()
                else:
                    os.makedirs(base_dir or ".", exist_ok=True)
                    save_path = os.path.join(base_dir or ".", f"plot_{param}_{i}.png")
                    fig.savefig(save_path, dpi=150, bbox_inches="tight")
                    print(f"[HistoryTracker] Graphique sauvegardé : {save_path}")
                    if created_fig:
                        plt.close(fig)
            return ax, fig

    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self._history)

    def __repr__(self) -> str:
        return f"<HistoryTracker n_records={len(self)}>"


class A:
    """Classe jouet pour illustrer l'usage de HistoryTracker."""

    def __init__(self, x0: float = 1.0, verbose: bool = True):
        
        if verbose not in [0, 1, 2]:
            raise ValueError("verbose must be 0, 1 or 2")
        
        self.x       = x0
        self.verbose = verbose
        self.history = HistoryTracker(verbose=verbose)

    def iterate_gen(self, n: Optional[int] = None):
        """Générateur qui calcule x_{k+1} = cos(x_k)."""
        k = 0
        while n is None or k < n:
            new_x = np.cos(self.x)
            diff = abs(new_x - self.x)
            record = {"iter": k, "x": self.x, "new_x": new_x, "diff": diff}
            self.history.record(**record)
            if self.verbose>1:
                logger.debug(f"[A] it={k} x={self.x:.4f} diff={diff:.4e}")
            yield record
            self.x = new_x
            k += 1

    def iterate_list(self, n: int):
        """Retourne la liste complète des itérations."""
        return list(self.iterate_gen(n))


if __name__ == "__main__":
    
    verbose = 1
    
    graph_dir = os.path.join('.', 'dataGenerated', 'plot')
    os.makedirs(graph_dir, exist_ok=True)
    tracker_dir = os.path.join('.', 'dataGenerated', 'historyTracker')
    os.makedirs(tracker_dir, exist_ok=True)
    
    a = A(x0=1.0, verbose=verbose)
    for step in a.iterate_gen(5):
        print(step)
    
    a.history.plot("x", color="blue", show=False, base_dir=graph_dir)
    a.history.save_pickle(os.path.join(tracker_dir, "history_run_a.pkl"))
