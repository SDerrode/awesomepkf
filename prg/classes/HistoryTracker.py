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
    def plot(self, list_param, list_label, fenetre, basename="plot", iter_key="iter", show=True, base_dir=None, **kwargs):
        """
        Trace l'évolution d'un paramètre au fil des itérations.
        Si show=False, chaque figure est sauvegardée dans base_dir.
        """

        df = pd.DataFrame(self._history.copy())

        assert not df.empty, "Aucune donnée enregistrée."
        for p in list_param:
            assert p in df.columns, f"'{p}' n'est pas une colonne connue : {list(df.columns)}"

        # Extraction des données
        y_values = df[list_param]
        x = df[iter_key] if iter_key in df.columns else df.index

        # Vérifie que le premier élément est bien un vecteur
        first = y_values[list_param[0]].iloc[0]
        assert hasattr(first, "shape"), f"Le premier élément de '{list_param[0]}' n'est pas un vecteur numpy"
        nb_components = first.shape[0]

        datafocus = pd.DataFrame()
        for p in list_param:
            labels = [f'{p}_{component}' for component in range(nb_components)]
            datafocus[labels] = df[p].apply(lambda x: pd.Series(x.flatten()))
        # On ne sélectione qu'une fenetre
        df_subset = datafocus.iloc[fenetre['xmin']:fenetre['xmax']]
        
        liste_ax = []
        for component in range(nb_components):
            fig, ax = plt.subplots(figsize=(6, 4))
            liste_ax.append(ax)

            labels = [f'{p}_{component}' for p in list_param]
            for col, label in zip(labels, list_label):
                df_subset[col].plot(ax=ax, label=label, alpha=0.5)

            ax.legend()
            ax.set_xlim(fenetre['xmin'], fenetre['xmax']-1)
            ax.set_xlabel(iter_key)
            ax.grid(True, linestyle="--", alpha=0.6)
            ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))

            if show:
                plt.show()
            else:
                os.makedirs(base_dir or ".", exist_ok=True)
                save_path = os.path.join(base_dir or ".", f"{basename}_{component}.png")
                ax.figure.savefig(save_path, dpi=150, bbox_inches="tight")
                if self.verbose > 0:
                    logger.info(f"[HistoryTracker] Graphique sauvegardé : {save_path}")
                plt.close(fig)
        return liste_ax

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
