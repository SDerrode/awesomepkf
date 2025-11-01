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

    def __init__(self, verbose: bool = True):
        self._history: list[dict[str, Any]] = []
        self.verbose = verbose

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
        if self.verbose:
            logger.info(f"[HistoryTracker] Historique sauvegardé dans '{path}' ({len(self)} enregistrements)")

    @classmethod
    def load_pickle(cls, path: str) -> "HistoryTracker":
        """Recharge un HistoryTracker à partir d'un fichier pickle."""
        with open(path, "rb") as f:
            data = pickle.load(f)
        tracker = cls()
        tracker._history = data
        logger.info(f"[HistoryTracker] Rechargé depuis '{path}' ({len(tracker)} enregistrements)")
        return tracker

    # ------------------------------------------------------------------
    #  Visualisation
    # ------------------------------------------------------------------
    def plot(
        self,
        param: str,
        iter_key: str = "iter",
        show: bool = True,
        ax: Optional[plt.Axes] = None,
        fig: Optional[plt.Figure] = None,
        base_dir: Optional[str] = None,
        **kwargs: Any,
    ):
        """
        Trace l'évolution d'un paramètre au fil des itérations.
        Si le paramètre est vectoriel, chaque composante est tracée séparément.
        """
        df = self.as_dataframe()
        if df.empty:
            raise ValueError("Aucune donnée enregistrée.")
        if param not in df.columns:
            raise KeyError(f"'{param}' n'est pas une colonne enregistrée. Colonnes disponibles : {list(df.columns)}")

        y_values = df[param]
        x = df[iter_key] if iter_key in df.columns else df.index
        first_val = y_values.iloc[0]
        is_vector = isinstance(first_val, (list, np.ndarray))

        if not is_vector:
            ax, fig = self._plot_scalar(x, y_values, param, iter_key, show, ax, fig, base_dir, **kwargs)
        else:
            arr = np.vstack(y_values.values)
            for i in range(arr.shape[1]):
                ax, fig = self._plot_scalar(
                    x, arr[:, i], f"{param}[{i}]", iter_key, show, None, None, base_dir, **kwargs
                )
        return ax, fig

    def _plot_scalar(
        self,
        x,
        y,
        label: str,
        iter_key: str,
        show: bool,
        ax: Optional[plt.Axes],
        fig: Optional[plt.Figure],
        base_dir: Optional[str],
        **kwargs: Any,
    ):
        """Trace un paramètre scalaire."""
        created_fig = False
        if ax is None:
            fig, ax = plt.subplots(figsize=(6, 4))
            created_fig = True

        ax.plot(x, y, **kwargs)
        ax.set_xlabel(iter_key)
        ax.set_ylabel(label)
        ax.set_title(f"Évolution de '{label}' ({len(x)} points)")
        ax.grid(True, linestyle="--", alpha=0.6)
        ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))

        if show:
            plt.show()
        else:
            os.makedirs(base_dir or ".", exist_ok=True)
            save_path = os.path.join(base_dir or ".", f"plot_{label.replace('[','_').replace(']','')}.png")
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
            if self.verbose:
                logger.info(f"[HistoryTracker] Graphique sauvegardé : {save_path}")
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
        self.x = x0
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
            if self.verbose:
                logger.debug(f"[A] it={k} x={self.x:.4f} diff={diff:.4e}")
            yield record
            self.x = new_x
            k += 1

    def iterate_list(self, n: int):
        """Retourne la liste complète des itérations."""
        return list(self.iterate_gen(n))


if __name__ == "__main__":
    a = A(x0=1.0)
    for step in a.iterate_gen(5):
        print(step)

    graph_dir = os.path.join('.', 'dataGenerated', 'plot')
    os.makedirs(graph_dir, exist_ok=True)
    a.history.plot("x", color="blue", show=False, base_dir=graph_dir)

    tracker_dir = os.path.join('.', 'dataGenerated', 'historyTracker')
    os.makedirs(tracker_dir, exist_ok=True)
    a.history.save_pickle(os.path.join(tracker_dir, "history_run_a.pkl"))
