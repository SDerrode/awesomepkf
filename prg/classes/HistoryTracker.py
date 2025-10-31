
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class A:
    def __init__(self, x0=1.0):
        self.x = x0
        self.history = HistoryTracker()

    def iterate(self, n=10):
        for i in range(n):
            new_x = np.cos(self.x)
            diff = abs(new_x - self.x)
            self.history.record(iter=i, x=self.x, new_x=new_x, diff=diff)
            self.x = new_x
        return self.x




class HistoryTracker:
    """
    Classe pour enregistrer l'évolution de plusieurs quantités
    au fil des itérations, les sauvegarder et les visualiser.
    """

    def __init__(self):
        self._history = []

    # ------------------------------------------------------------------
    #  Gestion des enregistrements
    # ------------------------------------------------------------------
    def record(self, **quantities):
        """Sauvegarde l'état courant sous forme de dictionnaire."""
        self._history.append(quantities.copy())

    def as_dataframe(self):
        """Retourne l'historique sous forme de DataFrame pandas."""
        return pd.DataFrame(self._history)

    def last(self):
        """Retourne le dernier enregistrement."""
        return self._history[-1] if self._history else None

    def clear(self):
        """Efface tout l'historique."""
        self._history.clear()

    # ------------------------------------------------------------------
    #  Sauvegarde et chargement
    # ------------------------------------------------------------------
    def save_pickle(self, path):
        """Sauvegarde l'historique complet dans un fichier .pkl"""
        with open(path, "wb") as f:
            pickle.dump(self._history, f)
        print(f"[HistoryTracker] Sauvegardé dans '{path}' ({len(self)} enregistrements)")

    @classmethod
    def load_pickle(cls, path):
        """Recharge un HistoryTracker à partir d'un fichier pickle."""
        with open(path, "rb") as f:
            data = pickle.load(f)
        tracker = cls()
        tracker._history = data
        print(f"[HistoryTracker] Rechargé depuis '{path}' ({len(tracker)} enregistrements)")
        return tracker

    # ------------------------------------------------------------------
    #  Visualisation
    # ------------------------------------------------------------------
    def plot(self, param, iter_key="iter", show=True, ax=None, save_path=None, **kwargs):
        """
        Trace l'évolution d'un paramètre au fil des itérations.
        Si show=False, sauvegarde automatiquement l'image.

        Arguments :
        -----------
        param : str
            Nom du paramètre à tracer
        iter_key : str
            Clé utilisée pour l'axe X (par défaut 'iter')
        show : bool
            Si True → affiche le graphique, sinon sauvegarde
        save_path : str | None
            Chemin du fichier à sauvegarder (par défaut 'plot_<param>.png')
        kwargs :
            Paramètres passés à matplotlib.plot() (couleur, style, etc.)
        """
        df = pd.DataFrame(self._history.copy())

        if df.empty:
            raise ValueError("Aucune donnée enregistrée.")
        if param not in df.columns:
            raise KeyError(f"'{param}' n'est pas une colonne enregistrée. Colonnes disponibles : {list(df.columns)}")

        x = df[iter_key] if iter_key in df.columns else df.index
        y = df[param]

        created_fig = False
        if ax is None:
            fig, ax = plt.subplots(figsize=(6, 4))
            created_fig = True

        ax.plot(x, y, **kwargs)
        ax.set_xlabel(iter_key)
        ax.set_ylabel(param)
        ax.set_title(f"Évolution de '{param}' ({len(df)} points)")
        ax.grid(True, linestyle="--", alpha=0.6)

        if show:
            plt.show()
        else:
            save_path += f"plot_{param}.png"
            os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
            ax.figure.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"[HistoryTracker] Graphique sauvegardé : {save_path}")
            if created_fig:
                plt.close(fig)

        return ax

    # ------------------------------------------------------------------
    # Utilitaires divers
    # ------------------------------------------------------------------
    def __len__(self):
        return len(self._history)

    def __repr__(self):
        return f"<HistoryTracker n_records={len(self)}>"


if __name__ == "__main__":
    a = A(1.0)
    a.iterate(8)

    # Affiche le graphique
    a.history.plot("x", color="blue", save_path='./dataGenerated/historyTracker')

    # Sauvegarde le graphique sans l'afficher
    a.history.plot("diff", color="red", marker="s", linestyle="-", show=False, save_path='./dataGenerated/plot/')

    # Sauvegarde l'historique en pickle
    a.history.save_pickle('./dataGenerated/historyTracker/history_run.pkl')

    # Rechargement
    # h2 = HistoryTracker.load_pickle("history_run.pkl")
    # print(h2.as_dataframe())
    