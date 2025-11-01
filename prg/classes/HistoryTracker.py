
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import numbers
import matplotlib.ticker as mticker


class A:
    def __init__(self, x0=1.0):
        self.x = x0
        self.history = HistoryTracker()  # Pour suivre l'évolution

    # ------------------------------------------------------------------
    # Générateur
    # ------------------------------------------------------------------
    def iterate_gen(self, n=None):
        """
        Générateur qui calcule x_{k+1} = cos(x_k) itérativement.
        Yield un dictionnaire avec itération et valeurs.
        Si n est None, itère indéfiniment.
        """
        k = 0
        while n is None or k < n:
            new_x = np.cos(self.x)
            diff = abs(new_x - self.x)

            print(f'k={k}')
            record = {"iter": k, "x": self.x, "new_x": new_x, "diff": diff}
            self.history.record(**record)

            yield record  # <-- renvoie l'état courant

            self.x = new_x  # mise à jour
            k += 1

    # ------------------------------------------------------------------
    # Méthode pour récupérer la liste complète
    # ------------------------------------------------------------------
    def iterate_list(self, n):
        """
        Lance le générateur pour obtenir la liste complète des états.
        """
        return list(self.iterate_gen(n))



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
    def plot(self, param, iter_key="iter", show=True, ax=None, fig=None, base_dir=None, **kwargs):
        """
        Trace l'évolution d'un paramètre au fil des itérations.
        Si le paramètre est vectoriel (ex: numpy.ndarray), chaque composante est tracée séparément.
        Si show=False, chaque figure est sauvegardée dans base_dir avec un indice dans le nom.

        Arguments :
        -----------
        param : str
            Nom du paramètre à tracer (scalaire ou vecteur)
        iter_key : str
            Clé utilisée pour l'axe X (par défaut 'iter')
        show : bool
            Si True → affiche le graphique, sinon sauvegarde dans base_dir
        ax : matplotlib.axes.Axes | None
            Axe existant (optionnel, utilisé seulement pour les scalaires)
        base_dir : str | None
            Dossier de sauvegarde si show=False
        kwargs :
            Paramètres passés à matplotlib.plot() (couleur, style, etc.)
        """

        df = pd.DataFrame(self._history.copy())

        if df.empty:
            raise ValueError("Aucune donnée enregistrée.")
        if param not in df.columns:
            raise KeyError(f"'{param}' n'est pas une colonne enregistrée. Colonnes disponibles : {list(df.columns)}")

        # Récupération des données
        y_values = df[param]
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
                print(f"[HistoryTracker] Graphique sauvegardé : {save_path}")
                if created_fig:
                    plt.close(fig)
            return ax, fig
        else:
            # --- Cas vectoriel ------------------------------------------------
            arr = np.vstack(y_values.values)
            n_components = arr.shape[1]

            for i in range(n_components):

                created_fig = False
                if ax is None:
                    print('Je cré une nouvelle figure')
                    fig, ax = plt.subplots(figsize=(6, 4))
                    created_fig = True
                
                yi = arr[:, i]
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
    # Utilitaires divers
    # ------------------------------------------------------------------
    def __len__(self):
        return len(self._history)

    def __repr__(self):
        return f"<HistoryTracker n_records={len(self)}>"


if __name__ == "__main__":
    a = A(x0=1.0)

    # Utilisation du générateur étape par étape
    print("Itérations individuelles avec iterate_gen:")
    for step in a.iterate_gen(5):
        print(step)

    b = A(x0=3.0)
    # Utilisation pour obtenir une liste complète
    all_steps = b.iterate_list(15)
    print("\nListe complète des étapes avec iterate_list:")
    print(all_steps)

    # Affichage de l'historique sous forme de DataFrame
    print("\nHistorique complet:")
    print(a.history.as_dataframe())

    # Affiche le graphique
    a.history.plot("x", color="blue")

    # Sauvegarde le graphique sans l'afficher
    graph_dir = os.path.join('.', 'dataGenerated', 'plot')
    os.makedirs(graph_dir, exist_ok=True)
    b.history.plot("diff", color="red", marker="s", linestyle="-", show=False, base_dir=graph_dir)

    # Sauvegarde l'historique en pickle
    tracker_dir = os.path.join('.', 'dataGenerated', 'historyTracker')
    os.makedirs(tracker_dir, exist_ok=True)
    a.history.save_pickle(os.path.join('.', 'dataGenerated', 'historyTracker', 'history_run_a.pkl'))
    b.history.save_pickle(os.path.join('.', 'dataGenerated', 'historyTracker', 'history_run_b.pkl'))

    # Rechargement
    # h2 = HistoryTracker.load_pickle("history_run.pkl")
    # print(h2.as_dataframe())
    