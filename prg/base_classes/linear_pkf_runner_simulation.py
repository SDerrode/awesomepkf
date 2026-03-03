#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import logging
from typing import Optional

from prg.classes.Linear_PKF import Linear_PKF
from prg.base_classes.linear_pkf_runner_base import BaseLinearPKFRunner
from prg.exceptions import FilterError, ParamError, PKFError

__all__ = ["LinearPKFRunnerSim"]


class LinearPKFRunnerSim(BaseLinearPKFRunner):
    """
    Runner for linear simulation + PKF filtering.
    """

    def __init__(
        self,
        model_name: str,
        N: int,
        sKey: Optional[int],
        verbose: int,
        plot: bool,
        save_history: bool,
        base_dir: str = ".",
    ) -> None:
        """
        Initialise le runner en mode simulation.

        Parameters
        ----------
        model_name : str
            Nom du modèle.
        N : int
            Nombre de pas de temps à simuler.
        sKey : int or None
            Graine aléatoire pour la reproductibilité.
        verbose : int
            Niveau de verbosité (0, 1 ou 2).
        plot : bool
            Si True, affiche les graphiques après le run.
        save_history : bool
            Si True, sauvegarde l'historique en pickle.
        base_dir : str, optional
            Répertoire de base pour les sorties.

        Raises
        ------
        ParamError
            Si ``verbose`` est invalide, ``N`` n'est pas un entier
            strictement positif, ou ``model_name`` est inconnu.
        PKFError
            Si l'instanciation du filtre échoue.
        """
        if not (isinstance(N, int) and N > 0):
            raise ParamError(f"N must be a strictly positive integer, got {N!r}.")

        self.N = N
        self.sKey = sKey

        super().__init__(model_name, verbose, plot, save_history, base_dir)

    # ==========================================================

    def run(self, i: int = 0) -> None:
        """
        Exécute la simulation et le filtrage PKF.

        Parameters
        ----------
        i : int, optional
            Indice de run, utilisé pour nommer le fichier d'historique.

        Returns
        -------
        list
            Copie de l'historique du filtre.

        Raises
        ------
        FilterError
            Si la simulation ou le filtrage échoue de manière inattendue.
        PKFError
            Si une erreur du domaine PKF remonte du filtre.
        """
        if self.verbose > 1:
            logging.info("Starting Linear PKF Runner (simulation mode)")

        try:
            self.runner_instance.process_N_data(N=self.N)
        except PKFError:
            raise
        except Exception as e:
            raise FilterError(
                f"Filtering failed (simulation mode) for model {self.model_name!r}."
            ) from e

        if self.save_history:
            self._save_history(f"history_run_pkf_simulation_{i}.pkl")

        self._compute_errors()

        if self.plot:
            self._plot_results()

        return self.runner_instance.history._history.copy()
