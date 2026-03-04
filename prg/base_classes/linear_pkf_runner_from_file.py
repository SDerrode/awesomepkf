#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
from typing import Optional

from prg.classes.Linear_PKF import Linear_PKF
from prg.utils.utils import file_data_generator
from prg.base_classes.linear_pkf_runner_base import BaseLinearPKFRunner
from prg.exceptions import FilterError, ParamError, PKFError

__all__ = ["LinearPKFRunnerFromFile"]


class LinearPKFRunnerFromFile(BaseLinearPKFRunner):
    """
    Runner for filtering linear data loaded from file.
    """

    def __init__(
        self,
        model_name: str,
        data_filename: Optional[str],
        verbose: int = 0,
        plot: bool = False,
        save_history: bool = False,
        base_dir: str = ".",
    ) -> None:
        """
        Initialise le runner en mode lecture depuis fichier.

        Parameters
        ----------
        model_name : str
            Nom du modèle.
        data_filename : str or None
            Nom du fichier CSV de données. Si None, utilise le nom par défaut.
        verbose : int, optional
            Niveau de verbosité (0, 1 ou 2).
        plot : bool, optional
            Si True, affiche les graphiques après le run.
        save_history : bool, optional
            Si True, sauvegarde l'historique en pickle.
        base_dir : str, optional
            Répertoire de base pour les sorties.

        Raises
        ------
        ParamError
            Si ``verbose`` est invalide ou ``model_name`` inconnu.
        PKFError
            Si l'instanciation du filtre échoue.
        """
        self.N = -1
        self.sKey = None

        super().__init__(model_name, verbose, plot, save_history, base_dir)

        self.data_filename = (
            os.path.join(self.datafile_dir, data_filename)
            if data_filename
            else os.path.join(self.datafile_dir, f"dataLinear_{model_name}.csv")
        )

    # ==========================================================

    def run(self, i: int = 0) -> None:
        """
        Exécute le filtrage sur les données chargées depuis le fichier.

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
        FileNotFoundError
            Si le fichier de données est introuvable.
        FilterError
            Si le filtrage échoue de manière inattendue.
        PKFError
            Si une erreur du domaine PKF remonte du filtre.
        """

        if not os.path.exists(self.data_filename):
            raise FileNotFoundError(f"Data file not found: {self.data_filename!r}.")

        try:
            self.runner_instance.process_N_data(
                N=None,
                data_generator=file_data_generator(
                    self.data_filename,
                    self.param.dim_x,
                    self.param.dim_y,
                    self.verbose,
                ),
            )
        except PKFError:
            raise
        except Exception as e:
            raise FilterError(
                f"Filtering failed (file mode) for model {self.model_name!r}."
            ) from e

        if self.save_history:
            self._save_history(f"history_run_pkf_file_{i}.pkl")

        self._compute_errors()

        if self.plot:
            self._plot_results()

        return self.runner_instance.history._history.copy()
