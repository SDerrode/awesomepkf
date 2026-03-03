#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import logging
from typing import Optional

from prg.utils.utils import file_data_generator
from prg.base_classes.nonlinear_upkf_runner_base import BaseNonLinearUPKFRunner
from prg.exceptions import FilterError, PKFError

__all__ = ["BaseNonLinearUPKFRunnerFromFile"]


class BaseNonLinearUPKFRunnerFromFile(BaseNonLinearUPKFRunner):
    """
    Runner for filtering nonlinear data loaded from file.
    """

    def __init__(
        self,
        model_name: str,
        sigmaSet: str,
        data_filename: Optional[str],
        verbose: int = 0,
        plot: bool = False,
        save_history: bool = False,
        base_dir: str = ".",
    ) -> None:
        """
        Raises
        ------
        ParamError
            Si ``verbose`` invalide, ``model_name`` inconnu,
            ou ``sigmaSet`` inconnu du registre.
        PKFError
            Si l'instanciation du filtre échoue.
        """
        self.N = -1
        self.sKey = None

        super().__init__(model_name, sigmaSet, verbose, plot, save_history, base_dir)

        self.data_filename = (
            os.path.join(self.datafile_dir, data_filename)
            if data_filename
            else os.path.join(self.datafile_dir, f"dataNonLinear_{model_name}.csv")
        )

    # ==========================================================

    def run(self, i: int = 0) -> list:
        """
        Raises
        ------
        FileNotFoundError
            Si le fichier de données est introuvable.
        FilterError
            Si le filtrage échoue de manière inattendue.
        PKFError
            Si une erreur du domaine PKF remonte du filtre.
        """
        if self.verbose > 1:
            logging.info("Starting NonLinear UPKF Runner (file mode)")

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
            self._save_history(f"history_run_upkf_file_{i}.pkl")

        self._compute_errors()

        if self.plot:
            self._plot_results()

        return self.runner_instance.history._history.copy()
