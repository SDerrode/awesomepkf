#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Optional

from prg.base_classes.nonlinear_upkf_runner_base import BaseNonLinearUPKFRunner
from prg.exceptions import FilterError, ParamError, PKFError

__all__ = ["BaseNonLinearUPKFRunnerSim"]


class BaseNonLinearUPKFRunnerSim(BaseNonLinearUPKFRunner):
    """
    Runner for nonlinear simulation + UPKF filtering.
    """

    def __init__(
        self,
        model_name: str,
        N: int,
        sKey: Optional[int],
        sigmaSet: Optional[str],
        verbose: int,
        plot: bool,
        save_history: bool,
        base_dir: str = ".",
    ) -> None:
        """
        Raises
        ------
        ParamError
            Si ``N`` n'est pas un entier strictement positif,
            ``verbose`` invalide, ``model_name`` inconnu,
            ou ``sigmaSet`` inconnu du registre.
        PKFError
            Si l'instanciation du filtre échoue.
        """
        if not (isinstance(N, int) and N > 0):
            raise ParamError(f"N must be a strictly positive integer, got {N!r}.")

        self.N = N
        self.sKey = sKey

        super().__init__(model_name, sigmaSet, verbose, plot, save_history, base_dir)

    # ==========================================================

    def run(self, i: int = 0) -> list:
        """
        Raises
        ------
        FilterError
            Si la simulation ou le filtrage échoue de manière inattendue.
        PKFError
            Si une erreur du domaine PKF remonte du filtre.
        """

        try:
            self.runner_instance.process_N_data(N=self.N)
        except PKFError:
            raise
        except Exception as e:
            raise FilterError(
                f"Filtering failed (simulation mode) for model {self.model_name!r}."
            ) from e

        if self.save_history:
            self._save_history(f"history_run_upkf_simulation_{i}.pkl")

        self._compute_errors()

        if self.plot:
            self._plot_results()

        return self.runner_instance.history._history.copy()
