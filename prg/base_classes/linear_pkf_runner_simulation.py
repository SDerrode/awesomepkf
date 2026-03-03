#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import logging
from typing import Optional

from prg.classes.Linear_PKF import Linear_PKF
from prg.base_classes.linear_pkf_runner_base import BaseLinearPKFRunner

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

        self.N = N
        self.sKey = sKey

        super().__init__(model_name, verbose, plot, save_history, base_dir)

    # ==========================================================

    def run(self, i: int = 0) -> None:

        if self.verbose > 1:
            logging.info("Starting Linear PKF Runner (simulation mode)")

        try:
            self.runner_instance.process_N_data(N=self.N)
        except RuntimeError as rte:
            raise

        if self.save_history:
            self._save_history(f"history_run_pkf_simulation_{i}.pkl")

        self._compute_errors()

        if self.plot:
            self._plot_results()

        return self.runner_instance.history._history.copy()
