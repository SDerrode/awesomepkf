#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import logging
from typing import Optional

from classes.Linear_PKF import Linear_PKF

from base_classes.linear_pkf_runner_base import BaseLinearPKFRunner


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

    def run(self) -> None:

        if self.verbose > 1:
            logging.info("Starting Linear PKF Runner (simulation mode)")

        self.runner_instance.process_N_data(N=self.N)

        if self.save_history:
            self._save_history("history_run_pkf_simulation.pkl")

        self._compute_errors()

        if self.plot:
            self._plot_results()
