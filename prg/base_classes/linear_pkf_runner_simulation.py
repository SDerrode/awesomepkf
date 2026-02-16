#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import logging
from typing import Optional

from classes.Linear_PKF import Linear_PKF
from others.plot_settings import WINDOW

from base_classes.linear_pkf_runner_base import LinearPKFRunner


class LinearPKFRunner(LinearPKFRunner):
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
        base_dir: str = "."
    ) -> None:

        super().__init__(model_name, verbose, plot, save_history, base_dir)

        self.N = N
        self.sKey = sKey

    # ==========================================================

    def run(self) -> None:

        if self.verbose>1:
            logging.info("Starting Linear PKF Runner (simulation mode)")

        self.runner_instance.process_N_data(N=self.N)

        if self.save_history:
            self._save_history("history_run_pkf_simulation.pkl")

        self._compute_errors()

        if self.plot:
            self._plot_results()

    # ----------------------------------------------------------

    def _plot_results(self) -> None:

        title = f"'{self.model_name}' model data filtered with PKF"

        self.runner_instance.history.plot(
            title,
            list_param=["xkp1", "Xkp1_update"],
            list_label=["x true", "x estimated"],
            list_covar=[None, "PXXkp1_update"],
            window=WINDOW,
            basename=f"pkf_{self.model_name}",
            show=False,
            base_dir=self.graph_dir
        )
