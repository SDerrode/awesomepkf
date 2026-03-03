#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import logging
from typing import Optional

from prg.base_classes.nonlinear_ppf_runner_base import BaseNonLinearPPFRunner

__all__ = ["BaseNonLinearPPFRunnerSim"]


class BaseNonLinearPPFRunnerSim(BaseNonLinearPPFRunner):
    """
    Runner for nonlinear simulation + PPF filtering.
    """

    def __init__(
        self,
        model_name: str,
        N: int,
        sKey: Optional[int],
        nbParticles: Optional[int],
        verbose: int,
        plot: bool,
        save_history: bool,
        base_dir: str = ".",
    ) -> None:

        self.N = N
        self.sKey = sKey

        super().__init__(model_name, nbParticles, verbose, plot, save_history, base_dir)

    # ==========================================================

    def run(self, i: int = 0) -> None:

        if self.verbose > 1:
            logging.info("Starting NonLinear PPF Runner (simulation mode)")

        try:
            self.runner_instance.process_N_data(N=self.N)
        except RuntimeError as rte:
            raise

        if self.save_history:
            self._save_history(f"history_run_ppf_simulation_{i}.pkl")

        self._compute_errors()

        if self.plot:
            self._plot_results()

        return self.runner_instance.history._history.copy()
