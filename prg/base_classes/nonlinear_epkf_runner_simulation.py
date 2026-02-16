#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import logging
from typing import Optional

from classes.NonLinear_EPKF import NonLinear_EPKF

from base_classes.nonlinear_epkf_runner_base import NonLinearEPKFRunner


class NonLinearEPKFRunner(NonLinearEPKFRunner):
    """
    Runner for nonlinear simulation + EPKF filtering.
    """

    def __init__(self, model_name: str, N: int, sKey: Optional[int], ell: Optional[int], verbose: int, plot: bool, save_history: bool, base_dir: str = ".") -> None:

        super().__init__(model_name, ell, verbose, plot, save_history, base_dir)

        self.N = N
        self.sKey = sKey

    # ==========================================================

    def run(self) -> None:

        if self.verbose>1:
            logging.info("Starting NonLinear EPKF Runner (simulation mode)")

        self.runner_instance.process_N_data(N=self.N)

        if self.save_history:
            self._save_history("history_run_epkf_simulation.pkl")

        self._compute_errors()

        if self.plot:
            self._plot_results()
