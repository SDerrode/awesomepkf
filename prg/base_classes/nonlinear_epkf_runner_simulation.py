#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import logging
from typing import Optional

from classes.NonLinear_EPKF import NonLinear_EPKF
from others.plot_settings import WINDOW

from base_classes.nonlinear_epkf_runner_base import NonLinearEPKFRunnerBase


class NonLinearEPKFRunner(NonLinearEPKFRunnerBase):
    """
    Runner for nonlinear simulation + EPKF filtering.
    """

    def __init__(
        self,
        model_name: str,
        N: int,
        sKey: Optional[int],
        ell: Optional[int],
        verbose: int,
        plot: bool,
        save_history: bool,
        base_dir: str = "."
    ) -> None:

        super().__init__(model_name, ell, verbose, plot, save_history, base_dir)

        self.N = N
        self.sKey = sKey

    # ==========================================================

    def run(self) -> None:

        if self.verbose>1:
            logging.info("Starting NonLinear EPKF Runner (simulation mode)")

        self.epkf = NonLinear_EPKF(
            self.param,
            sKey=self.sKey,
            ell=self.ell,
            verbose=self.verbose
        )

        self.epkf.process_N_data(N=self.N)

        if self.save_history:
            self._save_history("history_run_epkf_simulation.pkl")

        self._compute_errors()

        if self.plot:
            self._plot_results()

    # ----------------------------------------------------------

    def _plot_results(self) -> None:

        title = f"'{self.model_name}' model data filtered with EPKF"

        self.epkf.history.plot(
            title,
            list_param=["xkp1", "Xkp1_update"],
            list_label=["x true", "x estimated"],
            list_covar=[None, "PXXkp1_update"],
            window=WINDOW,
            basename=f"epkf_{self.model_name}",
            show=False,
            base_dir=self.graph_dir
        )
