#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import logging
from typing import Optional

from classes.NonLinear_UPKF import NonLinear_UPKF
from others.plot_settings import WINDOW

from base_classes.nonlinear_upkf_runner_base import NonLinearUPKFRunnerBase


class NonLinearUPKFRunner(NonLinearUPKFRunnerBase):
    """
    Runner for nonlinear simulation + UPKF filtering.
    """

    def __init__(
        self,
        model_name: str,
        N: int,
        sKey: Optional[int],
        sigmaSet: Optional[int],
        verbose: int,
        plot: bool,
        save_history: bool,
        base_dir: str = "."
    ) -> None:

        super().__init__(model_name, sigmaSet, verbose, plot, save_history, base_dir)

        self.N = N
        self.sKey = sKey

    # ==========================================================

    def run(self) -> None:

        if self.verbose>1:
            logging.info("Starting NonLinear UPKF Runner (simulation mode)")

        self.upkf = NonLinear_UPKF(
            self.param,
            sKey=self.sKey,
            sigmaSet=self.sigmaSet,
            verbose=self.verbose
        )

        self.upkf.process_N_data(N=self.N)

        if self.save_history:
            self._save_history("history_run_upkf_simulation.pkl")

        self._compute_errors()

        if self.plot:
            self._plot_results()

    # ----------------------------------------------------------

    def _plot_results(self) -> None:

        title = f"'{self.model_name}' model data filtered with UPKF"

        self.upkf.history.plot(
            title,
            list_param=["xkp1", "Xkp1_update"],
            list_label=["x true", "x estimated"],
            list_covar=[None, "PXXkp1_update"],
            window=WINDOW,
            basename=f"upkf_{self.model_name}",
            show=False,
            base_dir=self.graph_dir
        )
