#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import logging
from typing import Optional

from classes.NonLinear_EPKF import NonLinear_EPKF
from others.utils import file_data_generator
from others.plot_settings import WINDOW

from base_classes.nonlinear_epkf_runner_base import NonLinearEPKFRunnerBase


class NonLinearEPKFRunnerFromFile(NonLinearEPKFRunnerBase):
    """
    Runner for filtering nonlinear data loaded from file.
    """

    def __init__(
        self,
        model_name: str,
        ell: int,
        data_filename: Optional[str],
        verbose: int = 0,
        plot: bool = False,
        save_history: bool = False,
        base_dir: str = ".",
    ) -> None:

        super().__init__(model_name, ell, verbose, plot, save_history, base_dir)

        self.data_filename = (
            os.path.join(self.datafile_dir, data_filename)
            if data_filename
            else os.path.join(self.datafile_dir, f"dataNonLinear_{model_name}.csv")
        )

    # ==========================================================

    def run(self) -> None:

        if self.verbose>1:
            logging.info("Starting NonLinear EPKF Runner (file mode)")

        self.epkf = NonLinear_EPKF(
            param=self.param,
            ell=self.ell,
            verbose=self.verbose
        )

        self.epkf.process_N_data(
            N=None,
            data_generator=file_data_generator(
                self.data_filename,
                self.param.dim_x,
                self.param.dim_y,
                self.verbose
            )
        )

        if self.save_history:
            self._save_history("history_run_epkf_file.pkl")

        self._compute_errors()

        if self.plot:
            self._plot_results()

    # ----------------------------------------------------------

    def _plot_results(self) -> None:

        title = f"{self.model_name} filtered with EPKF"

        self.epkf.history.plot(
            title,
            list_param=["ykp1"],
            list_label=["Observations y"],
            list_covar=[None],
            window=WINDOW,
            basename=f"epkf_observations_{self.model_name}",
            show=False,
            base_dir=self.graph_dir
        )
