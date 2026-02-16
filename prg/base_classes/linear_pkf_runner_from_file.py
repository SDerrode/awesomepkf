#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import logging
from typing import Optional

from classes.Linear_PKF import Linear_PKF
from others.utils import file_data_generator
from others.plot_settings import WINDOW

from base_classes.linear_pkf_runner_base import LinearPKFRunner


class LinearPKFRunnerFromFile(LinearPKFRunner):
    """
    Runner for filtering linear data loaded from file.
    """

    def __init__(
        self,
        model_name: str,
        data_filename: Optional[str],
        verbose: int = 0,
        plot: bool = False,
        save_history: bool = False,
        base_dir: str = ".",
    ) -> None:

        super().__init__(model_name, verbose, plot, save_history, base_dir)

        self.data_filename = (
            os.path.join(self.datafile_dir, data_filename)
            if data_filename
            else os.path.join(self.datafile_dir, f"dataLinear_{model_name}.csv")
        )

    # ==========================================================

    def run(self) -> None:

        if self.verbose>1:
            logging.info("Starting Linear PKF Runner (file mode)")

        self.runner_instance.process_N_data(
            N=None,
            data_generator=file_data_generator(
                self.data_filename,
                self.param.dim_x,
                self.param.dim_y,
                self.verbose
            )
        )

        if self.save_history:
            self._save_history("history_run_pkf_file.pkl")

        self._compute_errors()

        if self.plot:
            self._plot_results()

    # ----------------------------------------------------------

    def _plot_results(self) -> None:

        title = f"{self.model_name} filtered with PKF"

        self.runner_instance.history.plot(
            title,
            list_param=["ykp1"],
            list_label=["Observations y"],
            list_covar=[None],
            window=WINDOW,
            basename=f"pkf_observations_{self.model_name}",
            show=False,
            base_dir=self.graph_dir
        )
