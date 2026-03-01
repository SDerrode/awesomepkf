#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import logging
from typing import Optional

from classes.Linear_PKF import Linear_PKF
from others.utils import file_data_generator

from base_classes.linear_pkf_runner_base import BaseLinearPKFRunner


class LinearPKFRunnerFromFile(BaseLinearPKFRunner):
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

        self.N = -1
        self.sKey = None

        super().__init__(model_name, verbose, plot, save_history, base_dir)

        self.data_filename = (
            os.path.join(self.datafile_dir, data_filename)
            if data_filename
            else os.path.join(self.datafile_dir, f"dataLinear_{model_name}.csv")
        )

    # ==========================================================

    def run(self, i: int = 0) -> None:

        if self.verbose > 1:
            logging.info("Starting Linear PKF Runner (file mode)")

        try:
            self.runner_instance.process_N_data(
                N=None,
                data_generator=file_data_generator(
                    self.data_filename, self.param.dim_x, self.param.dim_y, self.verbose
                ),
            )
        except RuntimeError as rte:
            raise

        if self.save_history:
            self._save_history(f"history_run_pkf_file_{i}.pkl")

        self._compute_errors()

        if self.plot:
            self._plot_results()

        return self.runner_instance.history._history.copy()
