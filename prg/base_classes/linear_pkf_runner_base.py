#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import logging
from abc import ABC, abstractmethod
from typing import Optional, Tuple

from models.linear import ModelFactoryLinear
from classes.ParamLinear import ParamLinear
from classes.Linear_PKF import Linear_PKF


class LinearPKFRunnerBase(ABC):
    """
    Base runner for Linear PKF workflows.
    Contains all shared infrastructure logic.
    """

    def __init__(
        self,
        model_name: str,
        verbose: int,
        plot: bool,
        save_history: bool,
        base_dir: str = "."
    ) -> None:

        self.model_name = model_name
        self.verbose = verbose
        self.plot = plot
        self.save_history = save_history
        self.base_dir = base_dir

        self._configure_logging()
        self.tracker_dir, self.datafile_dir, self.graph_dir = self._setup_directories()

        self.model, self.param = self._build_model()
        self.pkf: Optional[Linear_PKF] = None

    # ==========================================================
    # Shared infrastructure
    # ==========================================================

    def _configure_logging(self) -> None:
        level = logging.WARNING
        if self.verbose == 1:
            level = logging.INFO
        elif self.verbose >= 2:
            level = logging.DEBUG

        logging.basicConfig(
            level=level,
            format="%(asctime)s | %(levelname)s | %(message)s"
        )

    # ----------------------------------------------------------

    def _setup_directories(self) -> Tuple[str, str, str]:
        base_dir = os.path.join(self.base_dir, "data")

        tracker_dir  = os.path.join(base_dir, "historyTracker")
        datafile_dir = os.path.join(base_dir, "datafile")
        graph_dir    = os.path.join(base_dir, "plot")

        os.makedirs(tracker_dir, exist_ok=True)
        os.makedirs(datafile_dir, exist_ok=True)
        os.makedirs(graph_dir, exist_ok=True)

        return tracker_dir, datafile_dir, graph_dir

    # ----------------------------------------------------------

    def _build_model(self) -> Tuple[object, ParamLinear]:
        model = ModelFactoryLinear.create(self.model_name)
        params = model.get_params().copy()

        dim_x = params.pop("dim_x")
        dim_y = params.pop("dim_y")

        param = ParamLinear(self.verbose, dim_x, dim_y, **params)

        if self.verbose > 1:
            logging.debug(f"Model created: {model}")

        if self.verbose > 1:
            param.summary()

        return model, param

    # ==========================================================
    # Shared post-processing
    # ==========================================================

    def _compute_errors(self) -> None:
        if self.verbose > 1:
            logging.debug("Computing errors")

        self.pkf.history.compute_errors(
            self.pkf,
            ['xkp1'],
            ['Xkp1_update'],
            ['PXXkp1_update'],
            ['ikp1'],
            ['Skp1']
        )

    # ----------------------------------------------------------

    def _save_history(self, filename: str) -> None:
        filepath = os.path.join(self.tracker_dir, filename)
        self.pkf.history.save_pickle(filepath)

        if self.verbose > 1:
            logging.info(f"History saved to {filepath}")

    # ==========================================================
    # Abstract API
    # ==========================================================

    @abstractmethod
    def run(self) -> None:
        """Main execution logic (must be implemented by child class)."""
        pass
