#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import logging
from abc import ABC, abstractmethod
from typing import Optional, Tuple

from models.nonLinear import ModelFactoryNonLinear
from classes.ParamNonLinear import ParamNonLinear
from classes.NonLinear_EPKF import NonLinear_EPKF


class NonLinearEPKFRunnerBase(ABC):
    """
    Base runner for NonLinear EPKF workflows.
    Contains all shared infrastructure logic.
    """

    def __init__(
        self,
        model_name: str,
        ell: Optional[int],
        verbose: int,
        plot: bool,
        save_history: bool,
        base_dir: str = "."
    ) -> None:

        self.model_name = model_name
        self.ell = ell
        self.verbose = verbose
        self.plot = plot
        self.save_history = save_history
        self.base_dir = base_dir

        self._configure_logging()
        self.tracker_dir, self.datafile_dir, self.graph_dir = self._setup_directories()

        self.model, self.param = self._build_model()
        self.epkf: Optional[NonLinear_EPKF] = None

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

    def _build_model(self) -> Tuple[object, ParamNonLinear]:
        model = ModelFactoryNonLinear.create(self.model_name)
        params = model.get_params().copy()

        dim_x = params.pop("dim_x")
        dim_y = params.pop("dim_y")

        param = ParamNonLinear(self.verbose, dim_x, dim_y, **params)

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

        self.epkf.history.compute_errors(
            self.epkf,
            ['xkp1'],
            ['Xkp1_update'],
            ['PXXkp1_update'],
            ['ikp1'],
            ['Skp1']
        )

    # ----------------------------------------------------------

    def _save_history(self, filename: str) -> None:
        filepath = os.path.join(self.tracker_dir, filename)
        self.epkf.history.save_pickle(filepath)

        if self.verbose > 1:
            logging.info(f"History saved to {filepath}")

    # ==========================================================
    # Abstract API
    # ==========================================================

    @abstractmethod
    def run(self) -> None:
        """Main execution logic (must be implemented by child class)."""
        pass
