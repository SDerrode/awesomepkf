#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import logging
from abc import ABC, abstractmethod
from typing import Optional, Tuple, Type

# ---------------------------------------------------------
# Base Runner
# ---------------------------------------------------------

class BaseRunner(ABC):
    """
    Abstract base runner for all PKF/EPKF/PF/UPKF workflows.
    Factorizes logging, directories, model building, and history management.
    """

    def __init__(self, model_name: str, verbose: int = 1, plot: bool = False, save_history: bool = False, base_dir: str = ".", **kwargs) -> None:
        self.model_name = model_name
        self.verbose = verbose
        self.plot = plot
        self.save_history = save_history
        self.base_dir = base_dir
        self._extra_args = kwargs  # ell, nbParticles, sigmaSet, etc.

        self._configure_logging()
        self.tracker_dir, self.datafile_dir, self.graph_dir = self._setup_directories()

        self.model, self.param = self._build_model()
        self.runner_instance = None  # Will be set by child (pkf, epkf, pf, upkf)

    # ==========================================================
    # Infrastructure
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

    @abstractmethod
    def _get_model_factory(self):
        """
        Return the factory class for model creation (ModelFactoryLinear / ModelFactoryNonLinear)
        """
        pass

    @abstractmethod
    def _get_param_class(self) -> Type:
        """
        Return the Param class (ParamLinear / ParamNonLinear)
        """
        pass

    def _build_model(self):
        factory = self._get_model_factory()
        param_class = self._get_param_class()

        model = factory.create(self.model_name)
        params = model.get_params().copy()
        dim_x = params.pop("dim_x")
        dim_y = params.pop("dim_y")

        param = param_class(self.verbose, dim_x, dim_y, **params)

        if self.verbose > 1:
            logging.debug(f"Model created: {model}")
            param.summary()

        return model, param

    # ==========================================================
    # Post-processing
    # ==========================================================

    def _compute_errors(self) -> None:
        if self.verbose > 1:
            logging.debug("Computing errors")
        self.runner_instance.history.compute_errors(
            self.runner_instance,
            ['xkp1'],
            ['Xkp1_update'],
            ['PXXkp1_update'],
            ['ikp1'] if hasattr(self.runner_instance, 'ikp1') else None,
            ['Skp1'] if hasattr(self.runner_instance, 'Skp1') else None
        )

    # ----------------------------------------------------------

    def _save_history(self, filename: str) -> None:
        filepath = os.path.join(self.tracker_dir, filename)
        self.runner_instance.history.save_pickle(filepath)

        if self.verbose > 1:
            logging.info(f"History saved to {filepath}")

    # ==========================================================
    # Abstract execution
    # ==========================================================

    @abstractmethod
    def run(self) -> None:
        """Execute the main workflow."""
        pass
