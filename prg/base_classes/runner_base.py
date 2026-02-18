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
        # print(f'model_name={model_name}')
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
        # print('entrée dans _build_model - BaseRunner')
        
        factoryL, factoryNL = self._get_model_factory()
        # print(f'factoryL={factoryL}')
        # print('Liste models : ', factoryL.list_models())
        # print(f'factoryNL={factoryNL}')
        # print('Liste models : ', factoryNL.list_models())
        
        param_class_linear, param_class_nonlinear = self._get_param_class()
        # print(f'param_class_linear={param_class_linear}')
        # print(f'param_class_nonlinear={param_class_nonlinear}')
        
        if self.model_name in factoryL.list_models():
            factory     = factoryL
            param_class = param_class_linear
        elif self.model_name in factoryNL.list_models():
            factory     = factoryNL
            param_class = param_class_nonlinear
        
        # print(f'factory={factory}')
        # print(f'param_class={param_class}')
        # input('atretreterte')

        model = factory.create(self.model_name)
        params = model.get_params().copy()
        dim_x = params.pop("dim_x")
        dim_y = params.pop("dim_y")

        param = param_class(self.verbose, dim_x, dim_y, **params)
        # print(f'model={model}')
        # print(f'param={param}')
        # input('ATTENTE _build_model - BaseRunner')

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
            
        # print(f'self.runner_instance={self.runner_instance}')
        # print(f'self.runner_instance={self.runner_instance.history}')
        # # print(f'self.runner_instance={self.runner_instance.history._history}')
        # print(f'\nself.runner_instance={self.runner_instance.history.last()}')
        # print(f'\nself.runner_instance={self.runner_instance.history.last().keys()}')
        
        history_keys = self.runner_instance.history.last().keys()
        # print('ikp1' in history_keys)
        # input('ATTENTE oppio')
        self.runner_instance.history.compute_errors(
            self.runner_instance,
            ['xkp1'],
            ['Xkp1_update'],
            ['PXXkp1_update'],
            ['ikp1'] if 'ikp1' in history_keys else None,
            ['Skp1'] if 'Skp1' in history_keys else None
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
