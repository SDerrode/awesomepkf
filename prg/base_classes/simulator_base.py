#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import logging
from abc import ABC, abstractmethod

from prg.utils.utils import save_dataframe_to_csv, data_to_dataframe

__all__ = ["BaseDataSimulator"]

# =============================================================
# Logger configuration
# =============================================================


def setup_logger(verbose: int) -> logging.Logger:
    logger = logging.getLogger(__name__)

    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter("[%(levelname)s] %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    logger.propagate = False

    if verbose == 0:
        logger.setLevel(logging.WARNING)
    elif verbose == 1:
        logger.setLevel(logging.INFO)
    else:
        logger.setLevel(logging.DEBUG)

    return logger


# =============================================================
# Base class
# =============================================================


class BaseDataSimulator(ABC):
    """
    Abstract base class for data simulation.
    """

    def __init__(
        self,
        model_name: str,
        N: int,
        sKey: int | None,
        data_file_name: str | None,
        verbose: int,
        withoutX: bool,
    ) -> None:

        self.verbose = verbose
        self.logger = setup_logger(verbose)

        self.model_name = model_name
        self.N = N
        self.sKey = sKey
        self.withoutX = withoutX

        if data_file_name is None:
            data_file_name = self.default_filename()
        self.data_file_name = data_file_name

        self.base_dir = os.path.join(".", "data")
        self.datafile_dir = os.path.join(self.base_dir, "datafile")

        self._validate_inputs()
        self.param = self._build_parameters()

    # ---------------------------------------------------------
    # Abstract methods
    # ---------------------------------------------------------

    @abstractmethod
    def default_filename(self) -> str:
        pass

    @abstractmethod
    def create_model(self):
        pass

    @abstractmethod
    def create_param(self, dim_x, dim_y, params):
        pass

    @abstractmethod
    def create_pkf(self):
        pass

    # ---------------------------------------------------------
    # Shared logic
    # ---------------------------------------------------------

    def _validate_inputs(self) -> None:
        if self.sKey is not None and self.sKey < 0:
            raise ValueError("sKey must be >= 0")

    def _build_parameters(self):

        if self.verbose > 1:
            self.logger.debug("Creating model...")

        model = self.create_model()

        if self.verbose > 1:
            self.logger.debug(f"Model selected: {model.MODEL_NAME}")

        params = model.get_params().copy()
        dim_x = params.pop("dim_x")
        dim_y = params.pop("dim_y")

        param = self.create_param(dim_x, dim_y, params)

        if self.verbose > 1:
            self.logger.debug("Parameter summary:")
            param.summary()

        return param

    def run(self) -> None:

        if self.verbose > 1:
            self.logger.info("Starting simulation")

        pkf = self.create_pkf()

        if self.verbose > 1:
            self.logger.debug(f"Simulating N={self.N} data points...")

        list_data = pkf.simulate_N_data(N=self.N)

        df = data_to_dataframe(
            list_data,
            self.param.dim_x,
            self.param.dim_y,
            withoutX=self.withoutX,
        )

        os.makedirs(self.datafile_dir, exist_ok=True)

        filepath = os.path.join(self.datafile_dir, self.data_file_name)
        save_dataframe_to_csv(df, filepath)

        if self.verbose > 1:
            self.logger.info(f"Data saved to {filepath}")
