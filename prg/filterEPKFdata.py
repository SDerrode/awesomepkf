#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import argparse
import logging
from typing import Optional

# Models / Params / Algo
from models.nonLinear import ModelFactoryNonLinear
from classes.NonLinear_EPKF import NonLinear_EPKF
from classes.ParamNonLinear import ParamNonLinear

# Utils
from others.parser import addParseToParser
from others.plot_settings import WINDOW


# ==============================================================
# Runner (logique métier indépendante du CLI)
# ==============================================================

class NonLinearEPKFRunner:
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

        self.model_name = model_name
        self.N = N
        self.sKey = sKey
        self.ell = ell
        self.verbose = verbose
        self.plot = plot
        self.save_history = save_history
        self.base_dir = base_dir

        self._configure_logging()
        self.tracker_dir, _, self.graph_dir = self._setup_directories()

        self.model, self.param = self._build_model()
        self.epkf: Optional[NonLinear_EPKF] = None


    # ----------------------------------------------------------
    # Private helpers
    # ----------------------------------------------------------

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

    def _setup_directories(self) -> tuple[str, str, str]:
        base_dir = os.path.join(self.base_dir, "data")

        tracker_dir  = os.path.join(base_dir, "historyTracker")
        datafile_dir = os.path.join(base_dir, "datafile")
        graph_dir    = os.path.join(base_dir, "plot")

        os.makedirs(tracker_dir, exist_ok=True)
        os.makedirs(datafile_dir, exist_ok=True)
        os.makedirs(graph_dir, exist_ok=True)

        return tracker_dir, datafile_dir, graph_dir

    def _build_model(self) -> tuple[object, ParamNonLinear]:
        model = ModelFactoryNonLinear.create(self.model_name)
        params = model.get_params().copy()

        dim_x = params.pop("dim_x")
        dim_y = params.pop("dim_y")

        param = ParamNonLinear(self.verbose, dim_x, dim_y, **params)

        logging.debug(f"Model created: {model}")

        if self.verbose > 1:
            param.summary()

        return model, param


    # ----------------------------------------------------------
    # Public API
    # ----------------------------------------------------------

    def run(self) -> None:

        if self.verbose > 1:
            logging.info("Starting NonLinear EPKF Runner")

        self.epkf = NonLinear_EPKF(
            self.param,
            sKey=self.sKey,
            ell=self.ell,
            verbose=self.verbose
        )

        if self.verbose > 1:
            logging.info(f"Processing N={self.N} samples")

        listeEPKF = self.epkf.process_N_data(
            N=self.N
        )

        if self.verbose > 1:
            logging.debug(self.epkf.history.as_dataframe().head())

        if self.save_history:
            self._save_history()
        
        self._compute_errors()
        if  self.plot:
            self._plot_results()

    # ----------------------------------------------------------
    # Post-processing
    # ----------------------------------------------------------

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

    def _save_history(self) -> None:

        filepath = os.path.join(self.tracker_dir, "history_run_epfk_1.pkl")
        self.epkf.history.save_pickle(filepath)

        if self.verbose > 1:
            logging.info(f"History saved to {filepath}")

    def _plot_results(self) -> None:

        title = f"'{self.model_name}' model data filtered with EPKF"

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


# ==============================================================
# CLI
# ==============================================================

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Simulate and filter non linear data with EPKF"
    )

    addParseToParser(parser, ['nonLinearModelName', 'N', 'sKey', 'ell'])

    args = parser.parse_args()

    if args.sKey is not None and args.sKey < 0:
        parser.error("sKey must be >= 0")

    return args


def main() -> None:
    args = parse_arguments()

    runner = NonLinearEPKFRunner(
        model_name=args.nonLinearModelName,
        N=args.N,
        sKey=args.sKey,
        ell=args.ell,
        verbose=args.verbose,
        plot=args.plot,
        save_history=args.saveHistory,
    )

    runner.run()


if __name__ == "__main__":
    main()
