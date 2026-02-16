#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import logging
import argparse
from typing import Optional

# Models / Params / Algo
from models.nonLinear import ModelFactoryNonLinear
from classes.ParamNonLinear import ParamNonLinear
from classes.NonLinear_UPKF import NonLinear_UPKF

# Utils
from others.utils import file_data_generator
from others.parser import addParseToParser
from others.plot_settings import WINDOW


# ==============================================================
# Runner (logique métier indépendante du CLI)
# ==============================================================

class NonLinearUPKFRunner_fromFile:

    def __init__(
        self,
        model_name: str,
        sigmaSet,
        verbose: int = 0,
        plot: bool = False,
        save_history: bool = False,
        data_filename: Optional[str] = None,
        N: Optional[int] = None,
        base_dir: str = ".",
    ):
        self.model_name = model_name
        self.sigmaSet = sigmaSet
        self.verbose = verbose
        self.plot = plot
        self.save_history = save_history
        self.N = N
        self.base_dir = base_dir

        self._configure_logging()
        self.tracker_dir, self.datafile_dir, self.graph_dir = self._setup_directories()

        self.model, self.param = self._build_model()
        self.upkf: Optional[NonLinear_UPKF] = None
        
        self.data_filename = (
            os.path.join(self.datafile_dir, data_filename)
            if data_filename
            else os.path.join(self.datafile_dir, f"dataNonLinear_{model_name}.csv")
        )

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

    # ----------------------------------------------------------

    def _setup_directories(self) -> tuple[str, str, str]:
        base_dir = os.path.join(self.base_dir, "data")

        tracker_dir  = os.path.join(base_dir, "historyTracker")
        datafile_dir = os.path.join(base_dir, "datafile")
        graph_dir    = os.path.join(base_dir, "plot")

        os.makedirs(tracker_dir, exist_ok=True)
        os.makedirs(datafile_dir, exist_ok=True)
        os.makedirs(graph_dir, exist_ok=True)

        return tracker_dir, datafile_dir, graph_dir

    # ----------------------------------------------------------

    def _build_model(self) -> tuple[object, ParamNonLinear]:
        model = ModelFactoryNonLinear.create(self.model_name)
        params = model.get_params().copy()

        dim_x = params.pop("dim_x")
        dim_y = params.pop("dim_y")

        param = ParamNonLinear(self.verbose, dim_x, dim_y, **params)
        
        # In test to integrate a linear model as a non linear model
        # model        = ModelFactoryLinear.create("A_mQ_x1_y1")
        # params       = model.get_params().copy()
        # dim_x, dim_y = params.pop('dim_x'), params.pop('dim_y')
        # param        = ParamLinear(verbose, dim_x, dim_y, **params)

        if self.verbose > 0:
            logging.debug(f"Model created: {model}")

        if self.verbose > 1:
            param.summary()

        return model, param

    # ----------------------------------------------------------

    def run(self) -> None:
        
        if self.verbose > 1:
            logging.info("Starting NonLinear UPKF Runner")

        self.upkf = NonLinear_UPKF(
            param    = self.param,
            sigmaSet = self.sigmaSet,
            verbose  = self.verbose
        )
        
        if self.verbose > 1:
            logging.info(f"Processing N={self.N} samples")

        listeUPKF = self.upkf.process_N_data(
            N=None,
            data_generator=file_data_generator(
                self.data_filename,
                self.param.dim_x,
                self.param.dim_y,
                self.verbose
            )
        )
        
        if self.verbose > 1:
            logging.debug(self.upkf.history.as_dataframe().head())
        
        if self.save_history:
            self._save_history()
        
        if listeUPKF and listeUPKF[0][1] is not None:
            self._compute_errors()
            if  self.plot:
                self._plot_results()

    # ----------------------------------------------------------
    # Post-processing
    # ----------------------------------------------------------

    def _compute_errors(self) -> None:

        if self.verbose > 1:
            logging.debug("Computing errors")

        self.upkf.history.compute_errors(
            self.upkf,
            ['xkp1'],
            ['Xkp1_update'],
            ['PXXkp1_update'],
            ['ikp1'],
            ['Skp1']
        )

    def _save_history(self) -> None:

        filepath = os.path.join(self.tracker_dir, "history_run_epfk_2.pkl")
        self.upkf.history.save_pickle(filepath)

        if self.verbose > 1:
            logging.info(f"History saved to {filepath}")

    def _plot_results(self) -> None:

        title = f"{self.model_name} filtered with UPKF"

        self.upkf.history.plot(
            title,
            list_param=["ykp1"],
            list_label=["Observations y"],
            list_covar=[None],
            window=WINDOW,
            basename=f"upkf_observations_{self.model_name}",
            show=False,
            base_dir=self.graph_dir
        )

        
# ==============================================================
# CLI layer
# ==============================================================

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Filter non linear data with UPKF"
    )

    addParseToParser(
        parser,
        ['nonLinearModelName', 'dataFileName', 'sigmaSet']
    )

    return parser.parse_args()


def main() -> None:
    args = parse_arguments()

    runner = NonLinearUPKFRunner_fromFile(
        model_name=args.nonLinearModelName,
        sigmaSet=args.sigmaSet,
        verbose=args.verbose,
        plot=args.plot,
        save_history=args.saveHistory,
        data_filename=args.dataFileName,
        N=args.N if hasattr(args, "N") else None,
    )

    runner.run()


if __name__ == "__main__":
    main()
