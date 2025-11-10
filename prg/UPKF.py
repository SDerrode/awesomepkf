#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Module UPKF
####################################################################
Unscented Pairwise Kalman filter (UPKF) implementation
####################################################################
"""

from __future__ import annotations

import os
import math
import logging
import warnings
from typing import Generator, Optional, Tuple

import numpy as np

# non linear models 
from models.nonLinear import ModelFactory
# A few utils functions that are used several times
from others.Utils import rmse, file_data_generator, check_consistency, check_equality
# Manage parameters for the UPKF
from classes.ParamUPKF import ParamUPKF
# Keep trace of execution (all parameters at all iterations)
from classes.HistoryTracker import HistoryTracker
# To manage the seed for random generation
from classes.SeedGenerator import SeedGenerator

# ----------------------------------------------------------------------
# Configuration du logging global
# ----------------------------------------------------------------------
logging.basicConfig(format="[%(levelname)s] %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)

class UPKF:
    """Implementation of UPKF."""

    def __init__(
        self,
        param: ParamUPKF,
        sKey: Optional[int] = None,
        save_pickle: bool = False,
        verbose: int = 0
    ) -> None:
        
        if __debug__:
            if not isinstance(param, ParamUPKF):
                raise TypeError("param must be an object from class ParamUPKF")
            if not ((isinstance(sKey, int) and sKey > 0) or sKey is None):
                raise ValueError("sKey must be None or a number>0")
            if not isinstance(save_pickle, bool):
                raise TypeError("save_pickle must be a boolean")
            if verbose not in [0, 1, 2]:
                raise ValueError("verbose must be 0, 1 or 2")

        self.param: ParamUPKF = param
        self.dt: int = 1
        self.verbose: int = verbose
        self._seed_gen: SeedGenerator = SeedGenerator(sKey)

        # Shortcuts
        self.dim_x: int
        self.dim_y: int
        self.dim_xy: int
        self.dim_x, self.dim_y, self.dim_xy = param.dim_x, param.dim_y, param.dim_xy

        # Mean weights Wm, and correlation weights Wc
        self.Wm: np.ndarray = np.full(2 * self.dim_x + 1, 1. / (2. * (self.dim_x + param.lambda_)))
        self.Wc: np.ndarray = np.copy(self.Wm)
        self.Wm[0] = param.lambda_ / (self.dim_x + param.lambda_)
        self.Wc[0] = param.lambda_ / (self.dim_x + param.lambda_) + (1. - param.alpha**2 + param.beta)

        # HistoryTracker optional
        self.save_pickle: bool = save_pickle
        self._history: Optional[HistoryTracker] = HistoryTracker() if save_pickle else None

        # Logging
        self._set_log_level()
        if __debug__ and self.verbose >= 1:
            logger.info(f"[UPKF] Init with sKey={sKey}, verbose={verbose}, save_pickle={save_pickle}")

    # ------------------------------------------------------------------
    # Loger configuration according to verbose
    # ------------------------------------------------------------------
    def _set_log_level(self) -> None:
        if self.verbose == 0:
            logger.setLevel(logging.WARNING)
        elif self.verbose == 1:
            logger.setLevel(logging.INFO)
        else:
            logger.setLevel(logging.DEBUG)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------
    @property
    def seed_gen(self) -> int:
        """Return generator seed."""
        return self._seed_gen.seed

    @property
    def history(self) -> Optional[HistoryTracker]:
        """Return HistoryTracker object if save_pickle==True, else None."""
        return self._history

    # ------------------------------------------------------------------
    # Generators
    # ------------------------------------------------------------------
    def _data_generation(self, N: Optional[int] = None) -> Generator[Tuple[int, np.ndarray], None, None]:
        """
        Generator for the simulation of Z_{k+1} = A * Z_k + W_{k+1},
        with W_{k+1} ~ N(0, mQ) and Z_1 ~ N(0, Q1).
        """
        z00, Pz00, g, mQ = self.param._z00, self.param._Pz00, self.param.g, self.param.mQ
        k: int = 0
        Zkp1_simul: np.ndarray = self._seed_gen.rng.multivariate_normal(mean=z00.T.flatten(), cov=Pz00).reshape(-1,1)
        yield k, np.split(Zkp1_simul, [self.dim_x])

        zerosvector: np.ndarray = np.zeros(self.dim_xy)
        while N is None or k < N:
            k += 1
            Zkp1_simul = g(Zkp1_simul, self._seed_gen.rng.multivariate_normal(mean=zerosvector, cov=mQ).reshape(-1,1), self.dt)
            yield k, np.split(Zkp1_simul, [self.dim_x])

    def _sigma_points(self, x: np.ndarray, P: np.ndarray) -> np.ndarray:
        """Generate the 2*dim_x+1 sigma points around x"""
        A = np.linalg.cholesky(P)
        sigma: list[np.ndarray] = [x]
        for i in range(self.dim_x):
            sigma.append(x + self.param.gamma * A[:, i].reshape(-1,1))
            sigma.append(x - self.param.gamma * A[:, i].reshape(-1,1))
        return np.array(sigma)

    def process_upkf(self, N: Optional[int] = None, data_generator: Optional[Generator] = None) -> Generator:
        """
        Generator of UPKF filter using optional data generator.
        """
        if __debug__:
            if not ((isinstance(N, int) and N > 0) or N is None):
                raise ValueError("N must be None or a number >0")

        generator = data_generator if data_generator is not None else self._data_generation()
        g, mQ, dim_x = self.param.g, self.param.mQ, self.dim_x

        # First sample
        k, (xkp1, ykp1) = next(generator)
        Xkp1_update: np.ndarray = self.param.z00[0:self.dim_x]
        PXXkp1_update: np.ndarray = self.param.Pz00[0:self.dim_x, 0:self.dim_x]
        check_consistency(PXXkp1_update=PXXkp1_update)

        Xkp1_predict: np.ndarray = np.zeros((self.dim_x, 1))
        if self.save_pickle and self._history is not None:
            self._history.record(iter=k,
                                 xkp1=xkp1.copy(),
                                 ykp1=ykp1.copy(),
                                 Xkp1_predict=Xkp1_predict,
                                 PXXkp1_predict=np.eye(self.dim_x),
                                 Xkp1_update=Xkp1_update.copy(),
                                 PXXkp1_update=PXXkp1_update.copy())

        yield xkp1, ykp1, Xkp1_predict, Xkp1_update

        while N is None or k < N:
            # Sigma points
            sigma = self._sigma_points(Xkp1_update, PXXkp1_update)
            sigma_propag: list[np.ndarray] = [g(np.vstack((e, ykp1)), np.zeros((self.dim_xy,1)), self.dt) for e in sigma]

            # Prediction
            Zkp1_predict: np.ndarray = np.sum(self.Wm[:, None, None] * sigma_propag, axis=0)
            Xkp1_predict, Ykp1_predict = np.split(Zkp1_predict, [dim_x])
            Pkp1_predict: np.ndarray = mQ.copy()
            for i in range(2 * dim_x + 1):
                temp = sigma_propag[i] - Zkp1_predict
                Pkp1_predict += self.Wc[i] * np.outer(temp, temp)

            M_top, M_bottom = np.vsplit(Pkp1_predict, [dim_x])
            PXXkp1_predict, PXYkp1_predict = np.hsplit(M_top, [dim_x])
            PYXkp1_predict, PYYkp1_predict = np.hsplit(M_bottom, [dim_x])

            try:
                k, (xkp1, ykp1) = next(generator)
            except StopIteration:
                return

            Xkp1_update = Xkp1_predict + (PXYkp1_predict @ np.linalg.inv(PYYkp1_predict)) @ (ykp1 - Ykp1_predict)
            PXXkp1_update = PXXkp1_predict - PXYkp1_predict @ np.linalg.inv(PYYkp1_predict) @ PYXkp1_predict

            check_consistency(Pkp1_predict=Pkp1_predict, PXXkp1_update=PXXkp1_update)

            if self.save_pickle and self._history is not None:
                self._history.record(iter=k,
                                     xkp1=xkp1.copy(),
                                     ykp1=ykp1.copy(),
                                     Xkp1_predict=Xkp1_predict,
                                     PXXkp1_predict=PXXkp1_predict.copy(),
                                     Xkp1_update=Xkp1_update.copy(),
                                     PXXkp1_update=PXXkp1_update.copy())

            yield xkp1, ykp1, Xkp1_predict, Xkp1_update

    def process_N_data(self, N: Optional[int], data_generator: Optional[Generator] = None) -> list:
        return list(self.process_upkf(N=N, data_generator=data_generator))

    def simulate_N_data(self, N):
        return list(self._data_generation(N))

if __name__ == "__main__":
    """
        python prg/UPKF.py
    """
    
    import time
    start = time.perf_counter()


    # ------------------------------------------------------------------
    # Constants
    # ------------------------------------------------------------------
    save_pickle = True
    verbose     = 0
    N           = 20000 # > 20
    
    # ------------------------------------------------------------------
    # Output repo for data, traces and plots
    # ------------------------------------------------------------------
    base_dir     = os.path.join(".",      "data")
    tracker_dir  = os.path.join(base_dir, "historyTracker")
    datafile_dir = os.path.join(base_dir, "datafile")
    graph_dir    = os.path.join(base_dir, "plot")
    os.makedirs(tracker_dir, exist_ok=True)
    os.makedirs(graph_dir,   exist_ok=True)

    # ------------------------------------------------------------------
    # Test parameters
    # ------------------------------------------------------------------

    # Available : ['x1_y1_cubique', 'x1_y1_ext_saturant', 'x1_y1_gordon', 'x1_y1_sinus', 'x2_y1_withRetroactionsOfObservations', 'x2_y1']
    model = ModelFactory.create("x2_y1")
    if verbose>0:
        print(f'model={model}')

    params = model.get_params()
    param = ParamUPKF(verbose, **params)
    if verbose > 0:
        param.summary()

    # ------------------------------------------------------------------
    # Let's go
    # ------------------------------------------------------------------

    print("\nUPKF filtering with data generated from a UPKF... ")
    sKey   = None
    upkf_1 = UPKF(param, sKey=sKey, save_pickle=save_pickle, verbose=verbose)
    listeUPKF_1 = upkf_1.process_N_data(N=N)  # Call with the default data simulator generator

    # RMSE between simulated and the predicted and filtered
    first_arrays  = np.vstack([t[0] for t in listeUPKF_1])[20:]
    third_arrays  = np.vstack([t[2] for t in listeUPKF_1])[20:]
    fourth_arrays = np.vstack([t[3] for t in listeUPKF_1])[20:]
    print(f"RMSE (X, Esp[X] pred) : {rmse(first_arrays, third_arrays)}")
    print(f"RMSE (X, Esp[X] filt) : {rmse(first_arrays, fourth_arrays)}")
    
    if save_pickle and upkf_1.history is not None:
        df = upkf_1.history.as_dataframe()
        if verbose > 0:
            print("\nExtract of the resulting filtering with UPKF :")
            print(df.head())
            # print(df.info())

        # pickle storing and plots
        upkf_1.history.save_pickle(os.path.join(tracker_dir, f"history_run_upfk_1.pkl"))
        upkf_1.history.plot(list_param=["xkp1",             "Xkp1_predict"  , "Xkp1_update"], \
                            list_label=["X - Ground Truth", "X - Predicted", "X - Filtered"], \
                            basename='upkf_1', show=False, base_dir=graph_dir)

    # datafile = 'data_dim2x2.parquet'
    # #datafile = 'data_dim1x1.csv'
    # print("\nUPKF filtering with data generated from a file... ")
    # upkf_2 = UPKF(param, save_pickle=save_pickle, verbose=verbose)
    # data_generator=file_data_generator(os.path.join(datafile_dir, datafile), dim_x, verbose)
    # listeUPKF_2 = upkf_2.process_N_data(N=None, data_generator=data_generator) # Call with a file as data generator
    # # print(f'listeUPKF_2={listeUPKF_2}')

    # if save_pickle and upkf_2.history is not None:
    #     df = upkf_2.history.as_dataframe()
    #     if verbose > 0:
    #         print("\nExtract of the resulting filtering with UPKF :")
    #         print(df.head())
    #         # print(df.info())

    #     # pickle storing and plots
    #     upkf_2.history.save_pickle(os.path.join(tracker_dir, f"history_run_upfk_2.pkl"))
    #     upkf_2.history.plot(list_param=["xkp1",             "Xkp1_predict"  , "Xkp1_update"], \
    #                         list_label=["X - Ground Truth", "X - Predicted", "X - Filtered"], \
    #                         basename='upkf_2', show=False, base_dir=graph_dir)


    elapsed = time.perf_counter() - start
    print(f"Durée : {elapsed:.6f} s")
    