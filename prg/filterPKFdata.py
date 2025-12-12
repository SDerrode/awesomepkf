#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import numpy as np

# Linear models 
from models.linear import ModelFactoryLinear
# A few utils functions that are used several times
from others.utils import compute_errors
# Manage algorithms for the PKF
from classes.Linear_PKF import Linear_PKF
# Manage parameters for the PKF
from classes.ParamLinear import ParamLinear


if __name__ == "__main__":
    """
        python prg/filterPKFdata.py
    """
    # ------------------------------------------------------------------
    # Constants
    # ------------------------------------------------------------------
    save_pickle = True
    verbose     = 0
    N           = 10000 # > 20
    sKey        = 303 # Int or None (so that it is generated automatically)
    
    # ------------------------------------------------------------------
    # Output repo for data, traces and plots
    # ------------------------------------------------------------------
    base_dir     = os.path.join(".",      "data")
    tracker_dir  = os.path.join(base_dir, "historyTracker")
    datafile_dir = os.path.join(base_dir, "datafile")
    graph_dir    = os.path.join(base_dir, "plot")
    os.makedirs(tracker_dir,  exist_ok=True)
    os.makedirs(datafile_dir, exist_ok=True)
    os.makedirs(graph_dir,    exist_ok=True)

    # ------------------------------------------------------------------
    # Test parameters for the Two ((A, mQ) or Sigma) parametrizations
    # ------------------------------------------------------------------
    
    # Available linear models:
    # ['A_mQ_x1_y1', 'A_mQ_x1_y1_VPgreaterThan1', 'A_mQ_x2_y2', 'A_mQ_x3_y1', 'Sigma_x1_y1', 'Sigma_x2_y2', 'Sigma_x3_y1']
    model        = ModelFactoryLinear.create("Sigma_x3_y1")
    params       = model.get_params().copy()
    dim_x, dim_y = params.pop('dim_x'), params.pop('dim_y')
    param        = ParamLinear(verbose, dim_x, dim_y, **params)
    
    if verbose > 0:
        print(f'model={model}')
        param.summary()

    # ------------------------------------------------------------------
    # Let's go
    # ------------------------------------------------------------------

    print("\nPKF filtering with data generated from a linear model...")
    pkf_1   = Linear_PKF(param, sKey=sKey, save_pickle=save_pickle, verbose=verbose)
    # Call with the default data simulator generator
    listePKF = pkf_1.process_N_data(N=N)

    if save_pickle and pkf_1.history is not None:
        df = pkf_1.history.as_dataframe()
        if verbose > 0:
            print("\nExtract of the resulting filtering with PKF :")
            print(df.head())

        # print scoring
        ListeA = ['xkp1',           'xkp1']
        ListeB = ['Xkp1_predict',   'Xkp1_update_math']
        ListeC = ['PXXkp1_predict', 'PXXkp1_update_math']
        pkf_1.history.compute_errors(ListeA, ListeB, ListeC)

        # pickle storing and plots
        pkf_1.history.save_pickle(os.path.join(tracker_dir, f"history_run_pfk_1.pkl"))
        title = f"'{model.MODEL_NAME}' model data filtered with PKF"
        pkf_1.history.plot(title, 
                            list_param= ["ykp1"], \
                            list_label= ["Observations y"], \
                            # window    = {'xmin': min(20, N), 'xmax': min(min(20, N)+100, N) }, \
                            window    = {'xmin':20, 'xmax': 120 }, \
                            basename  = f'pkf_1_{model.MODEL_NAME}_observations', show=False, base_dir=graph_dir)
        pkf_1.history.plot(title, 
                            list_param= ["xkp1"  , "Xkp1_update_math"], \
                            list_label= ["x true", "x estimated"], \
                            # window    = {'xmin': min(20, N), 'xmax': min(min(20, N)+100, N) }, \
                            window    = {'xmin':20, 'xmax': 120 }, \
                            basename  = f'pkf_1_{model.MODEL_NAME}', show=False, base_dir=graph_dir)

