#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import numpy as np

# non linear models 
from models.nonLinear import ModelFactoryNonLinear
# A few utils functions that are used several times
from others.utils import compute_errors
# Manage algorithms for the UPKF
from classes.NonLinear_UPKF import NonLinear_UPKF
# Manage non parameters
from classes.ParamNonLinear import ParamNonLinear


if __name__ == "__main__":
    """
        python prg/filterUPKFdata.py
    """

    # ------------------------------------------------------------------
    # Constants
    # ------------------------------------------------------------------
    save_pickle = True
    verbose     = 0
    N           = 10000 # > 20
    sKey        = 41 # Int or None (so that it is generated automatically)
    
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
    # Test parameters
    # ------------------------------------------------------------------

    # Available Non linear models: 
    # ['x1_y1_cubique', 'x1_y1_ext_saturant', 'x1_y1_gordon', 'x1_y1_sinus', 'x2_y1_withRetroactionsOfObservations', 'x2_y1']
    model        = ModelFactoryNonLinear.create("x2_y1")
    params       = model.get_params()
    dim_x, dim_y = params.pop('dim_x'), params.pop('dim_y')
    param        = ParamNonLinear(verbose, dim_x, dim_y, **params)
    
    if verbose > 0:
        print(f'model={model}')
        param.summary()

    # ------------------------------------------------------------------
    # Let's go
    # ------------------------------------------------------------------

    print("\nUPKF filtering with data generated from a UPKF... ")
    upkf_1 = NonLinear_UPKF(param, sKey=sKey, save_pickle=save_pickle, verbose=verbose)
    listeUPKF = upkf_1.process_N_data(N=N)  # Call with the default data simulator generator

    if save_pickle and upkf_1.history is not None:
        df = upkf_1.history.as_dataframe()
        if verbose > 0:
            print("\nExtract of the resulting filtering with UPKF :")
            print(df.head())
            
        # print scoring
        ListeA = ['xkp1',           'xkp1']
        ListeB = ['Xkp1_predict',   'Xkp1_update']
        ListeC = ['PXXkp1_predict', 'PXXkp1_update']
        upkf_1.history.compute_errors(ListeA, ListeB, ListeC)

        # pickle storing and plots
        upkf_1.history.save_pickle(os.path.join(tracker_dir, f"history_run_upfk_1.pkl"))
        upkf_1.history.plot(list_param= ["xkp1",             "xkp1"  , "Xkp1_update"], \
                            list_label= ["X - Ground Truth", "X - Noisy", "X - Filtered"], \
                            window    = {'xmin': min(50, N), 'xmax': min(min(50, N)+50, N) }, \
                            basename  = 'upkf_1', show=False, base_dir=graph_dir)

