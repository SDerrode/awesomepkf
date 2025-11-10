#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import numpy as np

# non linear models 
from models.nonLinear import ModelFactoryNonLinear
# A few utils functions that are used several times
from others.Utils import rmse
# Manage algorithms for the UPKF
from classes.UPKF import UPKF
# Manage parameters for the UPKF
from classes.ParamUPKF import ParamUPKF


if __name__ == "__main__":
    """
        python prg/filterUPKFdata.py
    """

    import time
    start = time.perf_counter()

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

    # Available : ['x1_y1_cubique', 'x1_y1_ext_saturant', 'x1_y1_gordon', 'x1_y1_sinus', 'x2_y1_withRetroactionsOfObservations', 'x2_y1']
    model = ModelFactoryNonLinear.create("x2_y1")
    if verbose>0:
        print(f'model={model}')

    params = model.get_params()
    dim_x, dim_y = params['dim_x'], params['dim_y']
    param = ParamUPKF(verbose, **params)
    if verbose > 0:
        param.summary()

    # ------------------------------------------------------------------
    # Let's go
    # ------------------------------------------------------------------

    print("\nUPKF filtering with data generated from a UPKF... ")
    upkf_1 = UPKF(param, sKey=sKey, save_pickle=save_pickle, verbose=verbose)
    listeUPKF_1 = upkf_1.process_N_data(N=N)  # Call with the default data simulator generator
    # print(listeUPKF_1[0:6])
    # exit(1)

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

        # pickle storing and plots
        upkf_1.history.save_pickle(os.path.join(tracker_dir, f"history_run_upfk_1.pkl"))
        upkf_1.history.plot(list_param=["xkp1",             "Xkp1_predict"  , "Xkp1_update"], \
                            list_label=["X - Ground Truth", "X - Predicted", "X - Filtered"], \
                            fenetre   = {'xmin': min(50, N), 'xmax': min(min(50, N)+50, N) }, \
                            basename='upkf_1', show=False, base_dir=graph_dir)


    elapsed = time.perf_counter() - start
    print(f"Durée : {elapsed:.6f} s")
    