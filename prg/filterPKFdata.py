#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import numpy as np

# Linear models 
from models.linear import ModelFactoryLinear
# A few utils functions that are used several times
from others.Utils import mse
# Manage algorithms for the PKF
from classes.PKF import PKF
# Manage parameters for the PKF
from classes.ParamPKF import ParamPKF


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
    # Test parameters for the Two ((A, mQ) or Sigma) parametrizations
    # ------------------------------------------------------------------
    
    # Available linear models:
    # ['A_mQ_x1_y1', 'A_mQ_x1_y1_VPgreaterThan1', 'A_mQ_x2_y2', 'A_mQ_x3_y1', 'Sigma_x1_y1', 'Sigma_x2_y2', 'Sigma_x3_y1']
    model        = ModelFactoryLinear.create("A_mQ_x1_y1")
    params       = model.get_params().copy()
    dim_x, dim_y = params.pop('dim_x'), params.pop('dim_y')
    param        = ParamPKF(verbose, dim_x, dim_y, **params)
    
    if verbose > 0:
        print(f'model={model}')
        param.summary()

    # ------------------------------------------------------------------
    # Let's go
    # ------------------------------------------------------------------

    print("\nPKF filtering with data generated from a PKF... ")
    pkf_1 = PKF(param, sKey=sKey, save_pickle=save_pickle, verbose=verbose)
    # Call with the default data simulator generator
    listePKF_1 = pkf_1.process_N_data(N=N)

    # Calcul du MSE entre le simulé et l'estimation math, et entre le simulé et l'estimation phys.
    first_arrays  = np.vstack([t[0] for t in listePKF_1])[20:]
    third_arrays  = np.vstack([t[2] for t in listePKF_1])[20:]
    fourth_arrays = np.vstack([t[3] for t in listePKF_1])[20:]
    fith_arrays   = np.vstack([t[4] for t in listePKF_1])[20:]
    # Calcul du MSE global
    print(f"MSE (X, Esp[X] pred) : {mse(first_arrays, third_arrays)}")
    print(f"MSE (X, Esp[X]_math) : {mse(first_arrays, fourth_arrays)}")
    print(f"MSE (X, Esp[X]_phys) : {mse(first_arrays, fith_arrays)}")
    
    if save_pickle and pkf_1.history is not None:
        df = pkf_1.history.as_dataframe()
        if verbose > 0:
            print("\nExtract of the resulting filtering with PKF :")
            print(df.head())

        # pickle storing and plots
        pkf_1.history.save_pickle(os.path.join(tracker_dir, f"history_run_pfk_1.pkl"))
        pkf_1.history.plot(list_param= ["xkp1",             "Xkp1_update_math",                    "Xkp1_update_phys"], \
                           list_label= ["X - Ground Truth", "X - Filtered (mathematical version)", "X - Filtered (physical version)"], \
                           window    = {'xmin': min(50, N), 'xmax': min(min(50, N)+50, N) }, \
                           basename  = 'pkf_1', \
                           show=False, base_dir=graph_dir)

