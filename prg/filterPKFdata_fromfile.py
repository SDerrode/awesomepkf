#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import numpy as np

# Linear models
from models.linear import ModelFactoryLinear
# A few utils functions that are used several times
from others.utils import compute_errors, file_data_generator
# Manage algorithms for the PKF
from classes.Linear_PKF import Linear_PKF
# Manage Linear parameters
from classes.ParamLinear import ParamLinear

if __name__ == "__main__":
    """
        python prg/filterPKFdata_fromfile.py
    """
    # ------------------------------------------------------------------
    # Constants
    # ------------------------------------------------------------------
    save_pickle = True
    verbose     = 0

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

    #### ATTENTION Data dimensions in the file should be the same as model dimension above
    # datafile = 'dataLinear_Sigma_x3_y1_dimxy_3x1.parquet'
    # datafile = 'dataLinear_Sigma_x3_y1_dimxy_3x1.csv'
    # datafile = 'dataLinear_Sigma_x3_y1_dimxy_3x1.csv'
    datafile = 'dataLinear_A_mQ_x3_y1_dimxy_3x1.csv'
    # datafile = 'dataLinear_A_mQ_x1_y1_dimxy_1x1.csv'
    
    print("\nPKF filtering with data generated from a file... ")
    
    pkf_2    = Linear_PKF(param, save_pickle=save_pickle, verbose=verbose)
    filename = os.path.join(datafile_dir, datafile)
    listePKF = pkf_2.process_N_data(N=None, data_generator=file_data_generator(filename, dim_x, dim_y, verbose))
    N        = listePKF[-1][0]+1

    if save_pickle and pkf_2.history is not None:
        df = pkf_2.history.as_dataframe()
        if verbose > 0:
            print("\nExtract of the filtering with PKF :")
            print(df.head())
            
        # print scoring
        if listePKF[0][1] is not None:
            ListeA = ['xkp1',           'xkp1']
            ListeB = ['Xkp1_predict',   'Xkp1_update_math']
            ListeC = ['PXXkp1_predict', 'PXXkp1_update_math']
            pkf_2.history.compute_errors(ListeA, ListeB, ListeC)

        # pickle storing and plots
        pkf_2.history.save_pickle(os.path.join(tracker_dir, f"history_run_pfk_2.pkl"))
        if listePKF[0][1] is not None:
            pkf_2.history.plot( list_param = ["xkp1", "Xkp1_update_math" ], \
                                list_label = ["X - Noisy", "X - Filtered (mathematical version)"], \
                                window     = {'xmin': min(50, N), 'xmax': min(min(50, N)+50, N) }, \
                                basename   = 'pkf_2', \
                                show=False, base_dir=graph_dir)
