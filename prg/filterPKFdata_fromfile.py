#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import numpy as np

# Linear models
from models.linear import ModelFactoryLinear
# A few utils functions that are used several times
from others.utils import mse, file_data_generator
# Manage algorithms for the PKF
from classes.PKF import PKF
# Manage parameters for the PKF
from classes.ParamPKF import ParamPKF

if __name__ == "__main__":
    """
        python prg/filterPKFdata_fromfile.py
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
    model        = ModelFactoryLinear.create("Sigma_x3_y1")
    params       = model.get_params().copy()
    dim_x, dim_y = params.pop('dim_x'), params.pop('dim_y')
    param        = ParamPKF(verbose, dim_x, dim_y, **params)
    
    if verbose > 0:
        print(f'model={model}')
        param.summary()

    # ------------------------------------------------------------------
    # Let's go
    # ------------------------------------------------------------------

    #### ATTENTION Data dimensions in the file should be the same as model dimension above
    # datafile = 'dataPKF_Sigma_x3_y1_dimxy_3x1.parquet'
    # datafile = 'dataPKF_Sigma_x3_y1_dimxy_3x1.csv'
    datafile = 'dataPKF_Sigma_x3_y1_dimxy_3x1.csv'
    # datafile = 'dataPKF_Sigma_x3_y1_dimxy_3x1.csv'
    # datafile = 'dataPKF_A_mQ_x1_y1_dimxy_1x1.csv'
    
    print("\nPKF filtering with data generated from a file... ")
    
    pkf_2    = PKF(param, sKey=sKey, save_pickle=save_pickle, verbose=verbose)
    filename = os.path.join(datafile_dir, datafile)
    listePKF = pkf_2.process_N_data(N=None, data_generator=file_data_generator(filename, dim_x, dim_y, verbose))

    if save_pickle and pkf_2.history is not None:
        df = pkf_2.history.as_dataframe()
        if verbose > 0:
            print("\nExtract of the filtering with PKF :")
            print(df.head())
            
        # print scoring
        ListeA = ['xkp1_true',      'xkp1_true',          'xkp1',           'xkp1']
        ListeB = ['Xkp1_predict',   'Xkp1_update_math',   'Xkp1_predict',   'Xkp1_update_math']
        ListeC = ['PXXkp1_predict', 'PXXkp1_update_math', 'PXXkp1_predict', 'PXXkp1_update_math']
        pkf_2.history.compute_errors(ListeA, ListeB, ListeC)

        # pickle storing and plots
        pkf_2.history.save_pickle(os.path.join(tracker_dir, f"history_run_pfk_2.pkl"))
        if listePKF[0][0] is not None: # We got a ground truth
            # list_param= ["xkp1_true",        "xkp1",      "Xkp1_update_math",                    "Xkp1_update_phys"], \
            # list_label= ["X - Ground Truth", "X - Noisy", "X - Filtered (mathematical version)", "X - Filtered (physical version)"], \
            pkf_2.history.plot(list_param= ["xkp1_true", "xkp1", "Xkp1_update_math" ], \
                               list_label= ["X - Ground Truth", "X - Noisy", "X - Filtered (mathematical version)"], \
                               window    = {'xmin': min(50, N), 'xmax': min(min(50, N)+50, N) }, \
                               basename  = 'pkf_2', \
                               show=False, base_dir=graph_dir)
        else:
            pkf_2.history.plot(list_param= ["xkp1", "Xkp1_update_math"], \
                               list_label= ["X - Noisy", "X - Filtered (mathematical version)"], \
                               window    = {'xmin': min(50, N), 'xmax': min(min(50, N)+50, N) }, \
                               basename  = 'pkf_2', \
                               show=False, base_dir=graph_dir)
