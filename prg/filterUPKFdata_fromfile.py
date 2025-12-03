#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import numpy as np

# non linear models 
from models.nonLinear import ModelFactoryNonLinear
# Linear models
from models.linear import ModelFactoryLinear
# A few utils functions that are used several times
from others.utils import compute_errors, file_data_generator
# Manage algorithms for the UPKF
from classes.UPKF import UPKF
# Manage algorithms for the UPKF
from classes.PKF import PKF
# Manage parameters for the UPKF
from classes.ParamNonLinear import ParamNonLinear
# Manage parameters for the UPKF
from classes.ParamLinear import ParamLinear

if __name__ == "__main__":
    """
        python prg/filterUPKFdata_fromfile.py
    """
    # ------------------------------------------------------------------
    # Constants
    # ------------------------------------------------------------------
    save_pickle = True
    verbose     = 0
    N = 10000
    
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
    # Test parameters Chose a linear or a non linear model by commenting
    # ------------------------------------------------------------------

    # Available Non linear models: 
    # ['x1_y1_cubique', 'x1_y1_ext_saturant', 'x1_y1_gordon', 'x1_y1_sinus', 'x2_y1_withRetroactionsOfObservations', 'x2_y1']
    model        = ModelFactoryNonLinear.create("x2_y1_withRetroactionsOfObservations")
    params       = model.get_params()
    dim_x, dim_y = params.pop('dim_x'), params.pop('dim_y')
    param        = ParamNonLinear(verbose, dim_x, dim_y, **params)
    
    # Available Linear models: 
    # ['A_mQ_x1_y1', 'A_mQ_x1_y1_VPgreaterThan1', 'A_mQ_x2_y2', 'A_mQ_x3_y1', 'Sigma_x1_y1', 'Sigma_x2_y2', 'Sigma_x3_y1']
    # model        = ModelFactoryLinear.create("A_mQ_x1_y1")
    # params       = model.get_params().copy()
    # dim_x, dim_y = params.pop('dim_x'), params.pop('dim_y')
    # param        = ParamLinear(verbose, dim_x, dim_y, **params)
    
    if verbose > 0:
        print(f'model={model}')
        param.summary()

    # ------------------------------------------------------------------
    # Let's go
    # ------------------------------------------------------------------

    #### ATTENTION Data dimensions in the file should be the same as model dimension above
    # datafile = 'dataUPKF_x2_y1_dimxy_2x1.parquet'
    # datafile = 'dataUPKF_x2_y1_dimxy_2x1.csv'
    # datafile = 'dataUPKF_x2_y1_dimy_1.csv'
    datafile = 'dataUPKF_x2_y1_withRetroactionsOfObservations_dimxy_2x1.csv'

    print("\nUPKF filtering with data generated from a file... ")

    upkf_2      = UPKF(param, save_pickle=save_pickle, verbose=verbose)
    filename    = os.path.join(datafile_dir, datafile)
    listeUPKF = upkf_2.process_N_data(N=None, data_generator=file_data_generator(filename, dim_x, dim_y, verbose))

    if save_pickle and upkf_2.history is not None:
        df = upkf_2.history.as_dataframe()
        if verbose > 0:
            print("\nExtract of the filtering with UPKF :")
            print(df.head())

        # print scoring
        if listeUPKF[0][0] is not None:
            ListeA = ['xkp1',           'xkp1']
            ListeB = ['Xkp1_predict',   'Xkp1_update']
            ListeC = ['PXXkp1_predict', 'PXXkp1_update']
            upkf_2.history.compute_errors(ListeA, ListeB, ListeC)

        # pickle storing and plots
        upkf_2.history.save_pickle(os.path.join(tracker_dir, f"history_run_upfk_2.pkl"))
        if listeUPKF[0][0] is not None:
            upkf_2.history.plot(list_param= ["xkp1", "Xkp1_update" ], \
                                list_label= ["X - Noisy", "X - Filtered"], \
                                window    = {'xmin': min(50, N), 'xmax': min(min(50, N)+50, N) }, \
                                basename  = 'upkf_2', show=False, base_dir=graph_dir)

