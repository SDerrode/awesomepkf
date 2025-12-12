#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import numpy as np

import os
import numpy as np

# non linear models 
from models.nonLinear import ModelFactoryNonLinear
# Linear models
from models.linear import ModelFactoryLinear
# A few utils functions that are used several times
from others.utils import compute_errors, file_data_generator
# Manage algorithms for non linear EPKF
from classes.NonLinear_EPKF import NonLinear_EPKF
# Manage algorithms for the PKF
from classes.Linear_PKF import Linear_PKF
# Manage non linear and linear parameters
from classes.ParamNonLinear import ParamNonLinear
from classes.ParamLinear import ParamLinear

if __name__ == "__main__":
    """
        python prg/filterEPKFdata_fromfile.py
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
    # Test parameters Chose a linear or a non linear model by commenting
    # ------------------------------------------------------------------

    # Available non linear models:
    # ['x1_y1_cubique', 'x1_y1_ext_saturant', 'x1_y1_gordon', 'x1_y1_sinus', 'x1_y1_withRetroactions', 'x2_y1', 'x2_y1_rapport', 'x2_y1_withRetroactionsOfObservations']
    model        = ModelFactoryNonLinear.create("x1_y1_withRetroactions")
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
    # datafile = 'dataNonLinear_x2_y1_dimxy_2x1.parquet'
    # datafile = 'dataNonLinear_x2_y1_dimxy_2x1.csv'
    # datafile = 'dataNonLinear_x2_y1_dimxy_2x1.csv'
    # datafile = 'dataNonLinear_x2_y1_withRetroactionsOfObservations_dimxy_2x1.csv'
    datafile = 'dataNonLinear_x1_y1_withRetroactions_dimxy_1x1.csv'

    print("\nEPKF filtering with data generated from a file with data... ")

    epkf_2      = NonLinear_EPKF(param, save_pickle=save_pickle, verbose=verbose)
    filename    = os.path.join(datafile_dir, datafile)
    listeEPKF = epkf_2.process_N_data(N=None, data_generator=file_data_generator(filename, dim_x, dim_y, verbose))
    N = listeEPKF[-1][0] + 1

    if save_pickle and epkf_2.history is not None:
        df = epkf_2.history.as_dataframe()
        if verbose > 0:
            print("\nExtract of the filtering with EPKF :")
            print(df.head())

        # print scoring
        if listeEPKF[0][1] is not None:
            ListeA = ['xkp1',           'xkp1']
            ListeB = ['Xkp1_predict',   'Xkp1_update']
            ListeC = ['PXXkp1_predict', 'PXXkp1_update']
            epkf_2.history.compute_errors(ListeA, ListeB, ListeC)

        # pickle storing and plots
        epkf_2.history.save_pickle(os.path.join(tracker_dir, f"history_run_upfk_2.pkl"))
        if listeEPKF[0][1] is not None:
            title = f"'{model.MODEL_NAME}' model data filtered with EPKF"
            epkf_2.history.plot(title, 
                            list_param= ["ykp1"], \
                            list_label= ["Observations y"], \
                            # window    = {'xmin': min(20, N), 'xmax': min(min(20, N)+100, N) }, \
                            window    = {'xmin':20, 'xmax': 120 }, \
                            basename  = f'epkf_2_{model.MODEL_NAME}_observations', show=False, base_dir=graph_dir)
            epkf_2.history.plot(title, 
                            list_param= ["xkp1"  , "Xkp1_update"], \
                            list_label= ["x true", "x estimated"], \
                            # window    = {'xmin': min(20, N), 'xmax': min(min(20, N)+100, N) }, \
                            window    = {'xmin':20, 'xmax': 120 }, \
                            basename  = f'epkf_2_{model.MODEL_NAME}', show=False, base_dir=graph_dir)

