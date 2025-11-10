#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import numpy as np

# Linear models 
from models.linear import BaseModel, all_models
# # A few utils functions that are used several times
from others.Utils import rmse, file_data_generator
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
    
    # Available : ['A_mQ_x1_y1', 'A_mQ_x3_y1', 'Sigma_x1_y1', 'Sigma_x3_y1', 'A_mQ_x2_y2', 'Sigma_x2_y2', 'A_mQ_x1_y1_VPgreaterThan1']
    model_module = all_models['A_mQ_x3_y1']
    model = model_module.create_model()
    # print(f'model={model.info}')
    # print(f'model={model.get_params()}')
    
    params = model.get_params().copy()
    dim_x, dim_y = params.pop('dim_x'), params.pop('dim_y')
    param = ParamPKF(verbose, dim_x, dim_y, **params)
    if verbose > 0:
        param.summary()

    # ------------------------------------------------------------------
    # Let's go
    # ------------------------------------------------------------------

    #### ATTENTION Data dimensions in the file should be the same as model dimension above
    # datafile = 'dataPKF_Sigma_x3_y1_dimxy_3x1.parquet'
    # datafile = 'dataPKF_Sigma_x3_y1_dimxy_3x1.csv'
    # datafile = 'dataPKF_Sigma_x3_y1_dimy_1.csv'
    # datafile = 'dataPKF_Sigma_x3_y1_dimxy_3x1.csv'
    datafile = 'dataPKF_A_mQ_x3_y1_dimxy_3x1.csv'
    
    print("\nPKF filtering with data generated from a file... ")
    
    pkf_2      = PKF(param, sKey=sKey, save_pickle=save_pickle, verbose=verbose)
    filename   = os.path.join(datafile_dir, datafile)
    listePKF_2 = pkf_2.process_N_data(N=None, data_generator=file_data_generator(filename, dim_x, dim_y, verbose))
    # print(f'listePKF_2={listePKF_2[0:5]}')
    # print(listePKF_2[1][0].shape)
    
    if listePKF_2[1][0].shape != (0,1): # cela veut dire que l'on a une VT
        # on ne peut donc calculer les RMSE
        first_arrays  = np.vstack([t[0] for t in listePKF_2])[20:]
        third_arrays  = np.vstack([t[2] for t in listePKF_2])[20:]
        fourth_arrays = np.vstack([t[3] for t in listePKF_2])[20:]
        fith_arrays   = np.vstack([t[4] for t in listePKF_2])[20:]
        # Calcul du RMSE global
        print(f"RMSE (X, Esp[X] pred) : {rmse(first_arrays, third_arrays)}")
        print(f"RMSE (X, Esp[X]_math) : {rmse(first_arrays, fourth_arrays)}")
        print(f"RMSE (X, Esp[X]_phys) : {rmse(first_arrays, fith_arrays)}")

    if save_pickle and pkf_2.history is not None:
        df = pkf_2.history.as_dataframe()
        if verbose > 0:
            print("\nExtract of the resulting filtering with PKF :")
            print(df.head())
            # print(df.info())

        # pickle storing and plots
        pkf_2.history.save_pickle(os.path.join(tracker_dir, f"history_run_pfk_2.pkl"))
        if listePKF_2[1][0].shape != (0,1): # cela veut dire que l'on a une VT
            pkf_2.history.plot(list_param=["xkp1",             "Xkp1_update_math",                    "Xkp1_update_phys"], \
                               list_label=["X - Ground Truth", "X - Filtered (mathematical version)", "X - Filtered (physical version)"], \
                               fenetre   = {'xmin': min(50, N), 'xmax': min(min(50, N)+50, N) }, \
                               basename  ='pkf_2', \
                               show=False, base_dir=graph_dir)
        else:
            pkf_2.history.plot(list_param=["Xkp1_update_math",                    "Xkp1_update_phys"], \
                               list_label=["X - Filtered (mathematical version)", "X - Filtered (physical version)"], \
                               fenetre   = {'xmin': min(50, N), 'xmax': min(min(50, N)+50, N) }, \
                               basename  ='pkf_2', \
                               show=False, base_dir=graph_dir)
