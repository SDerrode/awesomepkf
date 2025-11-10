#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import pandas as pd
import numpy as np
#from pathlib import Path

# A few utils functions that are used several times
from others.Utils import save_dataframe_to_csv, data_to_dataframe
# Manage algorithms for the PKF
from classes.PKF import PKF
# Parameters for PKF
from classes.ParamPKF import ParamPKF
# Linear models 
from models.linear import BaseModel, all_models

if __name__ == "__main__":
    """
        python prg/simulatePKFdata.py
    """
    
    # ------------------------------------------------------------------
    # Constants
    # ------------------------------------------------------------------
    save_pickle = False
    verbose     = 0
    N           = 10000 # > 20
    sKey        = 41 # Int or None (so that it is generated automatically)
    
    # ------------------------------------------------------------------
    # Output repo for data
    # ------------------------------------------------------------------
    base_dir     = os.path.join(".",      "data")
    datafile_dir = os.path.join(base_dir, "datafile")

    # ------------------------------------------------------------------
    # Test parameters for the Two ((A, mQ) or Sigma) parametrizations
    # ------------------------------------------------------------------
    
    # Available : ['A_mQ_x1_y1', 'A_mQ_x3_y1', 'Sigma_x1_y1', 'Sigma_x3_y1', 'A_mQ_x2_y2', 'Sigma_x2_y2', 'A_mQ_x1_y1_VPgreaterThan1']
    model_module = all_models['A_mQ_x3_y1']
    model        = model_module.create_model()
    print(f'model={model_module.MODEL_NAME}')
    # print(f'model={model.get_params()}')
    
    params = model.get_params().copy()
    dim_x, dim_y = params.pop('dim_x'), params.pop('dim_y')
    param = ParamPKF(verbose, dim_x, dim_y, **params)
    if verbose > 0:
        param.summary()

    # ------------------------------------------------------------------
    # Let's go
    # ------------------------------------------------------------------
    
    withoutX = False # If True : les données X simulées ne seront pas enregistrées dans le fichier

    print("\nPKF simulation")
    pkf = PKF(param, sKey=sKey, save_pickle=save_pickle, verbose=verbose)
    
    # Simulate data with the simulator generator
    listData = pkf.simulate_N_data(N=N)
    # print(f'listData={listData}')
    
    # Save data as a dataframe using pandas
    df       = data_to_dataframe(listData, dim_x, dim_y, withoutX=withoutX)
    if withoutX == True:
        filename = f"dataPKF_{model_module.MODEL_NAME}_dimy_{dim_y}.csv"
    else:
        filename = f"dataPKF_{model_module.MODEL_NAME}_dimxy_{dim_x}x{dim_y}.csv"
    filepath = os.path.join(datafile_dir, filename)
    save_dataframe_to_csv(df, filepath)
    