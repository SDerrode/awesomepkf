#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import pandas as pd
import numpy as np
from pathlib import Path

# A few utils functions that are used several times
from others.Utils import save_dataframe_to_csv, data_to_dataframe
# PKF class
from UPKF import UPKF
# Parameters for PKF
from classes.ParamUPKF import ParamUPKF
# non linear models 
from models.nonLinear import ModelFactory

if __name__ == "__main__":
    """
        python prg/simulateUPKFdata.py
    """
 
    # ------------------------------------------------------------------
    # Constants
    # ------------------------------------------------------------------
    save_pickle = False
    verbose     = 1
    N           = 10000 # > 20
    sKey        = 41 # Int or None (so that it is generated automatically)
    
    # ------------------------------------------------------------------
    # Output repo for data
    # ------------------------------------------------------------------
    base_dir     = os.path.join(".",      "data")
    datafile_dir = os.path.join(base_dir, "datafile")

    # ------------------------------------------------------------------
    # Test parameters
    # ------------------------------------------------------------------

    # Available : ['x1_y1_cubique', 'x1_y1_ext_saturant', 'x1_y1_gordon', 'x1_y1_sinus', 'x2_y1_withRetroactionsOfObservations', 'x2_y1']
    model = ModelFactory.create("x2_y1")
    if verbose>0:
        print(f'model={model}, {model.MODEL_NAME}')
        print(f'model={model}')

    params = model.get_params()
    dim_x, dim_y = params['dim_x'], params['dim_y']
    param = ParamUPKF(verbose, **params)
    if verbose > 0:
        param.summary()
 
    # ------------------------------------------------------------------
    # Let's go
    # ------------------------------------------------------------------

    withoutX = False # If True : les données X simulées ne seront pas enregistrées dans le fichier

    print("\nUPKF simulation")
    upkf = UPKF(param, sKey=sKey, save_pickle=save_pickle, verbose=verbose)
    
    # Simulate data with the simulator generator
    listData = upkf.simulate_N_data(N=N)
    # print(f'listData={listData}')

    # Save data as a dataframe using pandas
    df       = data_to_dataframe(listData, dim_x, dim_y, withoutX=withoutX)
    if withoutX == True:
        filename = f"dataUPKF_{model.MODEL_NAME}_dimy_{dim_y}.csv"
    else:
        filename = f"dataUPKF_{model.MODEL_NAME}_dimxy_{dim_x}x{dim_y}.csv"
    filepath = os.path.join(datafile_dir, filename)
    save_dataframe_to_csv(df, filepath)