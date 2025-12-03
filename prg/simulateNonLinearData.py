#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import pandas as pd
import numpy as np
# from pathlib import Path

# non linear models 
from models.nonLinear import ModelFactoryNonLinear
# A few utils functions that are used several times
from others.utils import save_dataframe_to_csv, data_to_dataframe
# Manage algorithms for the UPKF
from classes.UPKF import UPKF
# Parameters for PKF
from classes.ParamNonLinear import ParamNonLinear


if __name__ == "__main__":
    """
        python prg/simulateNonLinearData.py
    """
 
    # ------------------------------------------------------------------
    # Constants
    # ------------------------------------------------------------------
    save_pickle = False
    verbose       = 0
    N             = 10000 # > 20
    sKey          = 41 # Int or None (so that it is generated automatically)
    withoutX_True = False # If True : simulated X will not be stored in the file
    
    # ------------------------------------------------------------------
    # Output repo for data
    # ------------------------------------------------------------------
    base_dir     = os.path.join(".",      "data")
    datafile_dir = os.path.join(base_dir, "datafile")

    # ------------------------------------------------------------------
    # Test parameters
    # ------------------------------------------------------------------

    # Available non linear models:
    # ['x1_y1_cubique', 'x1_y1_ext_saturant', 'x1_y1_gordon', 'x1_y1_sinus', 'x2_y1', 'x2_y1_rapport', 'x2_y1_withRetroactionsOfObservations']
    model = ModelFactoryNonLinear.create("x2_y1_rapport")
    if verbose>0:
        print(f'model={model}, {model.MODEL_NAME}')
        print(f'model={model}')

    params       = model.get_params().copy()
    dim_x, dim_y = params.pop('dim_x'), params.pop('dim_y')
    param        = ParamNonLinear(verbose, dim_x, dim_y, **params)
    if verbose > 0:
        param.summary()
 
    # ------------------------------------------------------------------
    # Let's go
    # ------------------------------------------------------------------

    print("\nUPKF simulation")
    upkf = UPKF(param, sKey=sKey, save_pickle=save_pickle, verbose=verbose)
    
    # Simulate data with the simulator generator
    listData = upkf.simulate_N_data(N=N)

    # Save data as a dataframe using pandas
    df = data_to_dataframe(listData, dim_x, dim_y, withoutX_True=withoutX_True)
    filename = f"dataUPKF_{model.MODEL_NAME}_dimxy_{dim_x}x{dim_y}.csv"
    filepath = os.path.join(datafile_dir, filename)
    save_dataframe_to_csv(df, filepath)
    