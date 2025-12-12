#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import pandas as pd
import numpy as np
#from pathlib import Path

# Linear models
from models.linear import ModelFactoryLinear
# A few utils functions that are used several times
from others.utils import save_dataframe_to_csv, data_to_dataframe
# Manage algorithms for the PKF
from classes.Linear_PKF import Linear_PKF
# Parameters linear model
from classes.ParamLinear import ParamLinear
# Parser d'options
from others.parser    import *

if __name__ == "__main__":
    """
    USAGES:
        python prg/simulateLinearData.py
        python prg/simulateLinearData.py --verbose 0 --linearModelName A_mQ_x1_y1 --sKey 303 --dataFileName test.csv
    """
    
    # ------------------------------------------------------------------
    # Constants (default value) - Parser
    # ------------------------------------------------------------------

    parser = argparse.ArgumentParser(description='Simulate linear data')
    addParseToParser(parser, ['linearModelName', 'dataFileName', 'N', 'sKey', 'withoutX'])
    args   = parser.parse_args()
    
    traceplot       = args.traceplot
    verbose         = args.verbose
    withoutX        = args.withoutX # If True : true X will not be stored in the file
    N               = args.N
    sKey            = args.sKey   # Int>0 or None (so that it is generated automatically)
    linearModelName = args.linearModelName
    dataFileName    = args.dataFileName
    if dataFileName == None:
        dataFileName = f"dataLinear_{linearModelName}.csv"
    if sKey is not None and sKey < 0:
        parser.error("sKey must be >= 0")
    if N < 200:
        parser.error("N must be >= 200")
    # exit(1)
    
    # ------------------------------------------------------------------
    # Output repo for data
    # ------------------------------------------------------------------
    base_dir     = os.path.join(".",      "data")
    datafile_dir = os.path.join(base_dir, "datafile")

    # ------------------------------------------------------------------
    # Test parameters
    # ------------------------------------------------------------------
    
    # Available linear models: 
    model = ModelFactoryLinear.create(linearModelName)
    if verbose>0:
        print(f'model={model}, {model.MODEL_NAME}')
        print(f'model={model}')

    params       = model.get_params().copy()
    dim_x, dim_y = params.pop('dim_x'), params.pop('dim_y')
    param        = ParamLinear(verbose, dim_x, dim_y, **params)
    if verbose > 0:
        param.summary()

    # ------------------------------------------------------------------
    # Let's go
    # ------------------------------------------------------------------
    if verbose>0:
        print("\nPKF simulation")
    pkf = Linear_PKF(param, sKey=sKey, save_pickle=traceplot, verbose=verbose)
    
    # Simulate data with the simulator generator
    listData = pkf.simulate_N_data(N=N)
    
    # Save data as a dataframe using pandas
    df = data_to_dataframe(listData, dim_x, dim_y, withoutX=withoutX)
    filepath = os.path.join(datafile_dir, dataFileName)
    save_dataframe_to_csv(df, filepath)
