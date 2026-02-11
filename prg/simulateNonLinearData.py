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
# Manage non linear simulations
from classes.NonLinear_PKF import NonLinear_PKF
# Parameters for PKF
from classes.ParamNonLinear import ParamNonLinear
# Parser d'options
from others.parser    import *

if __name__ == "__main__":
    """
    USAGES:
        python3 prg/simulateNonLinearData.py
        python3 prg/simulateNonLinearData.py --N 1000 --verbose 0 --nonLinearModelName "x1_y1_withRetroactions" --sKey 303 --dataFileName "testNL.csv"
    """

    # ------------------------------------------------------------------
    # Constants (default value) - Parser
    # ------------------------------------------------------------------

    parser = argparse.ArgumentParser(description='Simulate non linear data')
    addParseToParser(parser, ['nonLinearModelName', 'dataFileName', 'N', 'sKey', 'withoutX'])
    args   = parser.parse_args()

    verbose            = args.verbose
    withoutX           = args.withoutX
    N                  = args.N
    sKey               = args.sKey
    nonLinearModelName = args.nonLinearModelName
    dataFileName       = args.dataFileName
    if dataFileName is None:
        dataFileName = f"dataNonLinear_{nonLinearModelName}.csv"
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

    # Available non linear models:
    model = ModelFactoryNonLinear.create(nonLinearModelName)
    if verbose > 1:
        print(f'model={model}, {model.MODEL_NAME}')
        print(f'model={model}')

    params       = model.get_params().copy()
    dim_x, dim_y = params.pop('dim_x'), params.pop('dim_y')
    param        = ParamNonLinear(verbose, dim_x, dim_y, **params)
    if verbose > 1:
        param.summary()
 
    # ------------------------------------------------------------------
    # Let's go
    # ------------------------------------------------------------------
    if verbose > 1:
        print("Non Linear data simulation")
    upkf = NonLinear_PKF(param, sKey=sKey, verbose=verbose)
    
    # Simulate data with the simulator generator
    listData = upkf.simulate_N_data(N=N)

    # Save data as a dataframe using pandas
    df = data_to_dataframe(listData, dim_x, dim_y, withoutX=withoutX)
    filepath = os.path.join(datafile_dir, dataFileName)
    save_dataframe_to_csv(df, filepath)
    