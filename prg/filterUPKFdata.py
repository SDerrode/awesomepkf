#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import numpy as np

# non linear models 
from models.nonLinear import ModelFactoryNonLinear
# A few utils functions that are used several times
from others.utils import compute_errors
# Manage algorithms for the UPKF
from classes.NonLinear_UPKF import NonLinear_UPKF
# Manage non linear parameters
from classes.ParamNonLinear import ParamNonLinear
# Parser d'options
from others.parser    import *

if __name__ == "__main__":
    """
    USAGES:
        python prg/filterUPKFdata.py
        python prg/filterUPKFdata.py --N 1000 --nonLinearModelName x1_y1_withRetroactions --sKey 303 --verbose 0 --traceplot
    """

    # ------------------------------------------------------------------
    # Constants (default value) - Parser
    # ------------------------------------------------------------------

    parser = argparse.ArgumentParser(description='Simulate and filter non linear data with UPKF')
    addParseToParser(parser, ['nonLinearModelName', 'N', 'sKey'])
    args   = parser.parse_args()
    
    traceplot          = args.traceplot
    verbose            = args.verbose
    N                  = args.N
    sKey               = args.sKey 
    nonLinearModelName = args.nonLinearModelName
    if sKey is not None and sKey < 0:
        parser.error("sKey must be >= 0")
    if N < 200:
        parser.error("N must be >= 200")
    # exit(1)

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
    # Test parameters
    # ------------------------------------------------------------------

    model        = ModelFactoryNonLinear.create(nonLinearModelName)
    params       = model.get_params()
    dim_x, dim_y = params.pop('dim_x'), params.pop('dim_y')
    param        = ParamNonLinear(verbose, dim_x, dim_y, **params)

    if verbose > 0:
        print(f'model={model}')
        param.summary()

    # ------------------------------------------------------------------
    # Let's go
    # ------------------------------------------------------------------

    if verbose > 0:
        print("\nUPKF filtering with data generated from a non-linear model...")
    upkf_1    = NonLinear_UPKF(param, sKey=sKey, save_pickle=traceplot, verbose=verbose)
    listeUPKF = upkf_1.process_N_data(N=N)  # Call with the default data simulator generator

    if traceplot and upkf_1.history is not None:
        df = upkf_1.history.as_dataframe()
        if verbose > 0:
            print("\nExtract of the resulting filtering with UPKF :")
            print(df.head())

        # print scoring
        # ListeA = ['xkp1',           'xkp1']
        # ListeB = ['Xkp1_predict',   'Xkp1_update']
        # ListeC = ['PXXkp1_predict', 'PXXkp1_update']
        ListeA = ['xkp1']
        ListeB = ['Xkp1_update']
        ListeC = ['PXXkp1_update']
        upkf_1.history.compute_errors(ListeA, ListeB, ListeC)

        # pickle storing and plots
        upkf_1.history.save_pickle(os.path.join(tracker_dir, f"history_run_upfk_1.pkl"))
        title = f"'{nonLinearModelName}' model data filtered with UPKF"
        # Les observations
        upkf_1.history.plot(title, 
                            list_param= ["ykp1"], \
                            list_label= ["Observations y"], \
                            list_covar = [None], \
                            # window    = {'xmin': min(20, N), 'xmax': min(min(20, N)+100, N) }, \
                            window    = {'xmin':20, 'xmax': 120 }, \
                            basename  = f'upkf_1_{nonLinearModelName}_observations', show=False, base_dir=graph_dir)
        upkf_1.history.plot(title, 
                            list_param= ["xkp1"  , "Xkp1_update"], \
                            list_label= ["x true", "x estimated"], \
                            list_covar = [None, "PXXkp1_update"], \
                            # window    = {'xmin': min(20, N), 'xmax': min(min(20, N)+100, N) }, \
                            window    = {'xmin':20, 'xmax': 120 }, \
                            basename  = f'upkf_1_{nonLinearModelName}', show=False, base_dir=graph_dir)

