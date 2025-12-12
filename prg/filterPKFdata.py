#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import numpy as np

# Linear models 
from models.linear import ModelFactoryLinear
# A few utils functions that are used several times
from others.utils import compute_errors
# Manage algorithms for the PKF
from classes.Linear_PKF import Linear_PKF
# Manage parameters for the PKF
from classes.ParamLinear import ParamLinear
# Parser d'options
from others.parser    import *


if __name__ == "__main__":
    """
    USAGES:
        python prg/filterPKFdata.py
        python prg/filterPKFdata.py --N 1000 --linearModelName A_mQ_x1_y1 --sKey 303 --verbose 0 --traceplot
    """
    
    # ------------------------------------------------------------------
    # Constants (default value) - Parser
    # ------------------------------------------------------------------

    parser = argparse.ArgumentParser(description='Simulate and filter linear data with PKF')
    addParseToParser(parser, ['linearModelName', 'N', 'sKey'])
    args   = parser.parse_args()
    
    traceplot       = args.traceplot
    verbose         = args.verbose
    N               = args.N
    sKey            = args.sKey 
    linearModelName = args.linearModelName
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
    # Test parameters for the Two ((A, mQ) or Sigma) parametrizations
    # ------------------------------------------------------------------
    
    model        = ModelFactoryLinear.create(linearModelName)
    params       = model.get_params().copy()
    dim_x, dim_y = params.pop('dim_x'), params.pop('dim_y')
    param        = ParamLinear(verbose, dim_x, dim_y, **params)
    
    if verbose > 0:
        print(f'model={model}')
        param.summary()

    # ------------------------------------------------------------------
    # Let's go
    # ------------------------------------------------------------------

    if verbose > 0:
        print("\nPKF filtering with data generated from a linear model...")
    pkf_1   = Linear_PKF(param, sKey=sKey, save_pickle=traceplot, verbose=verbose)
    listePKF = pkf_1.process_N_data(N=N)

    if traceplot and pkf_1.history is not None:
        df = pkf_1.history.as_dataframe()
        if verbose > 0:
            print("\nExtract of the resulting filtering with PKF :")
            print(df.head())

        # print scoring
        # ListeA = ['xkp1',           'xkp1']
        # ListeB = ['Xkp1_predict',   'Xkp1_update']
        # ListeC = ['PXXkp1_predict', 'PXXkp1_update']
        ListeA = ['xkp1']
        ListeB = ['Xkp1_update']
        ListeC = ['PXXkp1_update']
        pkf_1.history.compute_errors(ListeA, ListeB, ListeC)

        # pickle storing and plots
        pkf_1.history.save_pickle(os.path.join(tracker_dir, f"history_run_pfk_1.pkl"))
        title = f"'{linearModelName}' model data filtered with PKF"
        pkf_1.history.plot(title, 
                            list_param= ["ykp1"], \
                            list_label= ["Observations y"], \
                            list_covar= [None], \
                            # window    = {'xmin': min(20, N), 'xmax': min(min(20, N)+100, N) }, \
                            window    = {'xmin':20, 'xmax': 120 }, \
                            basename  = f'pkf_1_{linearModelName}_observations', show=False, base_dir=graph_dir)
        pkf_1.history.plot(title, 
                            list_param = ["xkp1"  , "Xkp1_update"], \
                            list_label = ["x true", "x estimated"], \
                            list_covar = [None, "PXXkp1_update"], \
                            # window    = {'xmin': min(20, N), 'xmax': min(min(20, N)+100, N) }, \
                            window    = {'xmin':20, 'xmax': 120 }, \
                            basename  = f'pkf_1_{linearModelName}', show=False, base_dir=graph_dir)
